"""
受信任数据源注册中心与采集证据规约 (data/source_registry.py)
核心原则：
1. TRUSTED SOURCE REGISTRY: 仅允许已在注册中心备案的机构与规范域名 (CSI / SSE / SZSE / WIND / CHOICE)。
2. OPERATOR ATTESTATION: 针对 LICENSED_VENDOR 的人工导出凭据实施 Ed25519 密码学验签，严禁明文 mock 字符串通过。
3. EXACT BINDING: 原始证据元数据必须与物理文件、下载回执及哈希值保持双向强一致。
4. FAIL-CLOSED: 任何未登记来源、非 HTTPS 协议、公钥用途不符或篡改一律判定为 UNKNOWN / FAIL。
"""
import re
import json
import hashlib
from urllib.parse import urlparse
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

from data.crypto_anchor import (
    TRUSTED_KEY_REGISTRY,
    TRUSTED_OPERATOR_REGISTRY,
    DOMAIN_SEPARATOR_ACQUISITION,
    DOMAIN_SEPARATOR_OPERATOR,
    DOMAIN_SEPARATOR_CORPORATE_ACTION,
    verify_ed25519_signature,
    sign_with_environment_key
)

TRUSTED_ACQUISITION_KEYS = TRUSTED_KEY_REGISTRY


# =========================================================================
# 1. 受信任数据源注册表 (Trusted Source Registry)
# =========================================================================

TRUSTED_SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "CSI": {
        "source_class": "OFFICIAL_PRIMARY",
        "canonical_source_name": "China Securities Index Co., Ltd. (中证指数有限公司)",
        "allowed_domains": ["csindex.com.cn", "www.csindex.com.cn"],
        "allowed_evidence_types": ["INDEX_CONSTITUENT_ADJUSTMENT", "BASELINE_SNAPSHOT", "SECURITY_MASTER", "CALENDAR"]
    },
    "SSE": {
        "source_class": "OFFICIAL_PRIMARY",
        "canonical_source_name": "Shanghai Stock Exchange (上海证券交易所)",
        "allowed_domains": ["sse.com.cn", "www.sse.com.cn", "query.sse.com.cn"],
        "allowed_evidence_types": ["SECURITY_MASTER", "CALENDAR", "CORPORATE_ACTION"]
    },
    "SZSE": {
        "source_class": "OFFICIAL_PRIMARY",
        "canonical_source_name": "Shenzhen Stock Exchange (深圳证券交易所)",
        "allowed_domains": ["szse.cn", "www.szse.cn"],
        "allowed_evidence_types": ["SECURITY_MASTER", "CALENDAR", "CORPORATE_ACTION"]
    },
    "WIND": {
        "source_class": "LICENSED_VENDOR",
        "canonical_source_name": "Wind Financial Terminal (万得信息技术股份有限公司)",
        "allowed_domains": ["wind.com.cn", "www.wind.com.cn", "api.wind.com.cn"],
        "allowed_evidence_types": ["INDEX_CONSTITUENT_ADJUSTMENT", "BASELINE_SNAPSHOT", "SECURITY_MASTER", "CALENDAR", "CORPORATE_ACTION"]
    },
    "CHOICE": {
        "source_class": "LICENSED_VENDOR",
        "canonical_source_name": "EastMoney Choice Terminal (东方财富信息股份有限公司)",
        "allowed_domains": ["eastmoney.com", "choice.eastmoney.com"],
        "allowed_evidence_types": ["INDEX_CONSTITUENT_ADJUSTMENT", "BASELINE_SNAPSHOT", "SECURITY_MASTER", "CALENDAR", "CORPORATE_ACTION"]
    }
}


def validate_trusted_url(url: Optional[str], allowed_domains: List[str]) -> Tuple[bool, List[str]]:
    """严格校验官方/持牌来源 URL 的协议、域名及防伪装特征"""
    if not url:
        return False, ["missing_source_url"]
    try:
        parsed = urlparse(str(url).strip())
        if parsed.scheme.lower() != "https":
            return False, [f"insecure_url_scheme_{parsed.scheme}_only_https_allowed"]
        
        hostname = parsed.hostname
        if not hostname:
            return False, ["missing_hostname_in_url"]

        if parsed.username or parsed.password or "@" in (parsed.netloc or ""):
            return False, ["suspicious_userinfo_or_at_symbol_in_url"]

        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname):
            return False, ["ip_literal_host_rejected_for_official_source"]

        host_lower = hostname.lower()
        matched = False
        for allowed in allowed_domains:
            allowed_lower = allowed.lower()
            if host_lower == allowed_lower or host_lower.endswith("." + allowed_lower):
                matched = True
                break

        if not matched:
            return False, [f"domain_{hostname}_not_in_allowed_domains_{allowed_domains}"]

        return True, []
    except Exception as e:
        return False, [f"url_validation_error_{str(e)}"]


def extract_domain(url: Optional[str]) -> Optional[str]:
    """从 URL 中解析提取小写域名 Hostname"""
    if not url:
        return None
    try:
        parsed = urlparse(str(url).strip())
        return parsed.hostname.lower() if parsed.hostname else None
    except Exception:
        return None


# =========================================================================
# 2. 持牌供应商操作员鉴证实体 (OperatorAttestation - Ed25519 Protected)
# =========================================================================

@dataclass
class OperatorAttestation:
    """持牌供应商 (Wind/Choice) 终端操作员人工导出凭据实体 (P0 Ed25519 签名防护)"""
    operator_id: str
    vendor_source_id: str
    terminal_reference: str
    exported_at: str
    raw_sha256: str
    signing_key_id: str
    signature: str

    def compute_canonical_payload(self) -> Dict[str, Any]:
        """确定性计算操作员鉴证 Payload"""
        return {
            "exported_at": str(self.exported_at),
            "operator_id": str(self.operator_id),
            "raw_sha256": str(self.raw_sha256).lower(),
            "terminal_reference": str(self.terminal_reference),
            "vendor_source_id": str(self.vendor_source_id).upper()
        }

    def compute_canonical_bytes(self) -> bytes:
        payload = self.compute_canonical_payload()
        sorted_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        return sorted_json.encode('utf-8')

    def verify_against_file(self, raw_file_path: Path) -> Tuple[bool, List[str]]:
        """对操作员凭据进行注册合法性与 Ed25519 密码学非对称验签"""
        errors = []

        # 1. 检验 operator_id 登记状态
        if self.operator_id not in TRUSTED_OPERATOR_REGISTRY:
            return False, [f"unregistered_operator_id_{self.operator_id}"]

        op_info = TRUSTED_OPERATOR_REGISTRY[self.operator_id]
        if op_info.get("status") != "ACTIVE":
            return False, [f"operator_status_is_{op_info.get('status')}"]

        # 2. vendor_source_id 一致性
        if str(self.vendor_source_id).upper() != str(op_info.get("vendor_source_id")).upper():
            errors.append(f"operator_vendor_mismatch_{self.vendor_source_id}_vs_{op_info.get('vendor_source_id')}")

        # 3. signing_key_id 一致性
        if str(self.signing_key_id) != str(op_info.get("signing_key_id")):
            errors.append(f"operator_signing_key_mismatch_{self.signing_key_id}_vs_{op_info.get('signing_key_id')}")

        # 4. 物理文件存在性与哈希核验
        if not raw_file_path.exists():
            return False, [f"raw_file_missing_{raw_file_path.name}"]

        h = hashlib.sha256()
        with open(raw_file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_hash = h.hexdigest()

        if actual_hash.lower() != str(self.raw_sha256).lower():
            errors.append(f"operator_raw_sha256_mismatch_{self.raw_sha256}_vs_{actual_hash}")

        # 5. Ed25519 非对称密码学验签 (严禁任何 operator_signature='hello' 绕过)
        msg_bytes = self.compute_canonical_bytes()
        sig_ok, sig_errs = verify_ed25519_signature(
            message=msg_bytes,
            signature_hex=self.signature,
            key_id=self.signing_key_id,
            required_purpose="LICENSED_VENDOR_OPERATOR_ATTESTATION",
            domain_separator=DOMAIN_SEPARATOR_OPERATOR,
            created_at_iso=self.exported_at
        )
        if not sig_ok:
            errors.extend(sig_errs)

        return len(errors) == 0, errors


# =========================================================================
# 3. 数据采集凭证回执 (AcquisitionReceipt - Ed25519 Protected)
# =========================================================================

@dataclass
class AcquisitionReceipt:
    """受信任数据同步回执 (Acquisition Receipt - 包含信任锚点鉴真)"""
    receipt_id: str
    source_id: str
    source_url: str
    requested_at: str
    downloaded_at: str
    http_status: int = 200
    content_length: int = 0
    raw_sha256: str = ""
    original_filename: str = ""
    receipt_integrity_digest: Optional[str] = None  # 本地 JSON 完整性校验摘要 (不作为数字签名)
    trust_anchor_type: str = "SELF_DIGEST"          # TRUSTED_KEY_ATTESTATION / OPERATOR_ATTESTED / SELF_DIGEST
    signing_key_id: Optional[str] = None
    attestation_signature: Optional[str] = None     # 由受信任私钥对完整性摘要签署的 Ed25519 签名
    operator_attestation: Optional[Union[Dict[str, Any], OperatorAttestation]] = None
    trust_anchor_verified: bool = False

    def compute_integrity_digest(self) -> str:
        """确定性计算回执的完整性哈希 (Canonical Integrity Digest)"""
        raw_str = (
            f"{self.receipt_id}|{self.source_id}|{self.source_url}|"
            f"{self.requested_at}|{self.downloaded_at}|{self.http_status}|"
            f"{self.content_length}|{self.raw_sha256.lower()}|{self.original_filename}"
        )
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def verify_against_file(self, raw_file_path: Path) -> Tuple[bool, List[str]]:
        """严格核验采集回执内容、物理文件哈希及独立信任锚点签名"""
        errors = []
        if not raw_file_path.exists():
            return False, [f"raw_file_not_found_{raw_file_path.name}"]

        # 1. 验证 source_id 注册与规范名称
        if self.source_id not in TRUSTED_SOURCE_REGISTRY:
            errors.append(f"source_id_{self.source_id}_not_in_trusted_registry")
            return False, errors

        reg_info = TRUSTED_SOURCE_REGISTRY[self.source_id]

        # 2. 严格 HTTPS URL 校验
        url_ok, url_errs = validate_trusted_url(self.source_url, reg_info.get("allowed_domains", []))
        if not url_ok:
            errors.extend(url_errs)

        # 3. 验证 HTTP 状态码
        if self.http_status != 200:
            errors.append(f"invalid_http_status_{self.http_status}")

        # 4. 验证实体文件哈希
        h = hashlib.sha256()
        with open(raw_file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_hash = h.hexdigest()

        if actual_hash.lower() != self.raw_sha256.lower():
            errors.append(f"receipt_hash_mismatch_{self.raw_sha256}_vs_{actual_hash}")

        # 5. 验证原始文件名
        if self.original_filename and self.original_filename != raw_file_path.name:
            errors.append(f"receipt_filename_mismatch_{self.original_filename}_vs_{raw_file_path.name}")

        # 6. 验证完整性摘要
        computed_digest = self.compute_integrity_digest()
        if self.receipt_integrity_digest and self.receipt_integrity_digest.lower() != computed_digest.lower():
            errors.append("receipt_integrity_digest_tampered")

        # 7. 独立 Trust Anchor 非对称密码学数字签名校验 (P0 Trust Anchor Guard)
        if self.trust_anchor_type == "TRUSTED_KEY_ATTESTATION":
            if not self.signing_key_id or self.signing_key_id not in TRUSTED_KEY_REGISTRY:
                errors.append(f"untrusted_or_missing_signing_key_id_{self.signing_key_id}")
                self.trust_anchor_verified = False
            elif not self.attestation_signature:
                errors.append("missing_attestation_signature_for_trusted_key")
                self.trust_anchor_verified = False
            else:
                sig_ok, sig_errs = verify_ed25519_signature(
                    message=computed_digest.encode("utf-8"),
                    signature_hex=self.attestation_signature,
                    key_id=self.signing_key_id,
                    required_purpose="ACQUISITION_RECEIPT",
                    domain_separator=DOMAIN_SEPARATOR_ACQUISITION,
                    created_at_iso=self.downloaded_at
                )
                if sig_ok:
                    self.trust_anchor_verified = True
                else:
                    errors.extend(sig_errs)
                    self.trust_anchor_verified = False

        elif self.trust_anchor_type == "OPERATOR_ATTESTED":
            # 针对 LICENSED_VENDOR (Wind/Choice) 的终端人工导出凭据真验签
            if not self.operator_attestation:
                errors.append("missing_operator_attestation_for_licensed_vendor")
                self.trust_anchor_verified = False
            else:
                if isinstance(self.operator_attestation, dict):
                    try:
                        op_att = OperatorAttestation(**self.operator_attestation)
                    except Exception as e:
                        errors.append(f"invalid_operator_attestation_payload_{str(e)}")
                        op_att = None
                elif isinstance(self.operator_attestation, OperatorAttestation):
                    op_att = self.operator_attestation
                else:
                    errors.append("unsupported_operator_attestation_object_type")
                    op_att = None

                if op_att:
                    op_ok, op_errs = op_att.verify_against_file(raw_file_path)
                    if op_ok:
                        self.trust_anchor_verified = True
                    else:
                        errors.extend(op_errs)
                        self.trust_anchor_verified = False
                else:
                    self.trust_anchor_verified = False
        else:
            # 默认 SELF_DIGEST：属于自签名/本地无权威证明状态，不能获得生产最高认证
            self.trust_anchor_verified = False

        return len(errors) == 0, errors

    def verify_exact_binding(self, source_meta: Any, raw_file_path: Path) -> Tuple[bool, List[str]]:
        """严格核验 SourceMetadata 与 AcquisitionReceipt 的双向强绑定 (P0)"""
        errors = []
        if str(self.source_id).strip().upper() != str(getattr(source_meta, "source_id", "")).strip().upper():
            errors.append(f"binding_mismatch_source_id_{self.source_id}_vs_{getattr(source_meta, 'source_id', '')}")

        if str(self.source_url).strip() != str(getattr(source_meta, "source_url", "")).strip():
            errors.append(f"binding_mismatch_source_url_{self.source_url}_vs_{getattr(source_meta, 'source_url', '')}")

        if str(self.raw_sha256).strip().lower() != str(getattr(source_meta, "sha256", "")).strip().lower():
            errors.append(f"binding_mismatch_sha256_{self.raw_sha256}_vs_{getattr(source_meta, 'sha256', '')}")

        if str(self.original_filename).strip() != str(getattr(source_meta, "original_filename", "")).strip():
            errors.append(f"binding_mismatch_filename_{self.original_filename}_vs_{getattr(source_meta, 'original_filename', '')}")

        return len(errors) == 0, errors

    @classmethod
    def load_from_file(cls, receipt_path: Path) -> Tuple[Optional["AcquisitionReceipt"], List[str]]:
        """从 JSON 文件加载采集回执"""
        if not receipt_path.exists():
            return None, [f"missing_acquisition_receipt_{receipt_path.name}"]
        try:
            with open(receipt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "receipt_signature_hash" in data and "receipt_integrity_digest" not in data:
                data["receipt_integrity_digest"] = data.pop("receipt_signature_hash")
            receipt = cls(**data)
            return receipt, []
        except Exception as e:
            return None, [f"corrupted_acquisition_receipt_{receipt_path.name}_{str(e)}"]


# =========================================================================
# 4. 公司行为审计证据实体 (CorporateActionCoverageEvidence - Fail-Closed)
# =========================================================================

@dataclass
class CorporateActionCoverageEvidence:
    """公司行为覆盖度与零事件审计证据实体 (Fail-Closed Default - P0 Hardened)"""
    symbol: str
    query_start: str
    query_end: str
    source_id: str = "UNKNOWN"
    source_reference: Optional[str] = None
    query_success: bool = False  # 严格默认为 False
    raw_result_hash: Optional[str] = None
    raw_result_file: Optional[str] = None
    response_hash: Optional[str] = None
    response_file: Optional[str] = None  # 必须绑定物理响应证明文件
    queried_at: str = ""
    empty_result: bool = False
    empty_result_verified: bool = False  # 必须严格默认为 False (Fail-Closed)
    source_class: str = "UNKNOWN"
    evidence_manifest_hash: Optional[str] = None
    production_eligible: bool = True     # Evidence 默认具备进入生产认证的候选资格

    def is_valid_zero_event_proof(
        self,
        target_symbol: str,
        backtest_start: str,
        backtest_end: str,
        evidence_dir: Optional[Path] = None
    ) -> Tuple[bool, List[str]]:
        """检验该证据是否足以证明指定标的在指定回测区间内确实不存在任何除权除息事件"""
        errors = []

        if not self.production_eligible:
            errors.append("evidence_marked_non_production_eligible")

        if self.symbol.strip().upper() != target_symbol.strip().upper():
            errors.append(f"symbol_mismatch_{self.symbol}_vs_{target_symbol}")

        if not self.query_success:
            errors.append("query_not_successful")

        if not self.empty_result:
            errors.append("has_corporate_actions_cannot_claim_zero_event")

        if not self.empty_result_verified:
            errors.append("empty_result_not_verified_by_audit")

        if not self.raw_result_hash or not self.response_hash:
            errors.append("missing_raw_response_hash")

        if self.source_id not in TRUSTED_SOURCE_REGISTRY:
            errors.append(f"untrusted_source_id_{self.source_id}")

        if self.query_start > backtest_start:
            errors.append(f"query_start_{self.query_start}_after_backtest_start_{backtest_start}")

        if self.query_end < backtest_end:
            errors.append(f"query_end_{self.query_end}_before_backtest_end_{backtest_end}")

        # 生产环境强制要求 evidence_dir 与真实物理文件
        if not evidence_dir:
            errors.append("evidence_dir_missing_required_for_production_verification")
        else:
            evidence_p = Path(evidence_dir)
            # 1. 校验 raw_result_file
            if not self.raw_result_file:
                errors.append("missing_raw_result_file_in_evidence")
            else:
                raw_p = evidence_p / self.raw_result_file
                if not raw_p.exists():
                    errors.append(f"corporate_action_raw_result_file_missing_{self.raw_result_file}")
                else:
                    h = hashlib.sha256()
                    with open(raw_p, "rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
                    if h.hexdigest().lower() != str(self.raw_result_hash).lower():
                        errors.append(f"corporate_action_raw_file_hash_mismatch_{self.raw_result_hash}_vs_{h.hexdigest()}")

            # 2. 校验 response_file
            if not self.response_file:
                errors.append("missing_response_file_in_evidence")
            else:
                resp_p = evidence_p / self.response_file
                if not resp_p.exists():
                    errors.append(f"corporate_action_response_file_missing_{self.response_file}")
                else:
                    h_resp = hashlib.sha256()
                    with open(resp_p, "rb") as f:
                        while chunk := f.read(65536):
                            h_resp.update(chunk)
                    if h_resp.hexdigest().lower() != str(self.response_hash).lower():
                        errors.append(f"corporate_action_response_file_hash_mismatch_{self.response_hash}_vs_{h_resp.hexdigest()}")

        return len(errors) == 0, errors
