"""
受信任权威数据源注册表与数据获取回执 (data/source_registry.py)
核心原则：
1. 真实非对称签名：支持项目可信采集签名 (Ed25519) 与持牌机构终端操作员签名 (OperatorAttestation)。
2. 完整信任链：SourceMetadata + AcquisitionReceipt + Trust Anchor + Physical SHA256 双向强绑定。
3. 严格路径隔离：所有证据文件解析均强制经过 safe_resolve_path 进行路径穿越防御。
4. 命名准确性：机构名称明确为 PROJECT_REGISTERED_VENDOR_OPERATOR / PROJECT_TRUSTED_DOWNLOADER_AUTHORITY，绝不冒充交易所官方数字签名。
"""
import re
import json
import hashlib
from urllib.parse import urlparse
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd

from data.crypto_anchor import (
    TRUSTED_KEY_REGISTRY,
    TRUSTED_OPERATOR_REGISTRY,
    DOMAIN_SEPARATOR_ACQUISITION,
    DOMAIN_SEPARATOR_OPERATOR,
    verify_ed25519_signature,
    safe_resolve_path
)

# 仅允许用于数据采集鉴证的公钥子集 (保持向后兼容)
TRUSTED_ACQUISITION_KEYS = {
    k: v for k, v in TRUSTED_KEY_REGISTRY.items()
    if "ACQUISITION_RECEIPT" in v.get("allowed_purposes", [])
}


# =========================================================================
# 1. 注册受信任数据源 (TRUSTED SOURCE REGISTRY)
# =========================================================================

TRUSTED_SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "CSI": {
        "canonical_source_name": "China Securities Index Co., Ltd.",
        "short_name": "中证指数官网",
        "source_class": "OFFICIAL_PRIMARY",
        "allowed_domains": [
            "csindex.com.cn",
            "www.csindex.com.cn"
        ],
        "allowed_evidence_types": [
            "CONSTITUENT_CHANGE",
            "CONSTITUENT_WEIGHT",
            "CORPORATE_ACTION",
            "INDEX_DAILY",
            "BASELINE_SNAPSHOT",
            "INDEX_CONSTITUENT_ADJUSTMENT"
        ]
    },
    "SSE": {
        "canonical_source_name": "Shanghai Stock Exchange",
        "short_name": "上海证券交易所",
        "source_class": "OFFICIAL_PRIMARY",
        "allowed_domains": [
            "sse.com.cn",
            "www.sse.com.cn",
            "query.sse.com.cn"
        ],
        "allowed_evidence_types": [
            "CORPORATE_ACTION",
            "LISTING_NOTICE",
            "DELISTING_NOTICE",
            "ST_NOTICE",
            "TRADING_CALENDAR"
        ]
    },
    "SZSE": {
        "canonical_source_name": "Shenzhen Stock Exchange",
        "short_name": "深圳证券交易所",
        "source_class": "OFFICIAL_PRIMARY",
        "allowed_domains": [
            "szse.cn",
            "www.szse.cn"
        ],
        "allowed_evidence_types": [
            "CORPORATE_ACTION",
            "LISTING_NOTICE",
            "DELISTING_NOTICE",
            "ST_NOTICE",
            "TRADING_CALENDAR"
        ]
    },
    "WIND": {
        "canonical_source_name": "Wind Information Co., Ltd.",
        "short_name": "万得金融终端",
        "source_class": "LICENSED_VENDOR",
        "allowed_domains": [
            "wind.com.cn",
            "www.wind.com.cn"
        ],
        "allowed_evidence_types": [
            "PIT_UNIVERSE_DATASET",
            "HISTORICAL_ST_TIMELINE",
            "CORPORATE_ACTION_DATASET",
            "TRADING_CALENDAR"
        ]
    },
    "CHOICE": {
        "canonical_source_name": "Eastmoney Choice Financial Terminal",
        "short_name": "东方财富 Choice 终端",
        "source_class": "LICENSED_VENDOR",
        "allowed_domains": [
            "choice.eastmoney.com",
            "eastmoney.com"
        ],
        "allowed_evidence_types": [
            "PIT_UNIVERSE_DATASET",
            "HISTORICAL_ST_TIMELINE",
            "CORPORATE_ACTION_DATASET",
            "TRADING_CALENDAR"
        ]
    }
}


def extract_domain(url: str) -> Optional[str]:
    """严格解析 URL 中的域名"""
    try:
        parsed = urlparse(url.strip())
        if not parsed.scheme or not parsed.netloc:
            return None
        hostname = parsed.hostname
        if not hostname:
            return None
        return hostname.lower().strip()
    except Exception:
        return None


def validate_trusted_url(url: str, allowed_domains: List[str]) -> Tuple[bool, List[str]]:
    """严格校验 URL 合法性与域名白名单 (拒绝 HTTP、子域名仿冒、IP 与 UserInfo 混淆)"""
    errors = []
    if not url:
        return False, ["url_is_empty"]

    clean_url = str(url).strip()
    try:
        parsed = urlparse(clean_url)
    except Exception as e:
        return False, [f"url_parse_error_{str(e)}"]

    if parsed.scheme.lower() != "https":
        errors.append(f"insecure_url_scheme_{parsed.scheme}_must_be_https")

    if parsed.username or parsed.password:
        errors.append("url_contains_userinfo_disallowed")

    hostname = parsed.hostname
    if not hostname:
        errors.append("url_missing_hostname")
        return False, errors

    hostname = hostname.lower().strip()

    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname) or ":" in hostname:
        errors.append(f"ip_literal_host_disallowed_{hostname}")

    matched = False
    for allowed in allowed_domains:
        allowed = allowed.lower().strip()
        if hostname == allowed or hostname.endswith(f".{allowed}"):
            matched = True
            break

    if not matched:
        errors.append(f"untrusted_domain_{hostname}_not_in_allowed_{allowed_domains}")

    return len(errors) == 0, errors


# =========================================================================
# 2. 终端操作员凭据实体 (OperatorAttestation - Ed25519 Verified)
# =========================================================================

@dataclass
class OperatorAttestation:
    """针对商业终端 (Wind/Choice) 人工导出数据的操作员数字签名凭据"""
    operator_id: str
    vendor_source_id: str
    terminal_reference: str
    exported_at: str
    raw_sha256: str
    signing_key_id: str
    signature: str

    def compute_canonical_payload(self) -> Dict[str, Any]:
        """生成规范化字典"""
        return {
            "exported_at": self.exported_at,
            "operator_id": self.operator_id,
            "raw_sha256": self.raw_sha256.lower().strip(),
            "signing_key_id": self.signing_key_id,
            "terminal_reference": self.terminal_reference,
            "vendor_source_id": self.vendor_source_id
        }

    def compute_canonical_bytes(self) -> bytes:
        """确定性序列化签名载荷字节流"""
        sorted_json = json.dumps(self.compute_canonical_payload(), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        return sorted_json.encode("utf-8")

    def verify_against_file(self, raw_file_path: Path) -> Tuple[bool, List[str]]:
        """全方位校验操作员注册状态、物理文件哈希及 Ed25519 数字签名"""
        errors = []
        if not raw_file_path.exists():
            return False, [f"operator_attestation_target_file_missing_{raw_file_path.name}"]

        # 1. 验证操作员注册状态
        if self.operator_id not in TRUSTED_OPERATOR_REGISTRY:
            errors.append(f"unregistered_operator_id_{self.operator_id}")
            return False, errors

        op_info = TRUSTED_OPERATOR_REGISTRY[self.operator_id]
        if op_info.get("status") != "ACTIVE":
            errors.append(f"operator_{self.operator_id}_status_is_{op_info.get('status')}")

        if op_info.get("vendor_source_id") != self.vendor_source_id:
            errors.append(f"operator_vendor_mismatch_{op_info.get('vendor_source_id')}_vs_{self.vendor_source_id}")

        if op_info.get("signing_key_id") != self.signing_key_id:
            errors.append(f"operator_key_id_mismatch_{op_info.get('signing_key_id')}_vs_{self.signing_key_id}")

        # 2. 验证物理文件哈希
        h = hashlib.sha256()
        with open(raw_file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_sha = h.hexdigest()

        if actual_sha.lower() != self.raw_sha256.lower():
            errors.append(f"operator_attestation_file_hash_mismatch_{self.raw_sha256}_vs_{actual_sha}")

        # 3. 校验 Ed25519 签名
        msg_bytes = self.compute_canonical_bytes()
        sig_ok, sig_errs = verify_ed25519_signature(
            message=msg_bytes,
            signature_hex=self.signature,
            key_id=self.signing_key_id,
            required_purpose="LICENSED_VENDOR_OPERATOR_ATTESTATION",
            domain_separator=DOMAIN_SEPARATOR_OPERATOR,
            created_at_iso=self.exported_at,
            production_mode=True
        )
        if not sig_ok:
            errors.extend(sig_errs)

        return len(errors) == 0, errors


# =========================================================================
# 3. 数据采集凭证回执 (AcquisitionReceipt - Ed25519 Protected)
# =========================================================================

@dataclass
class AcquisitionReceipt:
    """官方原始证据下载凭据回执 (Trust Anchor Protected)"""
    receipt_id: str
    source_id: str
    source_url: str
    requested_at: str
    downloaded_at: str
    http_status: int = 200
    content_length: int = 0
    raw_sha256: str = ""
    original_filename: str = ""
    query_context: Optional[Dict[str, Any]] = None
    receipt_integrity_digest: Optional[str] = None
    trust_anchor_type: str = "SELF_DIGEST"
    signing_key_id: Optional[str] = None
    attestation_signature: Optional[str] = None
    operator_attestation: Optional[Union[Dict[str, Any], OperatorAttestation]] = None
    trust_anchor_verified: bool = False

    def compute_integrity_digest(self) -> str:
        """确定性计算回执的完整性哈希 (Canonical Integrity Digest，强制绑定 query_context)"""
        qc_str = json.dumps(self.query_context, sort_keys=True, separators=(',', ':')) if self.query_context else ""
        raw_str = (
            f"{self.receipt_id}|{self.source_id}|{self.source_url}|"
            f"{self.requested_at}|{self.downloaded_at}|{self.http_status}|"
            f"{self.content_length}|{self.raw_sha256.lower()}|{self.original_filename}|{qc_str}"
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

        # 7. 独立 Trust Anchor 非对称密码学数字签名校验
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
                    created_at_iso=self.downloaded_at,
                    production_mode=True
                )
                if sig_ok:
                    self.trust_anchor_verified = True
                else:
                    errors.extend(sig_errs)
                    self.trust_anchor_verified = False

        elif self.trust_anchor_type == "OPERATOR_ATTESTED":
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
# 4. 公司行为响应独立解析器与审计证据实体 (Fail-Closed)
# =========================================================================

HEX_64_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass
class CorporateActionParseResult:
    """公司行为响应解析结果强类型实体 (P0 Fail-Closed)"""
    parser_name: str
    schema_id: str
    parse_success: bool = False
    query_success: bool = False
    symbol: str = ""
    query_start: str = ""
    query_end: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)
    event_count: int = 0
    response_status: str = "UNKNOWN"
    failed_checks: List[str] = field(default_factory=list)


class SSECorporateActionParser:
    """上交所官方企业行为 JSON 响应规范解析器"""
    @classmethod
    def parse(cls, content_bytes: bytes, symbol: str = "", query_start: str = "", query_end: str = "") -> CorporateActionParseResult:
        res = CorporateActionParseResult(parser_name="SSECorporateActionParser", schema_id="SSE_CORP_ACTION_V1", symbol=symbol, query_start=query_start, query_end=query_end)
        if not content_bytes or len(content_bytes.strip()) == 0:
            res.failed_checks.append("response_empty_or_truncated")
            return res
        try:
            trimmed = content_bytes.strip()
            data = json.loads(trimmed.decode("utf-8"))
            if not isinstance(data, (dict, list)):
                res.failed_checks.append("unrecognized_corporate_action_response_schema_not_dict_or_list")
                return res
            if isinstance(data, dict):
                if len(data) == 0:
                    res.failed_checks.append("unrecognized_corporate_action_response_schema_empty_dict")
                    return res
                if data.get("error") or data.get("message") == "system busy" or (data.get("code") not in (0, None, 200) and "code" in data):
                    err_msg = data.get("error") or data.get("message") or data.get("code")
                    res.failed_checks.append(f"corporate_action_api_error_{err_msg}")
                    res.response_status = "API_ERROR"
                    return res
                if data.get("success") is False:
                    res.failed_checks.append("corporate_action_api_success_false")
                    res.response_status = "API_FAILED"
                    return res
                raw_events = None
                for k in ["data", "result", "items", "events", "rows"]:
                    if k in data and isinstance(data[k], list):
                        raw_events = data[k]
                        break
                if raw_events is None:
                    if "symbol" in data and "ex_date" in data:
                        raw_events = [data]
                    else:
                        res.failed_checks.append("unrecognized_corporate_action_response_schema_missing_data_field")
                        return res
                res.events = raw_events
                res.event_count = len(raw_events)
                res.parse_success = True
                res.query_success = True
                res.response_status = "SUCCESS"
            elif isinstance(data, list):
                res.events = data
                res.event_count = len(data)
                res.parse_success = True
                res.query_success = True
                res.response_status = "SUCCESS"
        except Exception as e:
            res.failed_checks.append(f"response_parsing_error_{str(e)}")
        return res


class SZSECorporateActionParser:
    """深交所官方企业行为 JSON 响应规范解析器"""
    @classmethod
    def parse(cls, content_bytes: bytes, symbol: str = "", query_start: str = "", query_end: str = "") -> CorporateActionParseResult:
        res = CorporateActionParseResult(parser_name="SZSECorporateActionParser", schema_id="SZSE_CORP_ACTION_V1", symbol=symbol, query_start=query_start, query_end=query_end)
        if not content_bytes or len(content_bytes.strip()) == 0:
            res.failed_checks.append("response_empty_or_truncated")
            return res
        try:
            trimmed = content_bytes.strip()
            data = json.loads(trimmed.decode("utf-8"))
            if not isinstance(data, (dict, list)):
                res.failed_checks.append("unrecognized_corporate_action_response_schema_not_dict_or_list")
                return res
            if isinstance(data, dict):
                if len(data) == 0:
                    res.failed_checks.append("unrecognized_corporate_action_response_schema_empty_dict")
                    return res
                if data.get("error") or data.get("message") == "system busy" or (data.get("code") not in (0, None, 200) and "code" in data):
                    err_msg = data.get("error") or data.get("message") or data.get("code")
                    res.failed_checks.append(f"corporate_action_api_error_{err_msg}")
                    res.response_status = "API_ERROR"
                    return res
                if data.get("success") is False:
                    res.failed_checks.append("corporate_action_api_success_false")
                    res.response_status = "API_FAILED"
                    return res
                raw_events = None
                for k in ["data", "result", "items", "events", "rows"]:
                    if k in data and isinstance(data[k], list):
                        raw_events = data[k]
                        break
                if raw_events is None:
                    if "symbol" in data and "ex_date" in data:
                        raw_events = [data]
                    else:
                        res.failed_checks.append("unrecognized_corporate_action_response_schema_missing_data_field")
                        return res
                res.events = raw_events
                res.event_count = len(raw_events)
                res.parse_success = True
                res.query_success = True
                res.response_status = "SUCCESS"
            elif isinstance(data, list):
                res.events = data
                res.event_count = len(data)
                res.parse_success = True
                res.query_success = True
                res.response_status = "SUCCESS"
        except Exception as e:
            res.failed_checks.append(f"response_parsing_error_{str(e)}")
        return res


class WindCorporateActionParser:
    """万得终端企业行为导出解析器"""
    @classmethod
    def parse(cls, content_bytes: bytes, symbol: str = "", query_start: str = "", query_end: str = "") -> CorporateActionParseResult:
        res = CorporateActionParseResult(parser_name="WindCorporateActionParser", schema_id="WIND_CORP_ACTION_V1", symbol=symbol, query_start=query_start, query_end=query_end)
        if not content_bytes or len(content_bytes.strip()) == 0:
            res.failed_checks.append("response_empty_or_truncated")
            return res
        try:
            trimmed = content_bytes.strip()
            if trimmed.startswith(b"{") or trimmed.startswith(b"["):
                data = json.loads(trimmed.decode("utf-8"))
                if isinstance(data, dict):
                    if len(data) == 0:
                        res.failed_checks.append("unrecognized_corporate_action_response_schema_empty_dict")
                        return res
                    if data.get("error") or data.get("ErrorCode", 0) != 0:
                        res.failed_checks.append("corporate_action_wind_error")
                        return res
                    raw_events = data.get("Data", data.get("data", []))
                    if isinstance(raw_events, list):
                        res.events = raw_events
                        res.event_count = len(raw_events)
                        res.parse_success = True
                        res.query_success = True
                    else:
                        res.failed_checks.append("unrecognized_corporate_action_response_schema")
                elif isinstance(data, list):
                    res.events = data
                    res.event_count = len(data)
                    res.parse_success = True
                    res.query_success = True
            else:
                return CSVCorporateActionParser.parse(content_bytes, symbol=symbol, query_start=query_start, query_end=query_end)
        except Exception as e:
            res.failed_checks.append(f"response_parsing_error_{str(e)}")
        return res


class ChoiceCorporateActionParser:
    """Choice 终端企业行为导出解析器"""
    @classmethod
    def parse(cls, content_bytes: bytes, symbol: str = "", query_start: str = "", query_end: str = "") -> CorporateActionParseResult:
        return WindCorporateActionParser.parse(content_bytes, symbol=symbol, query_start=query_start, query_end=query_end)


class CSVCorporateActionParser:
    """CSV 格式企业行为响应规范解析器 (严格验证关键列头与 Schema)"""
    @classmethod
    def parse(cls, content_bytes: bytes, symbol: str = "", query_start: str = "", query_end: str = "") -> CorporateActionParseResult:
        res = CorporateActionParseResult(parser_name="CSVCorporateActionParser", schema_id="CSV_CORP_ACTION_V1", symbol=symbol, query_start=query_start, query_end=query_end)
        if not content_bytes or len(content_bytes.strip()) == 0:
            res.failed_checks.append("response_empty_or_truncated")
            return res
        try:
            import io
            import pandas as pd
            df = pd.read_csv(io.BytesIO(content_bytes))
            cols_lower = [str(c).lower().strip() for c in df.columns]
            has_sym = any(c in cols_lower for c in ["symbol", "code", "sec_code", "ticker", "wind_code"])
            has_date = any(c in cols_lower for c in ["ex_date", "date", "exdate", "record_date"])
            has_type = any(c in cols_lower for c in ["action_type", "type", "event_type", "cash", "cash_dividend_per_share", "share_ratio"])
            if not (has_sym and has_date and has_type):
                res.failed_checks.append("corporate_action_csv_schema_invalid")
                return res
            res.events = df.to_dict(orient="records") if not df.empty else []
            res.event_count = len(res.events)
            res.parse_success = True
            res.query_success = True
            res.response_status = "SUCCESS"
        except Exception as e:
            res.failed_checks.append(f"corporate_action_csv_schema_invalid_{str(e)}")
        return res


CORPORATE_ACTION_RESPONSE_PARSERS: Dict[str, Any] = {
    "SSE": SSECorporateActionParser,
    "SZSE": SZSECorporateActionParser,
    "WIND": WindCorporateActionParser,
    "CHOICE": ChoiceCorporateActionParser,
    "CSV": CSVCorporateActionParser,
    "CSI": SSECorporateActionParser
}


class CorporateActionResponseParser:
    """对原始数据文件/API响应内容进行独立解析，提取事件数并推导真实 Zero Event 结论 (P0 Derived Proof)"""
    @classmethod
    def parse_response_content(
        cls,
        content_bytes: bytes,
        file_format: str = "json",
        source_id: str = "SSE",
        symbol: str = "",
        query_start: str = "",
        query_end: str = ""
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        s_id = str(source_id).strip().upper()
        if file_format.lower() == "csv" or (content_bytes and not content_bytes.strip().startswith((b"{", b"[")) and b"," in content_bytes):
            res = CSVCorporateActionParser.parse(content_bytes, symbol=symbol, query_start=query_start, query_end=query_end)
        else:
            parser_cls = CORPORATE_ACTION_RESPONSE_PARSERS.get(s_id, SSECorporateActionParser)
            res = parser_cls.parse(content_bytes, symbol=symbol, query_start=query_start, query_end=query_end)
        return res.events, res.failed_checks

    @classmethod
    def parse(
        cls,
        content_bytes: bytes,
        file_format: str = "json",
        source_id: str = "SSE",
        symbol: str = "",
        query_start: str = "",
        query_end: str = ""
    ) -> CorporateActionParseResult:
        s_id = str(source_id).strip().upper()
        if file_format.lower() == "csv" or (content_bytes and not content_bytes.strip().startswith((b"{", b"[")) and b"," in content_bytes):
            return CSVCorporateActionParser.parse(content_bytes, symbol=symbol, query_start=query_start, query_end=query_end)
        parser_cls = CORPORATE_ACTION_RESPONSE_PARSERS.get(s_id, SSECorporateActionParser)
        return parser_cls.parse(content_bytes, symbol=symbol, query_start=query_start, query_end=query_end)


@dataclass
class CorporateActionCoverageEvidence:
    """公司行为覆盖度与零事件审计证据实体 (Fail-Closed Default - P0 Hardened)"""
    symbol: str
    query_start: str
    query_end: str
    source_id: str = "UNKNOWN"
    source_reference: Optional[str] = None
    query_success: bool = False
    raw_result_hash: Optional[str] = None
    raw_result_file: Optional[str] = None
    response_hash: Optional[str] = None
    response_file: Optional[str] = None
    source_metadata_file: Optional[str] = None
    acquisition_receipt_file: Optional[str] = None
    queried_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    empty_result: bool = False
    empty_result_verified: bool = False
    source_class: str = "UNKNOWN"
    evidence_manifest_hash: Optional[str] = None
    production_eligible: bool = True

    def is_valid_zero_event_proof(
        self,
        target_symbol: str,
        backtest_start: str,
        backtest_end: str,
        evidence_dir: Optional[Path] = None
    ) -> Tuple[bool, List[str]]:
        """检验该证据是否足以证明指定标的在指定回测区间内确实不存在任何除权除息事件 (P0 必须完整闭环)"""
        from data.provenance import SourceEvidenceMetadata
        errors = []

        if not self.production_eligible:
            errors.append("evidence_marked_non_production_eligible")

        if self.symbol.strip().upper() != target_symbol.strip().upper():
            errors.append(f"symbol_mismatch_{self.symbol}_vs_{target_symbol}")

        if not self.query_success:
            errors.append("query_not_successful")

        if not self.empty_result:
            errors.append("has_corporate_actions_cannot_claim_zero_event")

        # P0: 强制要求采集回执与元数据文件，绝不可缺省跳过
        if not self.acquisition_receipt_file:
            errors.append("corporate_action_acquisition_receipt_required")

        if not self.source_metadata_file:
            errors.append("corporate_action_source_metadata_required")

        if not self.raw_result_file or not self.raw_result_hash:
            errors.append("missing_raw_result_file_or_hash")

        if not self.response_file or not self.response_hash:
            errors.append("missing_response_file_or_hash")

        if self.source_id not in TRUSTED_SOURCE_REGISTRY:
            errors.append(f"untrusted_source_id_{self.source_id}")

        if self.query_start > backtest_start:
            errors.append(f"query_start_{self.query_start}_after_backtest_start_{backtest_start}")

        if self.query_end < backtest_end:
            errors.append(f"query_end_{self.query_end}_before_backtest_end_{backtest_end}")

        if not evidence_dir:
            errors.append("evidence_dir_missing_required_for_production_verification")
            return False, errors

        # 1. 路径安全约束与物理文件哈希校验
        raw_p = safe_resolve_path(evidence_dir, self.raw_result_file) if self.raw_result_file else None
        if not raw_p or not raw_p.exists():
            errors.append(f"corporate_action_raw_result_file_missing_or_traversal_{self.raw_result_file}")
        else:
            h = hashlib.sha256()
            with open(raw_p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            if h.hexdigest().lower() != str(self.raw_result_hash).lower():
                errors.append(f"corporate_action_raw_file_hash_mismatch_{self.raw_result_hash}_vs_{h.hexdigest()}")

        resp_p = safe_resolve_path(evidence_dir, self.response_file) if self.response_file else None
        if not resp_p or not resp_p.exists():
            errors.append(f"corporate_action_response_file_missing_or_traversal_{self.response_file}")
        else:
            h_resp = hashlib.sha256()
            with open(resp_p, "rb") as f:
                while chunk := f.read(65536):
                    h_resp.update(chunk)
            if h_resp.hexdigest().lower() != str(self.response_hash).lower():
                errors.append(f"corporate_action_response_file_hash_mismatch_{self.response_hash}_vs_{h_resp.hexdigest()}")

            # 2. 从响应真实内容解析推导 Zero Event (拒绝裸布尔自证)
            with open(resp_p, "rb") as f:
                resp_bytes = f.read()
            parse_res = CorporateActionResponseParser.parse(
                resp_bytes,
                source_id=self.source_id,
                symbol=self.symbol,
                query_start=self.query_start,
                query_end=self.query_end
            )
            if not parse_res.parse_success or not parse_res.query_success:
                errors.extend(parse_res.failed_checks)
                self.empty_result_verified = False
            elif parse_res.event_count > 0:
                errors.append(f"response_contains_{parse_res.event_count}_events_cannot_claim_zero_event")
                self.empty_result_verified = False
            else:
                self.empty_result_verified = True

        # 3. 来源元数据 SourceEvidenceMetadata 强核验
        s_meta = None
        if self.source_metadata_file and raw_p and raw_p.exists():
            meta_p = safe_resolve_path(evidence_dir, self.source_metadata_file)
            if not meta_p or not meta_p.exists():
                errors.append(f"corporate_action_source_metadata_missing_{self.source_metadata_file}")
            else:
                s_meta, m_errs = SourceEvidenceMetadata.load_and_verify(meta_p, raw_p)
                if not s_meta:
                    errors.extend(m_errs)
                elif s_meta.source_id.strip().upper() != self.source_id.strip().upper():
                    errors.append(f"source_metadata_id_mismatch_{s_meta.source_id}_vs_{self.source_id}")

        # 4. 可信采集回执 AcquisitionReceipt & Trust Anchor & Query Context 强核验
        if self.acquisition_receipt_file and raw_p and raw_p.exists():
            rec_p = safe_resolve_path(evidence_dir, self.acquisition_receipt_file)
            if not rec_p or not rec_p.exists():
                errors.append(f"corporate_action_receipt_file_missing_{self.acquisition_receipt_file}")
            else:
                receipt, r_errs = AcquisitionReceipt.load_from_file(rec_p)
                if not receipt:
                    errors.extend(r_errs)
                else:
                    r_ok, v_errs = receipt.verify_against_file(raw_p)
                    if not r_ok:
                        errors.extend(v_errs)
                    if s_meta:
                        b_ok, b_errs = receipt.verify_exact_binding(s_meta, raw_p)
                        if not b_ok:
                            errors.extend(b_errs)
                    if not receipt.trust_anchor_verified:
                        errors.append("corporate_action_trust_anchor_unverified")
                    if receipt.source_id.strip().upper() != self.source_id.strip().upper():
                        errors.append(f"receipt_source_id_mismatch_{receipt.source_id}_vs_{self.source_id}")

                    # 5. Query Context 强绑定校验 (P0-1 Mandatory 防重放与篡改)
                    qc_errs = self._validate_signed_query_context(receipt)
                    if qc_errs:
                        errors.extend(qc_errs)

        return len(errors) == 0, errors

    def _validate_signed_query_context(self, receipt: AcquisitionReceipt) -> List[str]:
        """严格校验采集回执中的可信 Query Context (P0-1 Mandatory)"""
        qc_errors = []
        if not receipt.query_context or not isinstance(receipt.query_context, dict):
            qc_errors.append("corporate_action_signed_query_context_required")
            return qc_errors

        qc = receipt.query_context
        # 1. 必填字段存在性校验
        req_fields = ["resource_type", "symbol", "query_start", "query_end"]
        missing = [f for f in req_fields if f not in qc or qc[f] is None or str(qc[f]).strip() == ""]
        fp = qc.get("request_params_sha256") or qc.get("request_fingerprint_sha256")
        if not fp or str(fp).strip() == "":
            missing.append("request_params_sha256")

        if missing:
            qc_errors.append(f"corporate_action_query_context_missing_fields_{missing}")
            return qc_errors

        # 2. resource_type 校验
        if str(qc.get("resource_type", "")).strip().upper() != "CORPORATE_ACTION":
            qc_errors.append(f"corporate_action_query_context_invalid_resource_type_{qc.get('resource_type')}")

        # 3. 标的精确匹配校验
        qc_sym = str(qc.get("symbol", "")).strip().upper()
        if qc_sym != self.symbol.strip().upper():
            qc_errors.append(f"corporate_action_query_context_symbol_mismatch_{qc_sym}_vs_{self.symbol}")

        # 4. 日期合法性与范围校验
        try:
            qc_start = pd.to_datetime(qc.get("query_start")).strftime("%Y-%m-%d")
            qc_end = pd.to_datetime(qc.get("query_end")).strftime("%Y-%m-%d")
            if qc_start > qc_end:
                qc_errors.append(f"corporate_action_query_context_invalid_date_range_{qc_start}_gt_{qc_end}")
            else:
                if qc_start > self.query_start:
                    qc_errors.append(f"corporate_action_query_context_start_mismatch_{qc_start}_after_{self.query_start}")
                if qc_end < self.query_end:
                    qc_errors.append(f"corporate_action_query_context_end_mismatch_{qc_end}_before_{self.query_end}")
        except Exception as e:
            qc_errors.append(f"corporate_action_query_context_unparseable_dates_{str(e)}")

        # 5. 请求指纹 SHA256 格式校验
        if fp and not HEX_64_PATTERN.match(str(fp)):
            qc_errors.append(f"corporate_action_invalid_request_params_hash_{fp}")

        return qc_errors

    def verify_dataset_evidence(
        self,
        target_symbol: str,
        backtest_start: str,
        backtest_end: str,
        evidence_dir: Optional[Path] = None
    ) -> Tuple[bool, List[str]]:
        """非空公司行为事件数据集物理证据与来源完整认证核验 (P0)"""
        from data.provenance import SourceEvidenceMetadata
        errors = []
        if not self.production_eligible:
            errors.append("evidence_marked_non_production_eligible")

        if self.symbol.strip().upper() != target_symbol.strip().upper():
            errors.append(f"symbol_mismatch_{self.symbol}_vs_{target_symbol}")

        if not self.query_success:
            errors.append("query_not_successful")

        if not self.acquisition_receipt_file:
            errors.append("corporate_action_acquisition_receipt_required")

        if not self.source_metadata_file:
            errors.append("corporate_action_source_metadata_required")

        if not self.raw_result_file or not self.raw_result_hash:
            errors.append("missing_raw_result_file_or_hash")

        if not self.response_file or not self.response_hash:
            errors.append("missing_response_file_or_hash")

        if self.source_id not in TRUSTED_SOURCE_REGISTRY:
            errors.append(f"untrusted_source_id_{self.source_id}")

        if self.query_start > backtest_start:
            errors.append(f"query_start_{self.query_start}_after_backtest_start_{backtest_start}")

        if self.query_end < backtest_end:
            errors.append(f"query_end_{self.query_end}_before_backtest_end_{backtest_end}")

        if not evidence_dir:
            errors.append("evidence_dir_missing_required_for_production_verification")
            return False, errors

        raw_p = safe_resolve_path(evidence_dir, self.raw_result_file) if self.raw_result_file else None
        if not raw_p or not raw_p.exists():
            errors.append(f"corporate_action_raw_result_file_missing_or_traversal_{self.raw_result_file}")
        else:
            h = hashlib.sha256()
            with open(raw_p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            if h.hexdigest().lower() != str(self.raw_result_hash).lower():
                errors.append(f"corporate_action_raw_file_hash_mismatch_{self.raw_result_hash}_vs_{h.hexdigest()}")

        resp_p = safe_resolve_path(evidence_dir, self.response_file) if self.response_file else None
        if not resp_p or not resp_p.exists():
            errors.append(f"corporate_action_response_file_missing_or_traversal_{self.response_file}")
        else:
            h_resp = hashlib.sha256()
            with open(resp_p, "rb") as f:
                while chunk := f.read(65536):
                    h_resp.update(chunk)
            if h_resp.hexdigest().lower() != str(self.response_hash).lower():
                errors.append(f"corporate_action_response_file_hash_mismatch_{self.response_hash}_vs_{h_resp.hexdigest()}")

        # 来源元数据与采集回执
        s_meta = None
        if self.source_metadata_file and raw_p and raw_p.exists():
            meta_p = safe_resolve_path(evidence_dir, self.source_metadata_file)
            if not meta_p or not meta_p.exists():
                errors.append(f"corporate_action_source_metadata_missing_{self.source_metadata_file}")
            else:
                s_meta, m_errs = SourceEvidenceMetadata.load_and_verify(meta_p, raw_p)
                if not s_meta:
                    errors.extend(m_errs)
                elif s_meta.source_id.strip().upper() != self.source_id.strip().upper():
                    errors.append(f"source_metadata_id_mismatch_{s_meta.source_id}_vs_{self.source_id}")

        if self.acquisition_receipt_file and raw_p and raw_p.exists():
            rec_p = safe_resolve_path(evidence_dir, self.acquisition_receipt_file)
            if not rec_p or not rec_p.exists():
                errors.append(f"corporate_action_receipt_file_missing_{self.acquisition_receipt_file}")
            else:
                receipt, r_errs = AcquisitionReceipt.load_from_file(rec_p)
                if not receipt:
                    errors.extend(r_errs)
                else:
                    r_ok, v_errs = receipt.verify_against_file(raw_p)
                    if not r_ok:
                        errors.extend(v_errs)
                    if s_meta:
                        b_ok, b_errs = receipt.verify_exact_binding(s_meta, raw_p)
                        if not b_ok:
                            errors.extend(b_errs)
                    if not receipt.trust_anchor_verified:
                        errors.append("corporate_action_trust_anchor_unverified")
                    if receipt.source_id.strip().upper() != self.source_id.strip().upper():
                        errors.append(f"receipt_source_id_mismatch_{receipt.source_id}_vs_{self.source_id}")

                    # Query Context 强绑定校验 (P0-1 Mandatory)
                    qc_errs = self._validate_signed_query_context(receipt)
                    if qc_errs:
                        errors.extend(qc_errs)

        return len(errors) == 0, errors
