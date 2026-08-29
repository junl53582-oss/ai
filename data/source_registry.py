"""
受信任生产来源注册中心与采集凭证规范 (data/source_registry.py)
实现严格的 Trusted Source Registry 与 Acquisition Receipt 鉴证体系：
- 严格基于注册表 (Registry) 鉴权来源合法性，禁止仅凭字符串 (如 "CSI Official") 或任意伪造 URL 冒充官方
- 生产级 OFFICIAL_PRIMARY 认证必须绑定真实的采集回执 (Acquisition Receipt)
- 公司行为零事件证明必须基于严密的 CorporateActionCoverageEvidence，严禁以纯布尔值声称证明
- 严格遵循 Fail-Closed: 缺少凭证或数据时一律为 False / UNKNOWN / HIGH_RISK
"""
import re
import json
import hashlib
from urllib.parse import urlparse
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from pathlib import Path


TRUSTED_SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "CSI": {
        "source_class": "OFFICIAL_PRIMARY",
        "source_name": "China Securities Index Co., Ltd.",
        "allowed_domains": ["csindex.com.cn", "www.csindex.com.cn"],
        "allowed_evidence_types": ["INDEX_CONSTITUENT_ADJUSTMENT", "BASELINE_SNAPSHOT"]
    },
    "SSE": {
        "source_class": "OFFICIAL_PRIMARY",
        "source_name": "Shanghai Stock Exchange",
        "allowed_domains": ["sse.com.cn", "www.sse.com.cn"],
        "allowed_evidence_types": ["SECURITY_MASTER", "CALENDAR", "CORPORATE_ACTION"]
    },
    "SZSE": {
        "source_class": "OFFICIAL_PRIMARY",
        "source_name": "Shenzhen Stock Exchange",
        "allowed_domains": ["szse.cn", "www.szse.cn"],
        "allowed_evidence_types": ["SECURITY_MASTER", "CALENDAR", "CORPORATE_ACTION"]
    },
    "WIND": {
        "source_class": "LICENSED_VENDOR",
        "source_name": "Wind Information Co., Ltd.",
        "allowed_domains": ["wind.com.cn", "www.wind.com.cn"],
        "allowed_evidence_types": ["INDEX_CONSTITUENT_ADJUSTMENT", "BASELINE_SNAPSHOT", "SECURITY_MASTER", "CALENDAR", "CORPORATE_ACTION"]
    },
    "CHOICE": {
        "source_class": "LICENSED_VENDOR",
        "source_name": "EastMoney Choice Terminal",
        "allowed_domains": ["eastmoney.com", "choice.eastmoney.com"],
        "allowed_evidence_types": ["INDEX_CONSTITUENT_ADJUSTMENT", "BASELINE_SNAPSHOT", "SECURITY_MASTER", "CALENDAR", "CORPORATE_ACTION"]
    }
}


def extract_domain(url: Optional[str]) -> str:
    """提取 URL 中的域名小写主机名"""
    if not url:
        return ""
    try:
        parsed = urlparse(str(url).strip())
        netloc = parsed.netloc or parsed.path.split('/')[0]
        return netloc.split(':')[0].lower()
    except Exception:
        return ""


@dataclass
class AcquisitionReceipt:
    """真实数据采集回执实体 (Acquisition Receipt)"""
    receipt_id: str
    source_id: str
    source_url: str
    requested_at: str
    downloaded_at: str
    http_status: int = 200
    content_type: str = "text/csv"
    response_etag: Optional[str] = None
    response_last_modified: Optional[str] = None
    raw_sha256: str = ""
    original_filename: str = ""
    downloader_version: str = "1.0"
    acquisition_method: str = "OFFICIAL_HTTP_DOWNLOAD"
    receipt_signature_hash: Optional[str] = None

    def compute_signature(self) -> str:
        """根据采集回执核心要素计算防伪签名摘要"""
        raw_str = (
            f"{self.receipt_id}|{self.source_id}|{self.source_url}|"
            f"{self.requested_at}|{self.downloaded_at}|{self.http_status}|"
            f"{self.raw_sha256}|{self.original_filename}|{self.acquisition_method}"
        )
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def verify_against_file(self, raw_file_path: Path) -> Tuple[bool, List[str]]:
        """验证采集回执与本地 Raw 实体文件的一致性"""
        errors = []
        if not raw_file_path.exists():
            return False, [f"raw_file_not_found_{raw_file_path.name}"]

        # 1. 验证 source_id 是否在信任注册表中
        if self.source_id not in TRUSTED_SOURCE_REGISTRY:
            errors.append(f"source_id_{self.source_id}_not_in_trusted_registry")
            return False, errors

        reg_info = TRUSTED_SOURCE_REGISTRY[self.source_id]

        # 2. 验证域名是否在允许的官方白名单内
        domain = extract_domain(self.source_url)
        if not domain or domain not in reg_info.get("allowed_domains", []):
            errors.append(f"source_domain_{domain}_not_allowed_for_source_{self.source_id}")

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

        # 6. 验证签名摘要
        if self.receipt_signature_hash:
            computed_sig = self.compute_signature()
            if self.receipt_signature_hash.lower() != computed_sig.lower():
                errors.append("receipt_signature_tampered")

        return len(errors) == 0, errors

    @classmethod
    def load_from_file(cls, receipt_path: Path) -> Tuple[Optional["AcquisitionReceipt"], List[str]]:
        """从 JSON 文件加载采集回执"""
        if not receipt_path.exists():
            return None, [f"missing_acquisition_receipt_{receipt_path.name}"]
        try:
            with open(receipt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            receipt = cls(**data)
            return receipt, []
        except Exception as e:
            return None, [f"corrupted_acquisition_receipt_{receipt_path.name}_{str(e)}"]


@dataclass
class CorporateActionCoverageEvidence:
    """公司行为覆盖度与零事件审计证据实体 (Fail-Closed Default)"""
    symbol: str
    query_start: str
    query_end: str
    source_id: str = "UNKNOWN"
    source_reference: Optional[str] = None
    query_success: bool = False
    raw_result_hash: Optional[str] = None
    response_hash: Optional[str] = None
    queried_at: str = ""
    empty_result: bool = False
    empty_result_verified: bool = False  # 必须严格默认为 False (Fail-Closed)
    source_class: str = "UNKNOWN"
    evidence_manifest_hash: Optional[str] = None

    def is_valid_zero_event_proof(self, target_symbol: str, backtest_start: str, backtest_end: str) -> Tuple[bool, List[str]]:
        """检验该证据是否足以证明指定标的在指定回测区间内确实不存在任何除权除息事件"""
        errors = []
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

        return len(errors) == 0, errors
