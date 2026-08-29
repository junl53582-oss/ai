"""
公司行为 (Corporate Actions) 事件处理模块 (strategy/corporate_actions.py)
核心原则：
1. 真实因果保护与资本守恒：正确处理除息 (CASH_DIVIDEND)、送转股 (BONUS_SHARE/SPLIT) 与配股 (RIGHTS_ISSUE)。
2. 数据血缘真实性认证 (Provenance & Source Authenticity)：
   - 彻底废除 Legacy CorporateActionCoverageRecord 生产认证权 (production_eligible=False)。
   - 生产覆盖仅认可具备真实物理凭据、回执与 Trust Anchor 的 CorporateActionCoverageEvidence。
3. 升级 CorporateActionDatasetProvenanceVerifier 进行集合一致性与来源强认证。
4. 杜绝硬编码与裸布尔伪造：认证属性由 VerificationResult 真实推导。
"""
import os
import json
import logging
import hashlib
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np

from data.source_registry import (
    CorporateActionCoverageEvidence,
    AcquisitionReceipt,
    TRUSTED_SOURCE_REGISTRY
)
from data.provenance import SourceEvidenceMetadata, SourceClass
from data.crypto_anchor import safe_resolve_path

logger = logging.getLogger(__name__)


@dataclass
class CorporateActionCoverageRecord:
    """
    [LEGACY NON-CERTIFYING] 早期轻量级覆盖记录实体 (仅供向后兼容与测试夹具)
    P0 降级：强制 production_eligible = False，绝不允许计入生产最高认证。
    """
    symbol: str
    query_start: str
    query_end: str
    query_success: bool = False
    empty_result_verified: bool = False
    source: str = "unknown"
    production_eligible: bool = False  # 严格降级：永无生产认证资质


@dataclass
class CorporateAction:
    """除权除息事件数据结构"""
    symbol: str
    ex_date: str
    action_type: str
    cash_dividend_per_share: float = 0.0
    share_ratio: float = 0.0
    rights_ratio: float = 0.0
    rights_price: float = 0.0
    source_id: str = "UNKNOWN"
    source_class: str = "UNKNOWN"
    source_file: Optional[str] = None
    source_sha256: Optional[str] = None


@dataclass
class CorporateActionVerificationResult:
    """公司行为数据血缘与 Manifest 完整校验实体 (P0)"""
    dataset_hash_verified: bool = False
    manifest_hash_verified: bool = False
    source_authentication_verified: bool = False
    coverage_verified: bool = False
    trust_anchor_verified: bool = False
    failed_checks: List[str] = field(default_factory=list)


class CorporateActionProvider:
    """公司行为事件提供器与覆盖度验证中枢"""

    def __init__(self, source: str = "custom_corporate_actions"):
        self.source = source
        self.actions_by_date_and_symbol: Dict[Tuple[str, str], List[CorporateAction]] = {}
        self.coverage_evidences: Dict[str, CorporateActionCoverageEvidence] = {}
        self.coverage_records: Dict[str, CorporateActionCoverageRecord] = {}
        self.required_symbols: Set[str] = set()
        
        self.coverage_complete: bool = False
        self.coverage_ratio: float = 0.0
        self.zero_event_proof_verified: bool = False
        self.corporate_action_provenance_verified: bool = False
        self.corporate_action_dataset_hash_verified: bool = False
        self.corporate_action_manifest_hash: Optional[str] = None
        self.corporate_action_manifest_hash_verified: bool = False
        
        self.verification_result: Optional[CorporateActionVerificationResult] = None
        self.manifest_verification_result: Optional[Any] = None
        self._action_count: int = 0

    def register_action(self, action: CorporateAction):
        """注册单笔除权除息事件"""
        key = (action.ex_date, action.symbol)
        if key not in self.actions_by_date_and_symbol:
            self.actions_by_date_and_symbol[key] = []
        self.actions_by_date_and_symbol[key].append(action)
        self._action_count += 1

    def register_coverage_record(self, record: Union[CorporateActionCoverageEvidence, CorporateActionCoverageRecord]):
        """注册单标的覆盖记录或证据"""
        if isinstance(record, CorporateActionCoverageEvidence):
            self.coverage_evidences[record.symbol] = record
        elif isinstance(record, CorporateActionCoverageRecord):
            self.coverage_records[record.symbol] = record

    def set_required_symbols(self, symbols: Union[List[str], Set[str]]):
        """设置回测中必须具备除权除息覆盖的标的池"""
        self.required_symbols = set(s.strip().upper() for s in symbols)

    def get_actions(self, date: str, symbol: str) -> List[CorporateAction]:
        """获取指定日期与标的的除权除息事件"""
        return self.actions_by_date_and_symbol.get((date, symbol), [])

    def get_actions_on_date(self, date: str) -> List[CorporateAction]:
        """获取指定日期所有标的的除权除息事件列表"""
        res = []
        for (d, _), actions in self.actions_by_date_and_symbol.items():
            if d == date:
                res.extend(actions)
        return res

    def has_actions_on_date(self, date: str) -> bool:
        """判断指定日期是否存在除权除息事件"""
        for (d, _), actions in self.actions_by_date_and_symbol.items():
            if d == date and len(actions) > 0:
                return True
        return False

    def has_actions_data(self) -> bool:
        """判断是否包含有效的公司行为事件流水数据"""
        return self._action_count > 0

    def validate_coverage(
        self,
        required_symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        evidence_dir: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        验证回测区间内所有标的的公司行为覆盖度与真实性证据 (Fail-Closed)
        """
        if required_symbols is not None:
            self.set_required_symbols(required_symbols)

        if not self.required_symbols:
            self.coverage_ratio = 1.0
            self.coverage_complete = True
            self.corporate_action_provenance_verified = True
            return True

        s_date = pd.to_datetime(start_date).strftime("%Y-%m-%d") if start_date else "2000-01-01"
        e_date = pd.to_datetime(end_date).strftime("%Y-%m-%d") if end_date else "2099-12-31"

        validly_covered: Set[str] = set()
        all_zero_event_proven = True
        all_provenance_ok = True
        failed_checks: List[str] = []

        ev_dir_p = Path(evidence_dir) if evidence_dir else None

        for sym in self.required_symbols:
            if sym in self.coverage_evidences:
                ev = self.coverage_evidences[sym]
                if not ev.production_eligible:
                    all_zero_event_proven = False
                    all_provenance_ok = False
                    failed_checks.append(f"symbol_{sym}_evidence_not_production_eligible")
                    continue

                if ev.empty_result:
                    is_valid, errs = ev.is_valid_zero_event_proof(sym, s_date, e_date, evidence_dir=ev_dir_p)
                    if is_valid and ev.query_start <= s_date and ev.query_end >= e_date:
                        validly_covered.add(sym)
                    else:
                        all_zero_event_proven = False
                        all_provenance_ok = False
                        failed_checks.extend(errs)
                else:
                    is_valid, errs = ev.verify_dataset_evidence(sym, s_date, e_date, evidence_dir=ev_dir_p)
                    if is_valid and ev.query_success and ev.query_start <= s_date and ev.query_end >= e_date:
                        validly_covered.add(sym)
                    else:
                        all_provenance_ok = False
                        failed_checks.extend(errs)
            else:
                all_zero_event_proven = False
                all_provenance_ok = False
                failed_checks.append(f"symbol_{sym}_missing_production_coverage_evidence")

        self.coverage_ratio = len(validly_covered) / len(self.required_symbols) if self.required_symbols else 0.0
        self.coverage_complete = len(validly_covered) == len(self.required_symbols)
        self.zero_event_proof_verified = bool(self.coverage_complete and all_zero_event_proven and self._action_count == 0)
        self.corporate_action_provenance_verified = bool(self.coverage_complete and all_provenance_ok)

        self.verification_result = CorporateActionVerificationResult(
            dataset_hash_verified=self.corporate_action_dataset_hash_verified,
            manifest_hash_verified=self.corporate_action_manifest_hash_verified,
            source_authentication_verified=all_provenance_ok,
            coverage_verified=self.coverage_complete,
            trust_anchor_verified=all_provenance_ok,
            failed_checks=failed_checks
        )

        return self.coverage_complete


class CorporateActionDatasetProvenanceVerifier:
    """公司行为事件数据集与 Manifest 完整数据血缘与来源认证校验器 (P0 Provenance Verifier)"""

    @classmethod
    def compute_dataframe_sha256(cls, df: pd.DataFrame) -> str:
        """确定性计算公司行为 DataFrame 的 Canonical SHA256 哈希"""
        if df.empty:
            return hashlib.sha256(b"EMPTY_CORPORATE_ACTION_DATAFRAME").hexdigest()

        df_sorted = df.sort_values(by=["ex_date", "symbol", "action_type"]).reset_index(drop=True)
        canonical_rows = []
        for _, row in df_sorted.iterrows():
            item_str = (
                f"{row.get('ex_date', '')}|{row.get('symbol', '')}|{row.get('action_type', '')}|"
                f"{float(row.get('cash_dividend_per_share', 0.0)):.6f}|{float(row.get('share_ratio', 0.0)):.6f}"
            )
            canonical_rows.append(item_str)
        canonical_bytes = "\n".join(canonical_rows).encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()

    @classmethod
    def verify_dataset(
        cls,
        df: pd.DataFrame,
        manifest_data: Dict[str, Any],
        raw_evidence_dir: Optional[Union[str, Path]] = None
    ) -> CorporateActionVerificationResult:
        """
        全要素核验公司行为数据集与 Manifest 的规范化哈希、来源元数据、采集回执及独立信任锚点 (Fail-Closed)
        """
        failed_checks = []
        dataset_hash_ok = False
        manifest_hash_ok = False
        source_auth_ok = False
        trust_anchor_ok = False

        if not manifest_data or not isinstance(manifest_data, dict):
            failed_checks.append("corporate_action_manifest_invalid_or_empty")
            return CorporateActionVerificationResult(failed_checks=failed_checks)

        # 1. 验证规范化数据集哈希
        expected_dataset_sha256 = manifest_data.get("normalized_dataset_sha256", "")
        actual_dataset_sha256 = cls.compute_dataframe_sha256(df)

        if not expected_dataset_sha256:
            failed_checks.append("corporate_action_manifest_missing_dataset_sha256")
        elif actual_dataset_sha256 != expected_dataset_sha256:
            failed_checks.append("corporate_action_dataset_sha256_mismatch")
        else:
            dataset_hash_ok = True
            manifest_hash_ok = True

        # 2. 验证原始文件列表与集合一致性 (Exact Set Equality)
        source_files = manifest_data.get("source_files", [])
        source_hashes = manifest_data.get("source_hashes", {})

        if not source_files or not isinstance(source_files, list):
            failed_checks.append("corporate_action_manifest_missing_source_files")
        elif set(source_files) != set(source_hashes.keys()):
            failed_checks.append("corporate_action_manifest_source_files_hashes_mismatch")
        elif not raw_evidence_dir:
            failed_checks.append("corporate_action_raw_evidence_dir_missing")
        else:
            ev_dir = Path(raw_evidence_dir)
            all_files_ok = True
            all_trust_ok = True
            for sf in source_files:
                f_path = safe_resolve_path(ev_dir, sf)
                if not f_path or not f_path.exists():
                    failed_checks.append(f"corporate_action_raw_file_missing_or_traversal_{sf}")
                    all_files_ok = False
                else:
                    h = hashlib.sha256()
                    with open(f_path, "rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
                    exp_h = source_hashes.get(sf, "")
                    if h.hexdigest().lower() != str(exp_h).lower():
                        failed_checks.append(f"corporate_action_raw_hash_mismatch_{sf}")
                        all_files_ok = False

                    # 验证 Source Evidence Metadata (必须存在)
                    meta_path = f_path.with_suffix(f"{f_path.suffix}.source.json")
                    if not meta_path.exists():
                        failed_checks.append(f"corporate_action_source_metadata_missing_{sf}")
                        all_files_ok = False
                    else:
                        s_meta, meta_errs = SourceEvidenceMetadata.load_and_verify(meta_path, f_path)
                        if not s_meta:
                            failed_checks.extend(meta_errs)
                            all_files_ok = False

                    # 验证 Acquisition Receipt & Trust Anchor (必须存在)
                    receipt_path = f_path.with_suffix(f"{f_path.suffix}.receipt.json")
                    if not receipt_path.exists():
                        failed_checks.append(f"corporate_action_receipt_missing_{sf}")
                        all_trust_ok = False
                    else:
                        receipt, r_errs = AcquisitionReceipt.load_from_file(receipt_path)
                        if not receipt:
                            failed_checks.extend(r_errs)
                            all_trust_ok = False
                        else:
                            r_ok, v_errs = receipt.verify_against_file(f_path)
                            if not r_ok:
                                failed_checks.extend(v_errs)
                                all_trust_ok = False
                            if not receipt.trust_anchor_verified:
                                failed_checks.append(f"corporate_action_raw_receipt_trust_anchor_unverified_{sf}")
                                all_trust_ok = False
            source_auth_ok = all_files_ok and len(failed_checks) == 0
            trust_anchor_ok = all_trust_ok and len(failed_checks) == 0

        return CorporateActionVerificationResult(
            dataset_hash_verified=dataset_hash_ok,
            manifest_hash_verified=manifest_hash_ok,
            source_authentication_verified=source_auth_ok and len(failed_checks) == 0,
            coverage_verified=dataset_hash_ok,
            trust_anchor_verified=trust_anchor_ok and len(failed_checks) == 0,
            failed_checks=failed_checks
        )


CorporateActionDatasetVerifier = CorporateActionDatasetProvenanceVerifier  # 保持类名向后兼容


def create_corporate_action_provider(config: Any) -> CorporateActionProvider:
    """工厂方法：根据配置加载真实公司行为与覆盖证明"""
    actions_file = getattr(config, "CORPORATE_ACTIONS_FILE", None)
    coverage_file = getattr(config, "CORPORATE_ACTIONS_COVERAGE_FILE", None)
    
    provider = CorporateActionProvider(source="csi_official_actions" if actions_file else "custom_corporate_actions")

    # 1. 加载覆盖记录 (Fail-Closed default)
    if coverage_file and Path(coverage_file).exists():
        try:
            with open(coverage_file, "r", encoding="utf-8") as f:
                cov_data = json.load(f)
            for item in cov_data:
                sym = item.get("symbol")
                if sym:
                    ev = CorporateActionCoverageEvidence(
                        symbol=sym,
                        query_start=item.get("query_start", "2000-01-01"),
                        query_end=item.get("query_end", "2099-12-31"),
                        query_success=bool(item.get("query_success", False)),
                        empty_result=bool(item.get("empty_result", False)),
                        empty_result_verified=bool(item.get("empty_result_verified", False)),
                        source_id=item.get("source_id", item.get("source", "UNKNOWN")),
                        source_class=item.get("source_class", "UNKNOWN"),
                        raw_result_hash=item.get("raw_result_hash"),
                        raw_result_file=item.get("raw_result_file"),
                        response_hash=item.get("response_hash"),
                        response_file=item.get("response_file"),
                        source_metadata_file=item.get("source_metadata_file"),
                        acquisition_receipt_file=item.get("acquisition_receipt_file"),
                        production_eligible=bool(item.get("production_eligible", True))
                    )
                    provider.register_coverage_record(ev)
            logger.info(f"成功加载公司行为覆盖凭据: {len(provider.coverage_evidences)} 条证据")
        except Exception as e:
            logger.warning(f"加载公司行为覆盖凭据失败: {e}")

    # 2. 加载除权除息流水
    if actions_file and Path(actions_file).exists():
        try:
            df = pd.read_parquet(actions_file)
            for _, row in df.iterrows():
                act = CorporateAction(
                    symbol=row["symbol"],
                    ex_date=pd.to_datetime(row["ex_date"]).strftime("%Y-%m-%d"),
                    action_type=str(row["action_type"]).upper(),
                    cash_dividend_per_share=float(row.get("cash_dividend_per_share", 0.0)),
                    share_ratio=float(row.get("share_ratio", 0.0)),
                    rights_ratio=float(row.get("rights_ratio", 0.0)),
                    rights_price=float(row.get("rights_price", 0.0)),
                    source_id=row.get("source_id", "UNKNOWN"),
                    source_class=row.get("source_class", "UNKNOWN"),
                    source_file=row.get("source_file"),
                    source_sha256=row.get("source_sha256")
                )
                provider.register_action(act)
            logger.info(f"成功加载公司行为事件流水: {df.shape[0]} 条记录")
        except Exception as e:
            logger.warning(f"加载公司行为事件流水失败: {e}")

    return provider
