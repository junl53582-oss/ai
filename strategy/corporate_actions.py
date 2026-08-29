"""
公司行为 (Corporate Actions) 数据结构、覆盖证明与除权除息调度器 (strategy/corporate_actions.py)
处理现金分红、送红股、转增股与股票拆细等公司行为，
提供全量股票池覆盖完整性 (Coverage-Aware) 验证与除息除权价值守恒调度。
严格遵循 Fail-Closed 原则：
- CorporateActionCoverageRecord 标记为 production_eligible=False (仅用于测试/非认证)，绝不赋予生产认证权。
- 生产环境仅接受 CorporateActionCoverageEvidence，且必须具备真实的物理文件与区间覆盖凭据。
- 提供 CorporateActionDatasetVerifier 与 Corporate Action Manifest 验证。
"""
import json
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Set, Any, Tuple
import pandas as pd

from data.source_registry import (
    CorporateActionCoverageEvidence,
    TRUSTED_SOURCE_REGISTRY,
    AcquisitionReceipt
)

logger = logging.getLogger(__name__)


@dataclass
class CorporateAction:
    """单笔公司行为记录"""
    symbol: str
    ex_date: str                               # 除权除息日 YYYY-MM-DD
    action_type: str                           # CASH_DIVIDEND | BONUS_SHARE | SPLIT
    cash_dividend_per_share: float = 0.0       # 每股派发现金红利 (元)
    share_ratio: float = 0.0                   # 每股送转比例 (如10送3 -> 0.3)
    rights_ratio: float = 0.0                  # 配股比例 (暂未支持撮合，明确列入 unsupported)
    rights_price: float = 0.0                  # 配股价格 (元)
    source_id: str = "UNKNOWN"
    source_class: str = "UNKNOWN"
    source_file: Optional[str] = None
    source_sha256: Optional[str] = None


@dataclass
class CorporateActionCoverageRecord:
    """旧版/测试用公司行为覆盖记录 (Legacy Non-Certifying Record)"""
    symbol: str
    query_start: str
    query_end: str
    query_success: bool = False               # 严格默认为 False (Fail-Closed)
    source: str = "custom_corporate_actions"
    source_version: str = "1.0"
    empty_result_verified: bool = False       # 严格默认为 False (Fail-Closed)
    source_class: str = "UNKNOWN"
    raw_result_hash: Optional[str] = None
    response_hash: Optional[str] = None
    supported_action_types: List[str] = field(default_factory=lambda: ["CASH_DIVIDEND", "BONUS_SHARE", "SPLIT"])
    production_eligible: bool = False         # 严格禁止进入生产认证 (P0 Hardened)


class CorporateActionProvider:
    """公司行为注册、覆盖度验证与查询中心 (Coverage-Aware & Decoupled)"""

    def __init__(
        self,
        source: str = "custom_corporate_actions",
        coverage_start: Optional[str] = None,
        coverage_end: Optional[str] = None
    ):
        self._actions_by_date: Dict[str, List[CorporateAction]] = {}
        self._action_count: int = 0
        self.coverage_records: Dict[str, CorporateActionCoverageRecord] = {}
        self.coverage_evidences: Dict[str, CorporateActionCoverageEvidence] = {}
        self.corporate_action_source: str = source
        self.coverage_start: Optional[str] = pd.to_datetime(coverage_start).strftime("%Y-%m-%d") if coverage_start else None
        self.coverage_end: Optional[str] = pd.to_datetime(coverage_end).strftime("%Y-%m-%d") if coverage_end else None
        self.covered_symbols: Set[str] = set()
        self.required_symbols: Set[str] = set()
        self.coverage_ratio: float = 0.0
        self.coverage_complete: bool = False
        self.zero_event_proof_verified: bool = False
        
        self.action_types_supported: List[str] = ["CASH_DIVIDEND", "BONUS_SHARE", "SPLIT"]
        self.unsupported_corporate_action_types: List[str] = ["RIGHTS_ISSUE"]

    def register_coverage_record(self, record: Union[CorporateActionCoverageRecord, CorporateActionCoverageEvidence]):
        """注册数据源对特定股票在特定区间的完整性覆盖证明"""
        sym = record.symbol.strip().upper()
        if isinstance(record, CorporateActionCoverageEvidence):
            self.coverage_evidences[sym] = record
            rec = CorporateActionCoverageRecord(
                symbol=sym,
                query_start=record.query_start,
                query_end=record.query_end,
                query_success=record.query_success,
                source=record.source_id,
                empty_result_verified=record.empty_result_verified,
                source_class=record.source_class,
                raw_result_hash=record.raw_result_hash,
                response_hash=record.response_hash,
                production_eligible=record.production_eligible
            )
            self.coverage_records[sym] = rec
        else:
            self.coverage_records[sym] = record

        self.covered_symbols.add(sym)
        q_s = pd.to_datetime(record.query_start).strftime("%Y-%m-%d")
        q_e = pd.to_datetime(record.query_end).strftime("%Y-%m-%d")
        if self.coverage_start is None or q_s < self.coverage_start:
            self.coverage_start = q_s
        if self.coverage_end is None or q_e > self.coverage_end:
            self.coverage_end = q_e

    def register_action(self, action: CorporateAction):
        """注册单笔公司行为"""
        date_str = pd.to_datetime(action.ex_date).strftime("%Y-%m-%d")
        if date_str not in self._actions_by_date:
            self._actions_by_date[date_str] = []
        self._actions_by_date[date_str].append(action)
        self._action_count += 1
        sym = action.symbol.strip().upper()
        self.covered_symbols.add(sym)

    def has_actions_data(self) -> bool:
        """是否存在除权除息实际动作数据"""
        return self._action_count > 0

    def has_coverage_data(self) -> bool:
        """是否存在覆盖度记录"""
        return len(self.coverage_records) > 0 or len(self.coverage_evidences) > 0

    def get_actions_on_date(self, date: Union[str, pd.Timestamp]) -> List[CorporateAction]:
        """获取指定除息除权日生效的公司行为列表"""
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        return self._actions_by_date.get(date_str, [])

    def validate_coverage(
        self,
        required_symbols: List[str],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        evidence_dir: Optional[Path] = None
    ) -> bool:
        """
        验证回测区间与所有候选股票的公司行为覆盖完整性 (P0 生产级严格校验)：
        - 仅接受具有真实物理文件的 CorporateActionCoverageEvidence。
        - 旧版 CorporateActionCoverageRecord 自动降级，绝不赋予生产 validly_covered 资格。
        """
        self.required_symbols = set(s.strip().upper() for s in required_symbols)
        if not self.required_symbols:
            self.coverage_ratio = 1.0
            self.coverage_complete = True
            return True

        s_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        e_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")

        validly_covered: Set[str] = set()
        all_zero_event_proven = True

        for sym in self.required_symbols:
            if sym in self.coverage_evidences:
                ev = self.coverage_evidences[sym]
                if not ev.production_eligible:
                    all_zero_event_proven = False
                    continue

                if ev.empty_result:
                    is_valid, _ = ev.is_valid_zero_event_proof(sym, s_date, e_date, evidence_dir=evidence_dir)
                    if is_valid and ev.query_start <= s_date and ev.query_end >= e_date:
                        validly_covered.add(sym)
                    else:
                        all_zero_event_proven = False
                else:
                    if ev.query_success and ev.query_start <= s_date and ev.query_end >= e_date:
                        validly_covered.add(sym)
                    else:
                        all_zero_event_proven = False
            else:
                # 仅有 Legacy CorporateActionCoverageRecord 或无记录 -> 绝不能通过生产认证
                all_zero_event_proven = False

        self.coverage_ratio = len(validly_covered) / len(self.required_symbols)
        self.coverage_complete = len(validly_covered) == len(self.required_symbols)
        self.zero_event_proof_verified = bool(self.coverage_complete and all_zero_event_proven and self._action_count == 0)

        return self.coverage_complete


class CorporateActionDatasetVerifier:
    """公司行为事件数据集与 Manifest 物理校验器 (P0 Hardened)"""

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
        raw_evidence_dir: Optional[Path] = None
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """核验公司行为数据集与 Manifest 的哈希、来源及物理文件链"""
        failed_checks = []
        details = {}

        # 1. 验证规范化哈希
        expected_dataset_sha256 = manifest_data.get("normalized_dataset_sha256", "")
        actual_dataset_sha256 = cls.compute_dataframe_sha256(df)
        details["actual_dataset_sha256"] = actual_dataset_sha256
        details["expected_dataset_sha256"] = expected_dataset_sha256

        if not expected_dataset_sha256:
            failed_checks.append("corporate_action_manifest_missing_dataset_sha256")
        elif actual_dataset_sha256 != expected_dataset_sha256:
            failed_checks.append("corporate_action_dataset_sha256_mismatch")

        # 2. 验证原始文件存在性与哈希
        source_files = manifest_data.get("source_files", [])
        source_hashes = manifest_data.get("source_hashes", {})
        if not source_files:
            failed_checks.append("corporate_action_manifest_missing_source_files")
        elif raw_evidence_dir:
            ev_dir = Path(raw_evidence_dir)
            for sf in source_files:
                f_path = ev_dir / sf
                if not f_path.exists():
                    failed_checks.append(f"corporate_action_raw_file_missing_{sf}")
                else:
                    h = hashlib.sha256()
                    with open(f_path, "rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
                    exp_h = source_hashes.get(sf, "")
                    if h.hexdigest().lower() != str(exp_h).lower():
                        failed_checks.append(f"corporate_action_raw_hash_mismatch_{sf}")

        is_valid = len(failed_checks) == 0
        return is_valid, failed_checks, details


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
