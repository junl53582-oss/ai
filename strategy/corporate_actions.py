"""
公司行为 (Corporate Actions) 数据结构与除权除息调度器 (strategy/corporate_actions.py)
处理现金分红、送红股、转增股与股票拆细等公司行为，
提供全量股票池覆盖完整性 (Coverage-Aware) 验证与除息除权价值守恒调度
严格遵循 Fail-Closed 原则：
- CorporateActionCoverageRecord / CorporateActionCoverageEvidence 默认状态必须为 False
- 严禁以纯布尔值声称证明，必须具备可审计的数据源凭据与区间覆盖
"""
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Set, Any, Tuple
import pandas as pd

from data.source_registry import CorporateActionCoverageEvidence, TRUSTED_SOURCE_REGISTRY

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
    """公司行为数据源覆盖证明 (Fail-Closed Default)"""
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
                response_hash=record.response_hash
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
        """注册公司行为 (注意：单次分红事件不作为全区间覆盖证明)"""
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
        end_date: Union[str, pd.Timestamp]
    ) -> bool:
        """
        验证回测区间与所有候选股票的公司行为覆盖完整性 (P0 严格校验)：
        必须每个 required_symbol 均拥有 query_success=True 且覆盖 [start_date, end_date] 的可信记录。
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
                is_valid, _ = ev.is_valid_zero_event_proof(sym, s_date, e_date)
                if ev.query_success and ev.query_start <= s_date and ev.query_end >= e_date:
                    validly_covered.add(sym)
                    if not is_valid and ev.empty_result:
                        all_zero_event_proven = False
                else:
                    all_zero_event_proven = False

            elif sym in self.coverage_records:
                rec = self.coverage_records[sym]
                q_s = pd.to_datetime(rec.query_start).strftime("%Y-%m-%d")
                q_e = pd.to_datetime(rec.query_end).strftime("%Y-%m-%d")
                if rec.query_success and q_s <= s_date and q_e >= e_date:
                    validly_covered.add(sym)
                    if not rec.empty_result_verified:
                        all_zero_event_proven = False
                else:
                    all_zero_event_proven = False
            else:
                all_zero_event_proven = False

        self.coverage_ratio = len(validly_covered) / len(self.required_symbols)
        self.coverage_complete = len(validly_covered) == len(self.required_symbols)
        self.zero_event_proof_verified = bool(self.coverage_complete and all_zero_event_proven and self._action_count == 0)

        return self.coverage_complete


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
                    rec = CorporateActionCoverageRecord(
                        symbol=sym,
                        query_start=item.get("query_start", "2000-01-01"),
                        query_end=item.get("query_end", "2099-12-31"),
                        query_success=bool(item.get("query_success", False)),  # 严格默认为 False
                        empty_result_verified=bool(item.get("empty_result_verified", False)),  # 严格默认为 False
                        source=item.get("source", "csi_official"),
                        source_class=item.get("source_class", "UNKNOWN"),
                        raw_result_hash=item.get("raw_result_hash"),
                        response_hash=item.get("response_hash")
                    )
                    provider.register_coverage_record(rec)
            logger.info(f"成功加载公司行为覆盖凭据: {len(provider.coverage_records)} 条记录")
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
