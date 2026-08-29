"""
公司行为 (Corporate Actions) 数据结构与除权除息调度器 (strategy/corporate_actions.py)
处理现金分红、送红股、转增股与股票拆细等公司行为，
提供全量股票池覆盖完整性 (Coverage-Aware) 验证与除息除权价值守恒调度
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Set, Any
import pandas as pd

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


@dataclass
class CorporateActionCoverageRecord:
    """公司行为数据源覆盖证明 (P0-13 Coverage Record)"""
    symbol: str
    query_start: str
    query_end: str
    query_success: bool = True
    source: str = "custom_corporate_actions"
    source_version: str = "1.0"
    empty_result_verified: bool = True
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
        self.corporate_action_source: str = source
        self.coverage_start: Optional[str] = pd.to_datetime(coverage_start).strftime("%Y-%m-%d") if coverage_start else None
        self.coverage_end: Optional[str] = pd.to_datetime(coverage_end).strftime("%Y-%m-%d") if coverage_end else None
        self.covered_symbols: Set[str] = set()
        self.required_symbols: Set[str] = set()
        self.coverage_ratio: float = 0.0
        self.coverage_complete: bool = False
        
        self.action_types_supported: List[str] = ["CASH_DIVIDEND", "BONUS_SHARE", "SPLIT"]
        self.unsupported_corporate_action_types: List[str] = ["RIGHTS_ISSUE"]

    def register_coverage_record(self, record: CorporateActionCoverageRecord):
        """注册数据源对特定股票在特定区间的完整性覆盖证明 (P0-13)"""
        sym = record.symbol.strip().upper()
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
        验证回测区间与所有候选股票的公司行为覆盖完整性 (P0-8)：
        必须每个 required_symbol 均拥有 query_success=True 且覆盖 [start_date, end_date] 的 CorporateActionCoverageRecord。
        即使存在分红事件，若缺失显式 CoverageRecord，仍判定为 False。
        """
        self.required_symbols = set(s.strip().upper() for s in required_symbols)
        if not self.required_symbols:
            self.coverage_complete = False
            self.coverage_ratio = 0.0
            return False

        # 统计有效覆盖证明的交集
        valid_covered_syms = set()
        s_req = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        e_req = pd.to_datetime(end_date).strftime("%Y-%m-%d")

        for sym in self.required_symbols:
            if sym in self.coverage_records:
                rec = self.coverage_records[sym]
                if rec.query_success and rec.query_start <= s_req and rec.query_end >= e_req:
                    valid_covered_syms.add(sym)

        self.coverage_ratio = round(len(valid_covered_syms) / len(self.required_symbols), 4)

        if len(valid_covered_syms) == len(self.required_symbols):
            self.coverage_complete = True
            return True

        self.coverage_complete = False
        return False


    def has_actions_data(self) -> bool:
        """是否有注册的公司行为数据"""
        return self._action_count > 0


def create_corporate_action_provider(
    config: Optional[Any] = None,
    actions_df: Optional[pd.DataFrame] = None,
    coverage_records: Optional[List[CorporateActionCoverageRecord]] = None,
    source: str = "custom_corporate_actions"
) -> CorporateActionProvider:
    """
    统一公司行为提供器工厂函数 (P0-9)
    支持从数据存储加载 actions 与 coverage records。
    若无真实数据文件，返回默认未覆盖 provider，并在运行时如实报告 CONTROLLED_WITH_LIMITATIONS。
    """
    from config.settings import settings as default_settings
    cfg = config or default_settings
    provider = CorporateActionProvider(source=source)

    ca_dir = getattr(cfg, "DATA_DIR", None)
    if ca_dir:
        ca_file = ca_dir / "corporate_actions.parquet"
        cov_file = ca_dir / "corporate_action_coverage.json"

        if ca_file.exists():
            try:
                df = pd.read_parquet(ca_file)
                for _, row in df.iterrows():
                    provider.register_action(CorporateAction(
                        symbol=str(row["symbol"]),
                        ex_date=str(row["ex_date"]),
                        action_type=str(row["action_type"]),
                        cash_dividend_per_share=float(row.get("cash_dividend_per_share", 0.0)),
                        share_ratio=float(row.get("share_ratio", 0.0))
                    ))
            except Exception as e:
                logger.warning(f"加载公司行为数据失败: {e}")

        if cov_file.exists():
            try:
                import json
                with open(cov_file, "r", encoding="utf-8") as f:
                    cov_data = json.load(f)
                for item in cov_data:
                    provider.register_coverage_record(CorporateActionCoverageRecord(
                        symbol=item["symbol"],
                        query_start=item["query_start"],
                        query_end=item["query_end"],
                        query_success=item.get("query_success", True),
                        source=item.get("source", source),
                        empty_result_verified=item.get("empty_result_verified", True)
                    ))
            except Exception as e:
                logger.warning(f"加载公司行为覆盖证明失败: {e}")

    if actions_df is not None and not actions_df.empty:
        for _, row in actions_df.iterrows():
            provider.register_action(CorporateAction(
                symbol=str(row["symbol"]),
                ex_date=str(row["ex_date"]),
                action_type=str(row["action_type"]),
                cash_dividend_per_share=float(row.get("cash_dividend_per_share", 0.0)),
                share_ratio=float(row.get("share_ratio", 0.0))
            ))

    if coverage_records:
        for rec in coverage_records:
            provider.register_coverage_record(rec)

    return provider
