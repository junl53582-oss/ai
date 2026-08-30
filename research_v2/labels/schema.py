"""
Execution-Aligned Label Schema (research_v2/labels/schema.py)
定义实盘执行对齐的高保真前瞻标签 Schema（本阶段仅定义结构与时序约束，不执行批量计算）。
"""
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List


@dataclass
class ExecutionAlignedLabelSchema:
    label_schema_id: str = "EXECUTION_ALIGNED_V1"
    signal_time: str = "T_CLOSE"
    entry_offset_trading_days: int = 1
    entry_price_field: str = "adj_open"
    holding_period_trading_days: int = 20
    exit_offset_trading_days: int = 21
    exit_price_field: str = "adj_open"
    benchmark_entry_field: str = "benchmark_open"
    benchmark_exit_field: str = "benchmark_open"
    transaction_cost_adjusted: bool = True
    primary_label_col: str = "label_net_alpha_20d"
    supported_targets: List[str] = field(default_factory=lambda: [
        "gross_alpha_5d",
        "gross_alpha_20d",
        "net_alpha_5d",
        "net_alpha_20d",
        "direction_20d",
        "rank_20d",
        "downside_20d"
    ])

    def validate(self) -> None:
        """验证时序严格自洽"""
        if self.entry_offset_trading_days + self.holding_period_trading_days != self.exit_offset_trading_days:
            raise ValueError(
                f"Timing mismatch: entry_offset ({self.entry_offset_trading_days}) + "
                f"holding_period ({self.holding_period_trading_days}) != exit_offset ({self.exit_offset_trading_days})"
            )
        if self.signal_time != "T_CLOSE":
            raise ValueError("signal_time must be T_CLOSE")
        if self.entry_price_field not in ("adj_open", "open"):
            raise ValueError("Executable entry must use Open price")
        if self.exit_price_field not in ("adj_open", "open"):
            raise ValueError("Executable exit must use Open price")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_hash(self) -> str:
        self.validate()
        raw_json = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
