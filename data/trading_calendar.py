"""
权威交易所交易日历服务 (data/trading_calendar.py)
严格从物理 canonical_calendar_v1.parquet 及其 Manifest 加载并校验 SHA-256。
提供 fail-closed 的交易日查询、区间交易日统计与下一个交易日计算。
去推断化 (De-inference Hardened):
- 彻底移除 dynamic weekday() < 5 与硬编码 holiday 字典推断。
- 只有物理存在于 canonical_calendar_v1.parquet 中的日期才是有效交易日。
- 物理覆盖之外的日期强制标记为 CALENDAR_COVERAGE_BLOCKED 并 fail-closed。
"""
from pathlib import Path
import json
import hashlib
from typing import Union, List, Optional
from datetime import datetime, date
import pandas as pd
from config.settings import settings


class TradeDateError(ValueError):
    """交易日历校验异常"""
    pass


class CanonicalTradingCalendar:
    _instance = None
    COVERAGE_STATUS_VALID = "CALENDAR_COVERAGE_VALID"
    COVERAGE_STATUS_BLOCKED = "CALENDAR_COVERAGE_BLOCKED"

    def __init__(self, parquet_path: Optional[Union[str, Path]] = None, manifest_path: Optional[Union[str, Path]] = None):
        if parquet_path is None:
            parquet_path = Path(settings.BASE_DIR) / "data_storage" / "reference" / "canonical_calendar_v1.parquet"
        if manifest_path is None:
            manifest_path = Path(settings.BASE_DIR) / "data_storage" / "reference" / "canonical_calendar_v1.manifest.json"

        self.parquet_path = Path(parquet_path)
        self.manifest_path = Path(manifest_path)
        self._load_and_verify()

    def _load_and_verify(self):
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"权威交易日历文件不存在: {self.parquet_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"权威交易日历 Manifest 不存在: {self.manifest_path}")

        # 1. 验证 Manifest SHA-256
        file_bytes = self.parquet_path.read_bytes()
        actual_sha256 = hashlib.sha256(file_bytes).hexdigest().lower()

        try:
            manifest_data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"日历 Manifest 解析失败: {e}")

        expected_sha256 = manifest_data.get("calendar_artifact_sha256", "").lower()
        if not expected_sha256 or actual_sha256 != expected_sha256:
            raise ValueError(f"交易日历 SHA-256 校验失败: 实际 {actual_sha256} != Manifest 声明 {expected_sha256} (fail-closed)")

        # 2. 校验 schema 并仅加载物理存在的交易日
        df = pd.read_parquet(self.parquet_path)
        required_cols = {"date", "is_trading_day"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"日历文件 schema 不合规，缺少必需列: {required_cols - set(df.columns)}")

        # 严格仅保留物理 parquet 中标记为 True 的交易日，严禁任何时序延伸与 weekday 推断
        valid_trading_days = df[df["is_trading_day"] == True]["date"].astype(str).tolist()

        # 确保严格升序且无重复
        self.trading_days_list: List[str] = sorted(list(dict.fromkeys(valid_trading_days)))
        self.trading_days_set: set[str] = set(self.trading_days_list)
        self.date_min: str = self.trading_days_list[0] if self.trading_days_list else ""
        self.date_max: str = self.trading_days_list[-1] if self.trading_days_list else ""

    @classmethod
    def get_instance(cls) -> "CanonicalTradingCalendar":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _to_date_str(d: Union[str, datetime, date, pd.Timestamp]) -> str:
        if isinstance(d, (datetime, date, pd.Timestamp)):
            return d.strftime("%Y-%m-%d")
        d_str = str(d).strip()
        return d_str[:10]

    def check_coverage(self, d: Union[str, datetime, date, pd.Timestamp]) -> str:
        """检查给定日期是否在物理权威日历覆盖范围内"""
        d_str = self._to_date_str(d)
        if not self.date_min or not self.date_max:
            return self.COVERAGE_STATUS_BLOCKED
        if d_str < self.date_min or d_str > self.date_max:
            return self.COVERAGE_STATUS_BLOCKED
        return self.COVERAGE_STATUS_VALID

    def is_trading_day(self, d: Union[str, datetime, date, pd.Timestamp]) -> bool:
        """
        判断给定日期是否为交易所合法交易日:
        所有返回 True 的日期必须物理存在于 canonical_calendar_v1.parquet 中。
        超出物理覆盖范围 (截至 date_max) 的日期返回 False (Fail-Closed)。
        """
        d_str = self._to_date_str(d)
        return d_str in self.trading_days_set

    def validate_trading_day(self, d: Union[str, datetime, date, pd.Timestamp]) -> bool:
        """严格校验合法交易日，若超出物理覆盖或为休市日则强制抛出 TradeDateError (Fail-Closed)"""
        d_str = self._to_date_str(d)
        if self.check_coverage(d_str) == self.COVERAGE_STATUS_BLOCKED:
            raise TradeDateError(f"CALENDAR_COVERAGE_BLOCKED: 日期 {d_str} 超出权威交易日历物理覆盖范围 ({self.date_min} ~ {self.date_max})，禁止凭空推断未来交易日 (fail-closed)")
        if not self.is_trading_day(d_str):
            raise TradeDateError(f"非交易所交易日: {d_str} (fail-closed)")
        return True

    def trading_days_between(self, start: Union[str, datetime, date, pd.Timestamp], end: Union[str, datetime, date, pd.Timestamp]) -> List[str]:
        """计算闭区间 [start, end] 内的真实物理交易日列表"""
        s_str = self._to_date_str(start)
        e_str = self._to_date_str(end)
        if s_str > e_str:
            return []
        import bisect
        idx_start = bisect.bisect_left(self.trading_days_list, s_str)
        idx_end = bisect.bisect_right(self.trading_days_list, e_str)
        return self.trading_days_list[idx_start:idx_end]

    def next_trading_day(self, d: Union[str, datetime, date, pd.Timestamp]) -> Optional[str]:
        """
        获取下一个严格大于 d 的物理交易日。
        若 d 已达到或超出物理日历上限，禁止推断未来 session，返回 None (Fail-Closed)。
        """
        d_str = self._to_date_str(d)
        import bisect
        idx = bisect.bisect_right(self.trading_days_list, d_str)
        if idx < len(self.trading_days_list):
            return self.trading_days_list[idx]
        return None

    def prev_trading_day(self, d: Union[str, datetime, date, pd.Timestamp]) -> Optional[str]:
        """
        获取上一个严格小于 d 的物理交易日。
        若无历史交易日，返回 None (Fail-Closed)。
        """
        d_str = self._to_date_str(d)
        import bisect
        idx = bisect.bisect_left(self.trading_days_list, d_str) - 1
        if 0 <= idx < len(self.trading_days_list):
            return self.trading_days_list[idx]
        return None
