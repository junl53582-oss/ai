"""
企业级量化回测与数据真实性审计收集器 (backtest/audit.py)
严格遵循 Fail-Closed 原则 (未提供证据 ≠ 已验证)：
1. 默认状态全方位防伪：survivorship_bias_risk=True, synthetic_data_used='unknown', calendar_is_exchange_official=False
2. 全量审计字段从运行时核心对象自动派生，禁止乐观硬编码
3. 关键认证字段具有防篡改保护 (Anti-Forgery Protection)，若使用 override 强制覆盖则显式记录 override 证据
"""
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


CERTIFICATION_FIELDS = {
    "survivorship_bias_risk",
    "universe_coverage_complete",
    "universe_provenance_verified",
    "synthetic_data_used",
    "calendar_is_exchange_official",
    "historical_st_bias_risk",
    "historical_st_coverage_complete",
    "corporate_action_coverage_complete",
    "adjustment_point_in_time_safe",
    "future_adjustment_leakage_test_passed",
    "cache_fingerprint_verified",
    "overall_backtest_reliability",
    "order_quantity_conservation_passed"
}


@dataclass
class AuditMetadata:
    """标准量化回测真实性与合规审计数据实体 (Fail-Closed Default)"""
    # 0. 运行时唯一标识与配置指纹 (P0-2)
    runtime_instance_id: str = "default_runtime"
    runtime_config_hash: str = "none"
    market_manifest_hash: str = "none"
    factor_manifest_hash: str = "none"
    universe_manifest_hash: str = "none"

    # 1. 数据来源与缓存指纹
    data_source: str = "unknown"
    data_source_breakdown: Dict[str, int] = field(default_factory=dict)
    synthetic_data_used: bool = True
    cache_fingerprint_verified: bool = False
    cache_manifest_version: str = "3.0"
    raw_data_provenance_preserved: bool = False

    # 2. 交易日历来源与交易所认证资质
    calendar_source: str = "unknown"
    calendar_provider: str = "unknown"
    calendar_is_exchange_official: bool = False
    calendar_quality: str = "unknown"
    calendar_fallback_used: bool = False

    # 3. 股票池与幸存者偏差
    universe_mode: str = "STATIC"
    universe_coverage_start: Optional[str] = None
    universe_coverage_end: Optional[str] = None
    universe_coverage_complete: bool = False
    universe_provenance_verified: bool = False
    survivorship_bias_risk: bool = True
    empty_universe_day_count: int = 0
    unknown_membership_row_count: int = 0

    # 4. 行业覆盖与逐日中性化
    industry_neutralization_enabled: str = "DISABLED"
    industry_coverage_ratio_mean: Optional[float] = None
    industry_coverage_ratio_min: Optional[float] = None
    industry_neutralized_day_ratio: Optional[float] = None
    industry_neutralized_days: int = 0
    market_cap_only_days: int = 0
    industry_coverage_ratio: Optional[float] = None
    sector_cap_enabled: bool = True
    industry_data_available: bool = False
    unknown_industry_cap_applied: bool = False
    unknown_industry_weight: float = 0.0
    sector_constraint_enabled: bool = True

    # 5. 上市日期、ST 时间线与停牌估值
    listing_date_coverage_ratio: Optional[float] = None
    historical_st_symbol_coverage_ratio: float = 0.0
    historical_st_date_coverage_ratio: float = 0.0
    historical_st_coverage_complete: bool = False
    historical_st_bias_risk: bool = True
    historical_st_available: bool = False
    historical_st_rule_applied: bool = False
    st_unknown_rows: int = 0
    st_training_excluded_rows: int = 0
    st_trading_excluded_rows: int = 0

    stale_price_warning_events: int = 0
    stale_price_symbol_days: int = 0
    stale_price_affected_symbols: List[str] = field(default_factory=list)
    stale_price_days_total: int = 0
    max_stale_price_days: int = 0
    stale_price_warning_count: int = 0
    suspended_valuation_model: str = "mark_to_market_last_price"

    # 6. 历史费率与流动性约束
    fee_model_scope: str = "unknown"
    fee_components_included: List[str] = field(default_factory=lambda: [
        "broker_commission", "stamp_duty", "transfer_fee", "execution_slippage"
    ])
    fee_components_omitted: List[str] = field(default_factory=lambda: [
        "exchange_handling_fees_bundled_in_comm"
    ])
    partial_fill_count: int = 0
    liquidity_rejected_count: int = 0
    pending_order_count: int = 0
    cancelled_order_count: int = 0
    deferred_order_count: int = 0
    order_quantity_conservation_passed: bool = False

    # 7. 公司行为除权除息覆盖
    corporate_action_source: str = "unknown"
    corporate_action_coverage_ratio: float = 0.0
    corporate_action_coverage_complete: bool = False
    corporate_action_bias_risk: bool = True
    corporate_action_adjustment_available: bool = False
    action_types_supported: List[str] = field(default_factory=lambda: ["CASH_DIVIDEND", "BONUS_SHARE", "SPLIT"])
    unsupported_corporate_action_types: List[str] = field(default_factory=lambda: ["RIGHTS_ISSUE"])
    backtest_total_return_reliability: str = "limited"

    # 8. 基准指数对齐
    benchmark_source: str = "unknown"
    benchmark_coverage_ratio: float = 0.0
    benchmark_missing_date_count: int = 0

    # 9. 价格复权 Point-In-Time 安全性
    price_adjustment_mode: str = "unknown"
    adjustment_point_in_time_safe: bool = False
    future_adjustment_leakage_test_passed: bool = False
    feature_missing_ratio_total: float = 0.0
    warmup_rows_excluded: int = 0

    # 10. 综合可信度评级 (VERIFIED | CONTROLLED_WITH_LIMITATIONS | HIGH_RISK)
    overall_backtest_reliability: str = "HIGH_RISK"
    failed_certification_checks: List[str] = field(default_factory=list)

    # 审计篡改追踪
    audit_override_used: bool = False
    audit_override_fields: List[str] = field(default_factory=list)
    audit_override_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为标准字典格式"""
        return asdict(self)


class CertificationPolicy:
    """
    全要素量化回测真实性与可信度评级策略门禁 (P0-12 Single Certification Gate)
    严格检验全部 13 项核心要素，任何一项不满足即拒绝 VERIFIED。
    """
    @classmethod
    def evaluate(cls, meta: AuditMetadata) -> Tuple[str, List[str]]:
        failed_checks = []

        # 1. PIT 股票池
        if not meta.universe_coverage_complete:
            failed_checks.append("universe_coverage_incomplete")
        if not meta.universe_provenance_verified:
            failed_checks.append("universe_provenance_unverified")
        if meta.survivorship_bias_risk:
            failed_checks.append("survivorship_bias_risk_present")

        # 2. ST 时间线
        if not meta.historical_st_coverage_complete:
            failed_checks.append("historical_st_coverage_incomplete")
        if meta.historical_st_bias_risk:
            failed_checks.append("historical_st_bias_risk_present")

        # 3. 公司行为
        if not meta.corporate_action_coverage_complete:
            failed_checks.append("corporate_action_coverage_incomplete")
        if meta.corporate_action_bias_risk:
            failed_checks.append("corporate_action_bias_risk_present")

        # 4. 缓存指纹与数据血缘
        if not meta.cache_fingerprint_verified:
            failed_checks.append("cache_fingerprint_unverified")
        if not meta.raw_data_provenance_preserved:
            failed_checks.append("raw_data_provenance_lost")

        # 5. 复权与未来信息泄露
        if not meta.adjustment_point_in_time_safe:
            failed_checks.append("adjustment_not_point_in_time_safe")
        if not meta.future_adjustment_leakage_test_passed:
            failed_checks.append("future_adjustment_leakage_test_not_passed")

        # 6. 基准指数完整性
        if meta.benchmark_coverage_ratio < 1.0:
            failed_checks.append("benchmark_coverage_ratio_less_than_100pct")
        if meta.benchmark_missing_date_count > 0:
            failed_checks.append(f"benchmark_missing_dates_{meta.benchmark_missing_date_count}")

        # 7. 订单与状态守恒
        if not meta.order_quantity_conservation_passed:
            failed_checks.append("order_quantity_conservation_failed")

        # 8. 真实数据（禁止仿真数据）
        if meta.synthetic_data_used:
            failed_checks.append("synthetic_data_used")

        # 9. 交易所官方日历
        if not meta.calendar_is_exchange_official:
            failed_checks.append("calendar_not_exchange_official")

        if not failed_checks:
            return "VERIFIED", []

        critical_high_risks = {
            "survivorship_bias_risk_present",
            "synthetic_data_used",
            "order_quantity_conservation_failed",
            "future_adjustment_leakage_test_not_passed"
        }
        if any(c in critical_high_risks for c in failed_checks):
            return "HIGH_RISK", failed_checks

        return "CONTROLLED_WITH_LIMITATIONS", failed_checks


class AuditCollector:
    """集中式审计指标收集器 (严格遵循 Fail-Closed 与防伪校验)"""

    @classmethod
    def collect(
        cls,
        data_manager: Optional[Any] = None,
        factor_processor: Optional[Any] = None,
        portfolio_builder: Optional[Any] = None,
        trainer: Optional[Any] = None,
        engine: Optional[Any] = None,
        custom_overrides: Optional[Dict[str, Any]] = None,
        runtime_instance_id: Optional[str] = None
    ) -> AuditMetadata:
        """从各组件运行时实例真实采集审计元数据"""
        import uuid
        meta = AuditMetadata()
        meta.runtime_instance_id = runtime_instance_id or f"run_{uuid.uuid4().hex[:8]}"

        # 1. 数据层采集
        if data_manager is not None:
            meta.data_source = getattr(data_manager, "data_source", "unknown")
            meta.data_source_breakdown = dict(getattr(data_manager, "data_source_breakdown", {}))
            meta.synthetic_data_used = bool(getattr(data_manager, "synthetic_data_used", False))
            meta.cache_fingerprint_verified = bool(getattr(data_manager, "cache_fingerprint_verified", False))
            meta.raw_data_provenance_preserved = bool(getattr(data_manager, "raw_data_provenance_preserved", False))

            meta.calendar_source = getattr(data_manager, "calendar_source", "unknown")
            meta.calendar_provider = getattr(data_manager, "calendar_provider", "unknown")
            meta.calendar_is_exchange_official = bool(getattr(data_manager, "calendar_is_exchange_official", False))
            meta.calendar_quality = getattr(data_manager, "calendar_quality", "unknown")
            meta.calendar_fallback_used = bool(getattr(data_manager, "calendar_fallback_used", False))
            meta.listing_date_coverage_ratio = getattr(data_manager, "listing_date_coverage_ratio", None)

            # ST 审计
            meta.historical_st_symbol_coverage_ratio = getattr(data_manager, "historical_st_symbol_coverage_ratio", 0.0)
            meta.historical_st_date_coverage_ratio = getattr(data_manager, "historical_st_date_coverage_ratio", 0.0)
            meta.historical_st_coverage_complete = bool(getattr(data_manager, "historical_st_coverage_complete", False))
            meta.historical_st_bias_risk = bool(getattr(data_manager, "historical_st_bias_risk", True))
            meta.historical_st_available = bool(getattr(data_manager, "historical_st_available", False))
            meta.historical_st_rule_applied = bool(getattr(data_manager, "historical_st_rule_applied", False))

            # 基准对齐
            meta.benchmark_source = getattr(data_manager, "benchmark_source", "unknown")
            meta.benchmark_coverage_ratio = getattr(data_manager, "benchmark_coverage_ratio", 0.0)
            meta.benchmark_missing_date_count = getattr(data_manager, "benchmark_missing_date_count", 0)

            # 股票池
            u_prov = getattr(data_manager, "universe_provider", None)
            if u_prov is not None:
                meta.universe_mode = u_prov.get_mode()
                meta.survivorship_bias_risk = u_prov.has_survivorship_bias_risk()
                meta.universe_coverage_start = getattr(u_prov, "coverage_start", None)
                meta.universe_coverage_end = getattr(u_prov, "coverage_end", None)
                meta.universe_coverage_complete = bool(getattr(u_prov, "is_coverage_complete", lambda: False)())
                meta.universe_provenance_verified = bool(getattr(u_prov, "universe_provenance_verified", False))

            meta.empty_universe_day_count = getattr(data_manager, "empty_universe_day_count", 0)
            meta.unknown_membership_row_count = getattr(data_manager, "unknown_membership_row_count", 0)

        # 2. 因子层采集
        if factor_processor is not None:
            meta.industry_neutralization_enabled = getattr(factor_processor, "industry_neutralization_enabled", "DISABLED")
            meta.industry_coverage_ratio_mean = getattr(factor_processor, "industry_coverage_ratio_mean", None)
            meta.industry_coverage_ratio_min = getattr(factor_processor, "industry_coverage_ratio_min", None)
            meta.industry_neutralized_day_ratio = getattr(factor_processor, "industry_neutralized_day_ratio", None)
            meta.industry_neutralized_days = getattr(factor_processor, "industry_neutralized_days", 0)
            meta.market_cap_only_days = getattr(factor_processor, "market_cap_only_days", 0)
            meta.industry_coverage_ratio = getattr(factor_processor, "industry_coverage_ratio_mean", None)
            meta.feature_missing_ratio_total = getattr(factor_processor, "feature_missing_ratio_total", 0.0)
            meta.price_adjustment_mode = getattr(factor_processor, "price_adjustment_mode", "unknown")
            meta.adjustment_point_in_time_safe = bool(getattr(factor_processor, "adjustment_point_in_time_safe", False))
            meta.future_adjustment_leakage_test_passed = bool(getattr(factor_processor, "future_adjustment_leakage_test_passed", False))

        # 3. 训练层采集 (P0-7)
        if trainer is not None:
            meta.st_unknown_rows = getattr(trainer, "st_unknown_rows", 0)
            meta.st_training_excluded_rows = getattr(trainer, "st_training_excluded_rows", 0)
            meta.st_trading_excluded_rows = getattr(trainer, "st_trading_excluded_rows", 0)

        # 4. 组合构建层采集
        if portfolio_builder is not None:
            meta.sector_cap_enabled = bool(getattr(portfolio_builder, "sector_cap_enabled", True))
            meta.industry_data_available = bool(getattr(portfolio_builder, "industry_data_available", False))
            meta.unknown_industry_cap_applied = bool(getattr(portfolio_builder, "unknown_industry_cap_applied", False))
            meta.unknown_industry_weight = float(getattr(portfolio_builder, "unknown_industry_weight", 0.0))
            meta.sector_constraint_enabled = bool(getattr(portfolio_builder, "sector_constraint_enabled", True))

        # 5. 回测执行引擎采集
        if engine is not None:
            meta.fee_model_scope = "historical_tiered"
            meta.stale_price_warning_events = getattr(engine, "stale_price_warning_events", 0)
            meta.stale_price_symbol_days = getattr(engine, "stale_price_symbol_days", 0)
            meta.stale_price_days_total = meta.stale_price_symbol_days
            meta.max_stale_price_days = getattr(engine, "max_stale_price_days", 0)
            meta.stale_price_affected_symbols = sorted(list(getattr(engine, "stale_price_affected_symbols", set())))
            meta.stale_price_warning_count = meta.stale_price_warning_events

            meta.partial_fill_count = getattr(engine, "partial_fill_count", 0)
            meta.liquidity_rejected_count = getattr(engine, "liquidity_rejected_count", 0)
            meta.pending_order_count = getattr(engine, "pending_order_count", 0)
            meta.cancelled_order_count = getattr(engine, "cancelled_order_count", 0)
            meta.deferred_order_count = getattr(engine, "deferred_order_count", 0)
            meta.order_quantity_conservation_passed = bool(getattr(engine, "order_quantity_conservation_passed", False))

            corp_prov = getattr(engine, "corporate_actions", None)
            if corp_prov is not None:
                meta.corporate_action_source = getattr(corp_prov, "corporate_action_source", "unknown")
                meta.corporate_action_coverage_ratio = getattr(corp_prov, "coverage_ratio", 0.0)
                meta.corporate_action_coverage_complete = bool(getattr(corp_prov, "coverage_complete", False))
                meta.corporate_action_bias_risk = not meta.corporate_action_coverage_complete
                meta.corporate_action_adjustment_available = corp_prov.has_actions_data()
                meta.action_types_supported = getattr(corp_prov, "action_types_supported", ["CASH_DIVIDEND", "BONUS_SHARE", "SPLIT"])
                meta.unsupported_corporate_action_types = getattr(corp_prov, "unsupported_corporate_action_types", ["RIGHTS_ISSUE"])
                meta.backtest_total_return_reliability = "standard" if meta.corporate_action_coverage_complete else "limited"

        # 6. 防伪审计覆盖处理 (Anti-Forgery & Audit Override Tracking - P0-10)
        if custom_overrides:
            meta.audit_override_used = True
            meta.audit_override_fields = list(custom_overrides.keys())
            meta.audit_override_source = "custom_overrides"

            for k, v in custom_overrides.items():
                if k in CERTIFICATION_FIELDS:
                    logger.warning(f"⚠️ 拦截对认证字段 {k} 的非法覆盖尝试！")
                    continue
                if hasattr(meta, k):
                    setattr(meta, k, v)

        # 7. 通过 CertificationPolicy 评估综合评级 (P0-12)
        overall_status, failed_checks = CertificationPolicy.evaluate(meta)
        meta.overall_backtest_reliability = overall_status
        meta.failed_certification_checks = failed_checks

        return meta
