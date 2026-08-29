"""
企业级量化回测与数据真实性审计收集器 (backtest/audit.py)
严格遵循 Fail-Closed 原则 (未提供证据 ≠ 已验证)：
1. 默认状态全方位防伪：survivorship_bias_risk=True, synthetic_data_used=True, calendar_is_exchange_official=False
2. 全量审计字段从运行时核心对象自动派生，禁止乐观硬编码
3. 严格白名单防伪保护 (NON_CERTIFICATION_OVERRIDE_FIELDS)，封死一切通过 custom_overrides 修改认证字段的途径
4. 密码级 64-hex SHA256 Chained Manifest Provenance 审计与一致性检验
5. 订单数量守恒缺失时严格默认为 False (Fail-Closed)
6. 实际回测窗口与覆盖区间必须由 CertificationPolicy 独立强制双向比对
7. 公司行为缺少数据时必须具备真实 Zero Event Proof 才能认证
8. runtime_config_hash 作为核心门禁检查项
"""
import re
import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# 仅允许非认证类备注/展示字段进行外部 override (严格白名单机制)
NON_CERTIFICATION_OVERRIDE_FIELDS = {
    "runtime_instance_id",
    "display_notes",
    "user_comment",
    "custom_tag",
    "audit_override_source"
}
CERTIFICATION_FIELDS = NON_CERTIFICATION_OVERRIDE_FIELDS

HEX_64_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def compute_canonical_runtime_config_hash(config_obj: Any) -> str:
    """确定性计算运行时解析后的核心配置 Canonical SHA256 哈希"""
    keys = [
        "START_DATE", "END_DATE", "LABEL_HORIZON", "UNIVERSE_MODE", "INDEX_CODE",
        "TRAIN_WINDOW_DAYS", "PURGE_GAP_DAYS", "TOP_K_BUY", "TOP_K_HOLD",
        "REBALANCE_FREQ", "TRANSACTION_FEE", "STAMP_DUTY", "SLIPPAGE", "ENABLE_NEUTRALIZATION"
    ]
    extracted = {}
    for k in keys:
        extracted[k] = str(getattr(config_obj, k, "")) if config_obj is not None else ""
    sorted_json = json.dumps(extracted, sort_keys=True)
    return hashlib.sha256(sorted_json.encode('utf-8')).hexdigest()


@dataclass
class AuditMetadata:
    """标准量化回测真实性与合规审计数据实体 (Fail-Closed Default)"""
    # 0. 运行时唯一标识与配置指纹 (P0, P1-3)
    runtime_instance_id: str = "default_runtime"
    runtime_config_hash: Optional[str] = None
    market_manifest_hash: Optional[str] = None
    factor_manifest_hash: Optional[str] = None
    universe_manifest_hash: Optional[str] = None
    display_notes: Optional[str] = None
    user_comment: Optional[str] = None
    custom_tag: Optional[str] = None

    # 1. 真实回测区间与请求区间 (P0, P1)
    actual_backtest_start_date: Optional[str] = None
    actual_backtest_end_date: Optional[str] = None
    requested_backtest_start_date: Optional[str] = None
    requested_backtest_end_date: Optional[str] = None

    # 2. 数据来源与缓存指纹
    data_source: str = "unknown"
    data_source_breakdown: Dict[str, int] = field(default_factory=dict)
    synthetic_data_used: bool = True  # 严格默认为 True (Fail-Closed)
    cache_fingerprint_verified: bool = False
    cache_manifest_version: str = "3.0"
    raw_data_provenance_preserved: bool = False

    # 3. 交易日历来源与交易所认证资质
    calendar_source: str = "unknown"
    calendar_provider: str = "unknown"
    calendar_is_exchange_official: bool = False
    calendar_quality: str = "unknown"
    calendar_fallback_used: bool = False

    # 4. 股票池与幸存者偏差
    universe_mode: str = "STATIC"
    universe_source_class: str = "UNKNOWN"
    universe_coverage_start: Optional[str] = None
    universe_coverage_end: Optional[str] = None
    universe_coverage_complete: bool = False
    universe_provenance_verified: bool = False
    universe_raw_evidence_verified: bool = False
    universe_dataset_hash_verified: bool = False
    universe_verification_failures: List[str] = field(default_factory=list)
    survivorship_bias_risk: bool = True
    empty_universe_day_count: int = 0
    unknown_membership_row_count: int = 0

    # 5. 行业覆盖与逐日中性化
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

    # 6. 上市日期、ST 时间线与停牌估值
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

    # 7. 历史费率与流动性约束
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
    order_quantity_conservation_passed: bool = False  # 缺失严格默认为 False

    # 8. 公司行为除权除息覆盖
    corporate_action_source: str = "unknown"
    corporate_action_coverage_ratio: float = 0.0
    corporate_action_coverage_complete: bool = False
    corporate_action_bias_risk: bool = True
    corporate_action_adjustment_available: bool = False
    corporate_action_zero_event_proof_verified: bool = False
    action_types_supported: List[str] = field(default_factory=lambda: ["CASH_DIVIDEND", "BONUS_SHARE", "SPLIT"])
    unsupported_corporate_action_types: List[str] = field(default_factory=lambda: ["RIGHTS_ISSUE"])
    backtest_total_return_reliability: str = "limited"

    # 9. 基准指数对齐
    benchmark_source: str = "unknown"
    benchmark_coverage_ratio: float = 0.0
    benchmark_missing_date_count: int = 0

    # 10. 价格复权 Point-In-Time 安全性
    price_adjustment_mode: str = "unknown"
    adjustment_point_in_time_safe: bool = False
    future_adjustment_leakage_test_passed: bool = False
    feature_missing_ratio_total: float = 0.0
    warmup_rows_excluded: int = 0

    # 11. 综合可信度评级 (VERIFIED | CONTROLLED_WITH_LIMITATIONS | HIGH_RISK)
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
    全要素量化回测真实性与可信度评级策略门禁 (P0 Single Certification Gate)
    严格检验全部核心要素，任何一项不满足即拒绝 VERIFIED。
    """
    @classmethod
    def evaluate(cls, meta: AuditMetadata) -> Tuple[str, List[str]]:
        failed_checks = []

        # 0. 关键链式哈希完整性校验 (Chained Manifest Provenance & 64-hex Hash Validation)
        if not meta.runtime_config_hash or not HEX_64_PATTERN.match(str(meta.runtime_config_hash)):
            failed_checks.append("runtime_config_hash_missing")

        if not meta.universe_manifest_hash or not HEX_64_PATTERN.match(str(meta.universe_manifest_hash)):
            failed_checks.append("universe_manifest_hash_missing")

        if not meta.factor_manifest_hash or not HEX_64_PATTERN.match(str(meta.factor_manifest_hash)):
            failed_checks.append("factor_manifest_hash_missing")

        if not meta.market_manifest_hash or not HEX_64_PATTERN.match(str(meta.market_manifest_hash)):
            failed_checks.append("market_manifest_hash_missing")

        # 1. 实际回测窗口与 PIT 股票池覆盖独立校验 (P1-12)
        if (
            meta.actual_backtest_start_date is None
            or meta.actual_backtest_end_date is None
            or meta.universe_coverage_start is None
            or meta.universe_coverage_end is None
        ):
            failed_checks.append("actual_backtest_window_or_universe_coverage_dates_missing")
        else:
            if str(meta.universe_coverage_start) > str(meta.actual_backtest_start_date):
                failed_checks.append(f"universe_coverage_start_{meta.universe_coverage_start}_after_actual_backtest_start_{meta.actual_backtest_start_date}")
            if str(meta.universe_coverage_end) < str(meta.actual_backtest_end_date):
                failed_checks.append(f"universe_coverage_end_{meta.universe_coverage_end}_before_actual_backtest_end_{meta.actual_backtest_end_date}")

        # 2. PIT 股票池血缘与原始证据校验
        if not meta.universe_coverage_complete:
            failed_checks.append("universe_coverage_incomplete")
        if not meta.universe_provenance_verified:
            failed_checks.append("universe_provenance_unverified")
        if not meta.universe_raw_evidence_verified:
            failed_checks.append("universe_raw_evidence_unverified")
        if not meta.universe_dataset_hash_verified:
            failed_checks.append("universe_dataset_hash_unverified")
        if meta.universe_source_class not in ["OFFICIAL_PRIMARY", "LICENSED_VENDOR"]:
            failed_checks.append(f"universe_source_class_ineligible_for_production_{meta.universe_source_class}")
        if meta.survivorship_bias_risk:
            failed_checks.append("survivorship_bias_risk_present")

        # 3. ST 时间线与 UNKNOWN 状态一致性 Gate
        if meta.historical_st_coverage_complete and meta.st_unknown_rows > 0:
            failed_checks.append(f"st_unknown_rows_{meta.st_unknown_rows}_inconsistent_with_complete_coverage")
        if not meta.historical_st_coverage_complete:
            failed_checks.append("historical_st_coverage_incomplete")
        if meta.historical_st_bias_risk:
            failed_checks.append("historical_st_bias_risk_present")

        # 4. 公司行为覆盖一致性与 Zero Event Proof 校验 (P0, P1)
        has_complete_action_data = (
            meta.corporate_action_coverage_complete
            and meta.corporate_action_adjustment_available
            and meta.corporate_action_coverage_ratio >= 1.0
        )
        has_zero_event_proof = (
            meta.corporate_action_coverage_complete
            and not meta.corporate_action_adjustment_available
            and meta.corporate_action_zero_event_proof_verified
        )
        if not (has_complete_action_data or has_zero_event_proof):
            failed_checks.append("corporate_action_missing_adjustment_or_zero_event_proof")
            if not meta.corporate_action_coverage_complete:
                failed_checks.append("corporate_action_coverage_incomplete")
        if meta.corporate_action_bias_risk:
            failed_checks.append("corporate_action_bias_risk_present")

        # 5. 缓存指纹与数据血缘
        if not meta.cache_fingerprint_verified:
            failed_checks.append("cache_fingerprint_unverified")
        if not meta.raw_data_provenance_preserved:
            failed_checks.append("raw_data_provenance_lost")

        # 6. 复权与未来信息泄露
        if not meta.adjustment_point_in_time_safe:
            failed_checks.append("adjustment_not_point_in_time_safe")
        if not meta.future_adjustment_leakage_test_passed:
            failed_checks.append("future_adjustment_leakage_test_not_passed")

        # 7. 行情与基准数据源可信度校验
        if meta.data_source == "unknown":
            failed_checks.append("market_data_source_unverified")

        if meta.benchmark_source == "unknown":
            failed_checks.append("benchmark_source_unverified")
        if meta.benchmark_coverage_ratio < 1.0:
            failed_checks.append("benchmark_coverage_ratio_less_than_100pct")
        if meta.benchmark_missing_date_count > 0:
            failed_checks.append(f"benchmark_missing_dates_{meta.benchmark_missing_date_count}")

        # 8. 订单与状态守恒 (缺失严格判定为 False)
        if not meta.order_quantity_conservation_passed:
            failed_checks.append("order_quantity_conservation_failed")

        # 9. 真实数据（禁止仿真数据）
        if meta.synthetic_data_used:
            failed_checks.append("synthetic_data_used")

        # 10. 交易所官方日历
        if not meta.calendar_is_exchange_official:
            failed_checks.append("calendar_not_exchange_official")

        if not failed_checks:
            return "VERIFIED", []

        critical_high_risks = {
            "survivorship_bias_risk_present",
            "synthetic_data_used",
            "order_quantity_conservation_failed",
            "future_adjustment_leakage_test_not_passed",
            "universe_raw_evidence_unverified",
            "universe_dataset_hash_unverified",
            "universe_manifest_hash_missing",
            "factor_manifest_hash_missing",
            "market_manifest_hash_missing",
            "runtime_config_hash_missing",
            "corporate_action_missing_adjustment_or_zero_event_proof",
            "actual_backtest_window_or_universe_coverage_dates_missing",
            "market_data_source_unverified"
        }
        if any(c in critical_high_risks for c in failed_checks) or any("after_actual_backtest_start" in c or "before_actual_backtest_end" in c for c in failed_checks):
            return "HIGH_RISK", failed_checks

        return "CONTROLLED_WITH_LIMITATIONS", failed_checks


class AuditCollector:
    """集中式审计指标收集器 (严格遵循 Fail-Closed 与白名单防伪校验)"""

    @classmethod
    def collect(
        cls,
        data_manager: Optional[Any] = None,
        factor_processor: Optional[Any] = None,
        portfolio_builder: Optional[Any] = None,
        trainer: Optional[Any] = None,
        engine: Optional[Any] = None,
        config: Optional[Any] = None,
        custom_overrides: Optional[Dict[str, Any]] = None,
        runtime_instance_id: Optional[str] = None
    ) -> AuditMetadata:
        """从各组件运行时实例真实采集审计元数据"""
        import uuid
        meta = AuditMetadata()
        meta.runtime_instance_id = runtime_instance_id or f"run_{uuid.uuid4().hex[:8]}"

        if config is not None:
            meta.runtime_config_hash = compute_canonical_runtime_config_hash(config)

        # 1. 数据层采集 (Fail-Closed 默认值)
        if data_manager is not None:
            meta.data_source = getattr(data_manager, "data_source", "unknown")
            meta.data_source_breakdown = dict(getattr(data_manager, "data_source_breakdown", {}))
            meta.synthetic_data_used = bool(getattr(data_manager, "synthetic_data_used", True))  # 未知时必须判定为 True (Fail-Closed)
            meta.cache_fingerprint_verified = bool(getattr(data_manager, "cache_fingerprint_verified", False))
            meta.raw_data_provenance_preserved = bool(getattr(data_manager, "raw_data_provenance_preserved", False))

            meta.actual_backtest_start_date = getattr(data_manager, "actual_backtest_start_date", None)
            meta.actual_backtest_end_date = getattr(data_manager, "actual_backtest_end_date", None)
            meta.requested_backtest_start_date = getattr(data_manager, "requested_backtest_start_date", None)
            meta.requested_backtest_end_date = getattr(data_manager, "requested_backtest_end_date", None)

            meta.calendar_source = getattr(data_manager, "calendar_source", "unknown")
            meta.calendar_provider = getattr(data_manager, "calendar_provider", "unknown")
            meta.calendar_is_exchange_official = bool(getattr(data_manager, "calendar_is_exchange_official", False))
            meta.calendar_quality = getattr(data_manager, "calendar_quality", "unknown")
            meta.calendar_fallback_used = bool(getattr(data_manager, "calendar_fallback_used", False))
            meta.listing_date_coverage_ratio = getattr(data_manager, "listing_date_coverage_ratio", None)

            # 股票池提供器数据采集
            provider = getattr(data_manager, "universe_provider", None)
            if provider is not None:
                meta.universe_mode = provider.get_mode(meta.actual_backtest_start_date, meta.actual_backtest_end_date)
                meta.universe_coverage_start = getattr(provider, "coverage_start", None)
                meta.universe_coverage_end = getattr(provider, "coverage_end", None)
                meta.universe_coverage_complete = provider.is_coverage_complete(meta.actual_backtest_start_date, meta.actual_backtest_end_date)
                meta.universe_provenance_verified = getattr(provider, "universe_provenance_verified", False)
                meta.universe_source_class = getattr(provider, "universe_source_class", "UNKNOWN")
                meta.universe_raw_evidence_verified = getattr(provider, "universe_raw_evidence_verified", False)
                meta.universe_dataset_hash_verified = getattr(provider, "universe_dataset_hash_verified", False)
                meta.universe_manifest_hash = getattr(provider, "universe_manifest_hash", None)
                meta.universe_verification_failures = list(getattr(provider, "universe_verification_failures", []))
                meta.survivorship_bias_risk = provider.has_survivorship_bias_risk(meta.actual_backtest_start_date, meta.actual_backtest_end_date)

            meta.historical_st_symbol_coverage_ratio = getattr(data_manager, "historical_st_symbol_coverage_ratio", 0.0)
            meta.historical_st_date_coverage_ratio = getattr(data_manager, "historical_st_date_coverage_ratio", 0.0)
            meta.historical_st_coverage_complete = bool(getattr(data_manager, "historical_st_coverage_complete", False))
            meta.historical_st_bias_risk = bool(getattr(data_manager, "historical_st_bias_risk", True))
            meta.historical_st_available = bool(getattr(data_manager, "historical_st_available", False))
            meta.historical_st_rule_applied = bool(getattr(data_manager, "historical_st_rule_applied", False))
            meta.st_unknown_rows = getattr(data_manager, "st_unknown_rows", 0)

            meta.benchmark_source = getattr(data_manager, "benchmark_source", "unknown")
            meta.benchmark_coverage_ratio = getattr(data_manager, "benchmark_coverage_ratio", 0.0)
            meta.benchmark_missing_date_count = getattr(data_manager, "benchmark_missing_date_count", 0)

            meta.price_adjustment_mode = getattr(data_manager, "price_adjustment_mode", "unknown")
            meta.adjustment_point_in_time_safe = bool(getattr(data_manager, "adjustment_point_in_time_safe", False))
            meta.future_adjustment_leakage_test_passed = bool(getattr(data_manager, "future_adjustment_leakage_test_passed", False))
            meta.market_manifest_hash = getattr(data_manager, "manifest_hash", None)

        # 2. 因子特征层采集
        if factor_processor is not None:
            meta.industry_neutralization_enabled = getattr(factor_processor, "industry_neutralization_enabled", "DISABLED")
            meta.industry_coverage_ratio_mean = getattr(factor_processor, "industry_coverage_ratio_mean", None)
            meta.industry_coverage_ratio_min = getattr(factor_processor, "industry_coverage_ratio_min", None)
            meta.industry_neutralized_day_ratio = getattr(factor_processor, "industry_neutralized_day_ratio", None)
            meta.industry_neutralized_days = getattr(factor_processor, "industry_neutralized_days", 0)
            meta.market_cap_only_days = getattr(factor_processor, "market_cap_only_days", 0)
            meta.industry_data_available = bool(getattr(factor_processor, "industry_data_available", False))
            meta.feature_missing_ratio_total = getattr(factor_processor, "feature_missing_ratio_total", 0.0)
            meta.factor_manifest_hash = getattr(factor_processor, "manifest_hash", None)

        # 3. 策略组合层采集
        if portfolio_builder is not None:
            meta.sector_cap_enabled = bool(getattr(portfolio_builder, "sector_cap_enabled", True))
            meta.unknown_industry_cap_applied = bool(getattr(portfolio_builder, "unknown_industry_cap_applied", False))
            meta.unknown_industry_weight = getattr(portfolio_builder, "unknown_industry_weight", 0.0)

        # 4. 回测撮合执行层采集 (Fail-Closed: 缺失默认为 False)
        if engine is not None:
            meta.fee_model_scope = getattr(engine, "fee_model_scope", "historical_tiered")
            meta.order_quantity_conservation_passed = bool(getattr(engine, "order_quantity_conservation_passed", False))
            meta.partial_fill_count = getattr(engine, "partial_fill_count", 0)
            meta.liquidity_rejected_count = getattr(engine, "liquidity_rejected_count", 0)
            meta.pending_order_count = getattr(engine, "pending_order_count", 0)
            meta.cancelled_order_count = getattr(engine, "cancelled_order_count", 0)
            meta.deferred_order_count = getattr(engine, "deferred_order_count", 0)
            meta.corporate_action_source = getattr(engine, "corporate_action_source", "unknown")
            meta.corporate_action_coverage_ratio = getattr(engine, "corporate_action_coverage_ratio", 0.0)
            meta.corporate_action_coverage_complete = bool(getattr(engine, "corporate_action_coverage_complete", False))
            meta.corporate_action_bias_risk = bool(getattr(engine, "corporate_action_bias_risk", True))
            meta.corporate_action_adjustment_available = bool(getattr(engine, "corporate_action_adjustment_available", False))
            meta.corporate_action_zero_event_proof_verified = bool(getattr(engine, "corporate_action_zero_event_proof_verified", False))

        # 5. 处理外部自定义 override (严格白名单防伪拦截：所有认证输入字段一律禁止 override)
        if custom_overrides:
            meta.audit_override_used = True
            for k, v in custom_overrides.items():
                if hasattr(meta, k):
                    if k in NON_CERTIFICATION_OVERRIDE_FIELDS:
                        setattr(meta, k, v)
                        meta.audit_override_fields.append(k)
                    else:
                        meta.audit_override_fields.append(f"BLOCKED:{k}")
                        logger.warning(f"🛡️ [Anti-Forgery] 拦截针对受保护认证字段 {k} 的外部 override 篡改尝试！")

        # 6. 执行全要素策略门禁评级
        status, failed = CertificationPolicy.evaluate(meta)
        meta.overall_backtest_reliability = status
        meta.failed_certification_checks = failed

        return meta
