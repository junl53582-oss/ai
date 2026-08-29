"""
Alpha 因子处理与截面标准化/中性化引擎 (factors/processor.py)
严格遵循 Point-In-Time 原则：
1. 严禁全因子盲目 fillna(0.0)！Warmup 缺失值真实保留 NaN，仅布尔 Lag 标记填充 0.0
2. 截面标准化与行业+市值中性化严格仅基于 in_universe == True 样本计算均值、方差与 OLS 回归系数，非成分股绝不污染截面
3. 统计各因子缺失率 feature_missing_ratio_by_factor 与总缺失率 feature_missing_ratio_total
4. Manifest 缓存指纹管理，缓存读取直接还原真实统计指标
"""
import logging
import json
import hashlib
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import pandas as pd
import numpy as np

from config.settings import settings
from .alpha158 import Alpha158Subset
from .custom_ashare import AShareFactorCalculator
from .registry import FactorRegistry
from . import alternative_factors
from data.fundamentals import FUNDAMENTAL_FACTOR_NAMES

logger = logging.getLogger(__name__)


class MembershipIntegrityError(ValueError):
    """股票池成员完整性异常 (P0-5 Fail-Closed)"""
    pass


def _neutralize_one_day(date_key, group, factor_cols, industry_col, actual_mv_col):
    """
    单日截面中性化核心 (纯函数式，无共享状态)：
    串行与并行两条路径共用此实现，保证数值结果逐位一致。
    返回: (date_key, mode_str, daily_coverage, was_empty, processed_group)
    """
    group = group.copy()

    # 严格筛选当日 in_universe 样本用于拟合回归 (P0-5 Fail-Closed)
    if "in_universe" in group.columns:
        in_univ_mask = group["in_universe"].fillna(False).astype(bool).values
    else:
        in_univ_mask = np.ones(len(group), dtype=bool)

    # 当日截面如果 0 个成分股，记录 EMPTY_UNIVERSE，因子保持 NaN / 不参与训练 (P0-5)
    if not in_univ_mask.any():
        for col in factor_cols:
            if col in group.columns and col not in [actual_mv_col, "LOG_CIRC_MV", "log_circ_mv"]:
                group[col] = np.nan
        return date_key, "EMPTY_UNIVERSE", 0.0, True, group

    univ_indices = np.where(in_univ_mask)[0]
    n_univ_samples = len(univ_indices)

    # 1. 统计当日成分股行业有效覆盖率
    if industry_col and industry_col in group.columns:
        ind_series = group.loc[in_univ_mask, industry_col].fillna("UNKNOWN")
        valid_mask = ind_series.notna() & (ind_series != "UNKNOWN") & (ind_series.astype(str).str.strip() != "")
        daily_coverage = float(valid_mask.mean()) if len(ind_series) > 0 else 0.0
    else:
        daily_coverage = 0.0

    daily_coverage_rounded = round(daily_coverage, 4)

    # 2. 判定当日是否满足行业中性化条件
    can_do_industry = False
    if daily_coverage >= 0.50 and industry_col and industry_col in group.columns:
        ind_series_univ = group.loc[in_univ_mask, industry_col].fillna("UNKNOWN")
        unique_valid_ind = [ind for ind in ind_series_univ.unique() if ind not in ["UNKNOWN", "", None]]
        if len(unique_valid_ind) >= 2:
            can_do_industry = True

    # 3. 构造自变量矩阵 X (截距项 + 市值 + 行业Dummy)
    ones = np.ones((len(group), 1))
    X_components = [ones]

    if actual_mv_col and actual_mv_col in group.columns:
        mv_vals = group[actual_mv_col].values.astype(float)
        mv_mean = np.nanmean(mv_vals[in_univ_mask]) if np.isnan(mv_vals[in_univ_mask]).any() else np.mean(mv_vals[in_univ_mask])
        mv_filled = np.where(np.isnan(mv_vals), mv_mean if not np.isnan(mv_mean) else 0.0, mv_vals)
        X_components.append(mv_filled.reshape(-1, 1))

    if can_do_industry:
        raw_ind = group[industry_col].fillna("UNKNOWN").values
        valid_ind_names = sorted([ind for ind in np.unique(raw_ind[in_univ_mask]) if ind not in ["UNKNOWN", "", None]])
        if len(valid_ind_names) >= 2:
            # K-1 个行业哑变量
            dummies = []
            for ind_name in valid_ind_names[:-1]:
                dummy_col = (raw_ind == ind_name).astype(float).reshape(-1, 1)
                dummies.append(dummy_col)
            if dummies:
                X_components.append(np.hstack(dummies))
        else:
            can_do_industry = False

    X_full = np.hstack(X_components)
    n_features = X_full.shape[1]

    # 自由度检验
    if n_univ_samples <= n_features + 1 and can_do_industry:
        can_do_industry = False
        X_components = [ones]
        if actual_mv_col and actual_mv_col in group.columns:
            X_components.append(mv_filled.reshape(-1, 1))
        X_full = np.hstack(X_components)
        n_features = X_full.shape[1]

    mode_str = "INDUSTRY_AND_MCAP" if can_do_industry else "MCAP_ONLY"

    # 4. 对每个因子执行 OLS 回归 (仅用成分股拟合参数 beta，映射到全集残差)
    for col in factor_cols:
        if col in group.columns and col not in [actual_mv_col, "LOG_CIRC_MV", "log_circ_mv"]:
            y_series = group[col].values.astype(float)

            # 仅提取 in_universe 且非 NaN 的有效样本参与拟合
            valid_mask_full = ~np.isnan(y_series)
            valid_fit_mask = in_univ_mask & valid_mask_full
            fit_count = int(valid_fit_mask.sum())

            if fit_count > n_features + 1:
                X_fit = X_full[valid_fit_mask]
                y_fit = y_series[valid_fit_mask]

                try:
                    rank = np.linalg.matrix_rank(X_fit)
                    if rank < n_features:
                        beta, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=1e-5)
                    else:
                        beta, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)

                    # 残差映射到全部非空行
                    residuals = y_series[valid_mask_full] - X_full[valid_mask_full] @ beta
                    if np.std(residuals) > 1e-7:
                        group.loc[valid_mask_full, col] = residuals
                except Exception:
                    pass

    return date_key, mode_str, daily_coverage_rounded, False, group


def _neutralize_chunk(chunk_items, factor_cols, industry_col, actual_mv_col):
    """
    并行任务的分块单元: 处理一段连续交易日列表，按原时间顺序返回逐日结果元组列表。
    模块级定义以便 joblib/loky 进程池直接 pickle 引用。
    """
    results = []
    for dt, group in chunk_items:
        date_key = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
        results.append(_neutralize_one_day(date_key, group, factor_cols, industry_col, actual_mv_col))
    return results


class FactorProcessor:
    """因子计算、清洗、截面标准化与逐日行业中性化处理器 (Enterprise Hardened)"""

    def __init__(self, factor_dir: Optional[Path] = None):
        self.factor_dir = factor_dir or settings.FACTORS_DIR
        self.factor_dir.mkdir(parents=True, exist_ok=True)
        self.alpha_calc = Alpha158Subset()
        self.ashare_calc = AShareFactorCalculator()

        # 审计属性
        self.industry_neutralization_enabled: str = "DISABLED" # "FULL" | "PARTIAL" | "DISABLED"
        self.industry_coverage_ratio_mean: Optional[float] = None
        self.industry_coverage_ratio_min: Optional[float] = None
        self.industry_neutralized_days: int = 0
        self.market_cap_only_days: int = 0
        self.industry_neutralized_day_ratio: Optional[float] = None
        self.industry_coverage_by_date: Dict[str, float] = {}
        self.neutralization_mode_by_date: Dict[str, str] = {}
        self.industry_coverage_ratio: Optional[float] = None # 兼容旧接口

        # 缺失值与 Warmup 审计指标 (P0-1)
        self.feature_missing_ratio_by_factor: Dict[str, float] = {}
        self.feature_missing_ratio_total: float = 0.0
        self.warmup_rows_excluded: int = 0
        self.price_adjustment_mode: str = "unknown"
        self.adjustment_point_in_time_safe: bool = False
        self.future_adjustment_leakage_test_passed: bool = False
        self.empty_universe_day_count: int = 0
        self.unknown_membership_row_count: int = 0

        # Manifest 校验指纹与实体 (P0)
        self.manifest_hash: Optional[str] = None
        self.manifest_hash_verified: bool = False
        self.manifest_verification_result: Optional[Any] = None

    def verify_factor_manifest(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        expected_hash: Optional[str] = None,
        parent_runtime_config_hash: Optional[str] = None,
        parent_market_manifest_hash: Optional[str] = None,
        parent_universe_manifest_hash: Optional[str] = None
    ) -> Any:
        """从磁盘实际校验因子 Manifest 物理文件、严格 Schema 与父链"""
        from backtest.audit import ManifestVerifier, ManifestType
        target_path = Path(manifest_path) if manifest_path else self.factor_dir / "factor_matrix.manifest.json"
        parents = {}
        if parent_runtime_config_hash:
            parents["parent_runtime_config_hash"] = parent_runtime_config_hash
        if parent_market_manifest_hash:
            parents["parent_market_manifest_hash"] = parent_market_manifest_hash
        if parent_universe_manifest_hash:
            parents["parent_universe_manifest_hash"] = parent_universe_manifest_hash

        res = ManifestVerifier.verify_manifest_file(
            manifest_path=target_path,
            expected_hash=expected_hash,
            expected_parents=parents if parents else None,
            manifest_type=ManifestType.FACTOR
        )
        self.manifest_verification_result = res
        self.manifest_hash = res.actual_hash
        self.manifest_hash_verified = res.hash_verified and res.schema_verified
        return res

    def cross_sectional_standardize(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        clip_mad_sigma: float = 3.0,
        strict_pit: bool = False
    ) -> pd.DataFrame:
        """
        按日期执行截面标准化与 MAD 去极值 (向量化高性能实现)：
        严格仅使用 in_universe == True 样本计算截面中位数与 MAD 标准差 (P0-2, P0-5)！
        非成分股不参与截面统计量的计算，且保留其自身的标准化映射。
        缺失值 (NaN) 严格保留，不转为 0！
        """
        df = df.copy()
        if "in_universe" not in df.columns:
            if strict_pit:
                raise MembershipIntegrityError("缺失 in_universe 列，严禁在 PIT 模式下盲目假设全部为成分股！")
            in_univ_mask = pd.Series(True, index=df.index)
        else:
            if df["in_universe"].isna().any():
                self.unknown_membership_row_count += int(df["in_universe"].isna().sum())
                in_univ_mask = df["in_universe"].fillna(False).astype(bool)
            else:
                in_univ_mask = df["in_universe"].astype(bool)

        date_col = df["date"]

        for col in factor_cols:
            if col in df.columns:
                # 将 ±inf 归一为 NaN: inf 会污染截面中位数/MAD 与 Z-Score,
                # 并最终以 inf 形态进入 LightGBM 导致训练异常或静默产出垃圾模型。
                s_all = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
                s_univ = s_all.where(in_univ_mask)

                # 1. 截面中位数与 MAD 去极值
                med = s_univ.groupby(date_col).transform("median")
                abs_diff = (s_univ - med).abs()
                mad = abs_diff.groupby(date_col).transform("median")

                up_limit = med + clip_mad_sigma * 1.4826 * mad
                low_limit = med - clip_mad_sigma * 1.4826 * mad

                valid_mad = (mad > 1e-8) & mad.notna()
                clipped = np.where(valid_mad, np.clip(s_all, low_limit, up_limit), s_all)
                clipped_s = pd.Series(clipped, index=df.index)

                # 2. 截面 Z-Score 均值与方差
                clipped_univ = clipped_s.where(in_univ_mask)
                mean_val = clipped_univ.groupby(date_col).transform("mean")
                std_val = clipped_univ.groupby(date_col).transform(lambda s: s.std(ddof=1) if len(s.dropna()) > 1 else np.nan)

                valid_std = (std_val > 1e-8) & std_val.notna()
                zscored = np.where(valid_std, (clipped_s - mean_val) / std_val, clipped_s - mean_val)

                # 严格保留原有 NaN (不把 NaN 变 0)
                df[col] = np.where(s_all.isna(), np.nan, zscored)

        return df

    def neutralize_cross_section(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        industry_col: Optional[str] = "industry",
        market_cap_col: Optional[str] = "LOG_CIRC_MV"
    ) -> pd.DataFrame:
        """
        截面行业中性化与对数流通市值中性化 (动态自适应降级与严格 in_universe 隔离)：
        1. 仅在 in_universe == True 样本上构建回归设计矩阵 X 与因变量 y
        2. 若当日所有样本均为非成分股 (in_universe 全 False)，记录 EMPTY_UNIVERSE 且因子值保持 NaN，杜绝非成分股混入 OLS
        3. 缺失因子值 (NaN) 严格保留，不填充为 0 参与 OLS 拟合
        4. UNKNOWN 绝不作为真实行业 Dummy
        """
        df = df.copy()
        if "in_universe" not in df.columns:
            in_univ_global = pd.Series(True, index=df.index)
        else:
            in_univ_global = df["in_universe"].fillna(False).astype(bool)

        processed_dfs = []

        actual_mv_col = None
        if market_cap_col and market_cap_col in df.columns:
            actual_mv_col = market_cap_col
        elif "log_circ_mv" in df.columns:
            actual_mv_col = "log_circ_mv"
        elif "LOG_CIRC_MV" in df.columns:
            actual_mv_col = "LOG_CIRC_MV"

        date_groups = df.groupby("date")
        day_items = list(date_groups)
        n_days = len(day_items)
        neutral_start_ts = time.monotonic()
        logger.info(f"开始逐日截面中性化: 共 {n_days} 个交易日 × {len(factor_cols)} 个因子...")

        # ---------------- 并行度决策 ----------------
        # settings.NEUTRALIZATION_N_JOBS: 0=自动(CPU核数-1上限8, 仅当日数>=60) | 1=强制串行 | >=2=指定进程数
        n_jobs_setting = int(getattr(settings, "NEUTRALIZATION_N_JOBS", 0))
        if n_jobs_setting == 0:
            effective_n_jobs = max(1, min(8, (os.cpu_count() or 2) - 1)) if n_days >= 60 else 1
        elif n_jobs_setting >= 2:
            effective_n_jobs = n_jobs_setting
        else:
            effective_n_jobs = 1

        day_results = []
        if effective_n_jobs > 1:
            chunk_days = max(int(getattr(settings, "NEUTRALIZATION_CHUNK_DAYS", 25)), 1)
            chunks = [day_items[i:i + chunk_days] for i in range(0, len(day_items), chunk_days)]
            logger.info(f"并行模式启动: {effective_n_jobs} 进程 / {len(chunks)} 批 (每批约 {chunk_days} 个交易日)")
            try:
                from joblib import Parallel, delayed
                batch_results = Parallel(n_jobs=effective_n_jobs, backend="loky")(
                    delayed(_neutralize_chunk)(chunk, factor_cols, industry_col, actual_mv_col) for chunk in chunks
                )
                for batch in batch_results:
                    day_results.extend(batch)
            except ImportError:
                logger.warning("joblib 未安装，自动退回单线程串行中性化。")
                day_results = []
            except Exception as e:
                logger.warning(f"并行中性化执行异常 ({type(e).__name__}: {e})，安全退回单线程串行...")
                day_results = []

        if not day_results:
            # 单线程串行路径 (与并行共用同一单日实现 _neutralize_one_day，数值结果完全一致)
            serial_counter = 0
            for dt, group in day_items:
                serial_counter += 1
                if serial_counter % 200 == 0 or serial_counter == n_days:
                    elapsed = time.monotonic() - neutral_start_ts
                    eta_seconds = elapsed / serial_counter * (n_days - serial_counter)
                    logger.info(
                        f"截面中性化进度: {serial_counter}/{n_days} 日 ({serial_counter / max(n_days, 1) * 100:.1f}%) | "
                        f"已耗时 {elapsed:.0f}s | 预计剩余 {eta_seconds:.0f}s"
                    )
                date_key = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
                day_results.append(_neutralize_one_day(date_key, group, factor_cols, industry_col, actual_mv_col))

        # ---------------- 汇总逐日审计状态与结果 (保持原始时间顺序) ----------------
        processed_dfs = []
        empty_day_count = 0
        mode_by_date = {}
        cov_by_date = {}
        for date_key, mode_str, daily_coverage_rounded, was_empty, group in day_results:
            mode_by_date[date_key] = mode_str
            cov_by_date[date_key] = daily_coverage_rounded
            if was_empty:
                empty_day_count += 1
            processed_dfs.append(group)

        self.neutralization_mode_by_date.update(mode_by_date)
        self.industry_coverage_by_date.update(cov_by_date)
        self.empty_universe_day_count += empty_day_count

        neutral_elapsed = time.monotonic() - neutral_start_ts
        logger.info(f"逐日截面中性化完成: 共 {n_days} 个交易日, 总耗时 {neutral_elapsed:.1f}s ({neutral_elapsed / max(n_days, 1) * 1000:.0f}ms/日)")

        # 5. 汇总全局逐日中性化审计指标
        total_days = len(self.neutralization_mode_by_date)
        self.industry_neutralized_days = sum(1 for m in self.neutralization_mode_by_date.values() if m == "INDUSTRY_AND_MCAP")
        self.market_cap_only_days = sum(1 for m in self.neutralization_mode_by_date.values() if m == "MCAP_ONLY")
        self.industry_neutralized_day_ratio = round(self.industry_neutralized_days / max(total_days, 1), 4)

        cov_vals = list(self.industry_coverage_by_date.values())
        self.industry_coverage_ratio_mean = round(float(np.mean(cov_vals)), 4) if cov_vals else 0.0
        self.industry_coverage_ratio_min = round(float(np.min(cov_vals)), 4) if cov_vals else 0.0

        if self.industry_neutralized_day_ratio >= 0.999:
            self.industry_neutralization_enabled = "FULL"
        elif self.industry_neutralized_day_ratio > 0.0:
            self.industry_neutralization_enabled = "PARTIAL"
        else:
            self.industry_neutralization_enabled = "DISABLED"

        res_df = pd.concat(processed_dfs, ignore_index=True) if processed_dfs else df
        res_df.sort_values(by=["date", "symbol"], inplace=True)
        res_df.reset_index(drop=True, inplace=True)
        return res_df

    def build_and_save_factor_matrix(
        self,
        market_df: pd.DataFrame,
        force_update: bool = False,
        enable_neutralization: bool = True
    ) -> pd.DataFrame:
        """
        端到端构建因子矩阵：
        1. 严格按 symbol 分组前向填充，禁止全局 fillna(0.0)！Warmup 缺失值真实保留 NaN
        2. 仅布尔 Lag 因子显式填 0.0/False
        3. 统计各因子缺失率 feature_missing_ratio_by_factor
        4. 建立 Manifest 缓存指纹与完整审计元信息
        """
        factor_file = self.factor_dir / "factor_matrix.parquet"
        manifest_file = self.factor_dir / "factor_matrix.manifest.json"

    @staticmethod
    def _compute_streaming_content_hash(df: pd.DataFrame) -> str:
        """使用 pd.util.hash_pandas_object 计算完整数据流式 SHA256 (P0-14 杜绝采样碰撞)"""
        critical_cols = [c for c in ["date", "symbol", "open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close", "volume", "amount", "benchmark_close", "in_universe", "is_st", "is_suspended"] if c in df.columns]
        if not critical_cols or df.empty:
            return "empty"
        h_series = pd.util.hash_pandas_object(df[critical_cols], index=False)
        return hashlib.sha256(h_series.values.tobytes()).hexdigest()

    @staticmethod
    def _compute_in_universe_hash(df: pd.DataFrame) -> str:
        """流式计算 in_universe 掩码序列哈希 (P0-14)"""
        if "in_universe" not in df.columns:
            return "none"
        h_series = pd.util.hash_pandas_object(df[["in_universe"]], index=False)
        return hashlib.sha256(h_series.values.tobytes()).hexdigest()

    def build_and_save_factor_matrix(
        self,
        market_df: pd.DataFrame,
        force_update: bool = False,
        enable_neutralization: bool = True
    ) -> pd.DataFrame:
        """
        端到端构建因子矩阵：
        1. 严格按 symbol 分组前向填充，禁止全局 fillna(0.0)！Warmup 缺失值真实保留 NaN
        2. 仅布尔 Lag 因子显式填 0.0/False
        3. 统计各因子缺失率 feature_missing_ratio_by_factor
        4. 建立 Manifest 缓存指纹与完整审计元信息 (Streaming SHA256)
        """
        factor_file = self.factor_dir / "factor_matrix.parquet"
        manifest_file = self.factor_dir / "factor_matrix.manifest.json"

        # 1. 计算上游行情 Manifest 与内容哈希 (P0-7, P0-14 完整 64 hex SHA256 父链)
        upstream_manifest_file = settings.PARQUET_DIR / "market_daily.manifest.json"
        if upstream_manifest_file.exists():
            try:
                with open(upstream_manifest_file, "r", encoding="utf-8") as f:
                    up_text = f.read()
                input_market_manifest_hash = hashlib.sha256(up_text.encode("utf-8")).hexdigest()
            except Exception:
                input_market_manifest_hash = hashlib.sha256(b"unknown_market_manifest").hexdigest()
        else:
            input_market_manifest_hash = hashlib.sha256(b"none_market_manifest").hexdigest()

        market_content_hash = self._compute_streaming_content_hash(market_df)
        in_universe_mask_hash = self._compute_in_universe_hash(market_df)

        all_factor_cols = self.get_all_factor_cols()
        # 基本面因子若已并入 market_df 则纳入指纹 (其存在依赖 ENABLE_FUNDAMENTALS)
        fundamental_present = [
            c for c in FUNDAMENTAL_FACTOR_NAMES if c in market_df.columns
        ]
        factor_cols_hash = hashlib.sha256(",".join(sorted(all_factor_cols)).encode("utf-8")).hexdigest()[:16]
        fundamentals_hash = hashlib.sha256(",".join(sorted(fundamental_present)).encode("utf-8")).hexdigest()[:16]
        current_fingerprint = {
            "cache_schema_version": "3.0",
            "factor_code_version": "3.0",
            "factor_columns_hash": factor_cols_hash,
            "factor_count": len(all_factor_cols),
            "fundamentals_hash": fundamentals_hash,
            "neutralization_enabled": enable_neutralization,
            "input_market_manifest_hash": input_market_manifest_hash,
            "market_content_hash": market_content_hash,
            "in_universe_mask_hash": in_universe_mask_hash,
            "market_rows": len(market_df),
            "market_symbols_count": len(market_df["symbol"].unique()) if "symbol" in market_df.columns else 0
        }

        if factor_file.exists() and manifest_file.exists() and not force_update:
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                
                match = (
                    manifest.get("factor_columns_hash") == current_fingerprint["factor_columns_hash"]
                    and manifest.get("factor_count") == current_fingerprint["factor_count"]
                    and manifest.get("fundamentals_hash") == current_fingerprint["fundamentals_hash"]
                    and manifest.get("neutralization_enabled") == current_fingerprint["neutralization_enabled"]
                    and manifest.get("input_market_manifest_hash") == current_fingerprint["input_market_manifest_hash"]
                    and manifest.get("market_content_hash") == current_fingerprint["market_content_hash"]
                    and manifest.get("in_universe_mask_hash") == current_fingerprint["in_universe_mask_hash"]
                    and manifest.get("market_rows") == current_fingerprint["market_rows"]
                )
                if match:
                    logger.info(f"发现已有且链式指纹完全匹配的因子矩阵缓存: {factor_file}，正在加载...")
                    df_cached = pd.read_parquet(factor_file)
                    # 防御性完整性校验 1: 缓存必须包含全部声明的因子列, 否则视为残次缓存强制重建
                    missing_cols = [c for c in all_factor_cols if c not in df_cached.columns]
                    if missing_cols:
                        logger.warning(
                            f"缓存因子矩阵缺少 {len(missing_cols)} 列 ({missing_cols[:5]}...)，"
                            f"判定为残次缓存，触发重新计算..."
                        )
                        match = False
                    # 防御性完整性校验 2: 行数必须与上游行情一致。
                    elif len(df_cached) != len(market_df):
                        logger.warning(
                            f"缓存因子矩阵行数 {len(df_cached)} 与上游行情 {len(market_df)} 不一致，"
                            f"判定为被污染缓存，触发重新计算..."
                        )
                        match = False
                if match:
                    self._restore_audit_from_manifest(manifest, df_cached)
                    return df_cached
                else:
                    logger.info("因子矩阵缓存指纹或上游行情血缘发生变化，触发重新计算...")
            except Exception as e:
                logger.warning(f"读取因子 Manifest 异常: {e}，将重新构建...")

        logger.info("开始执行端到端因子构建流程...")
        df_alpha = self.alpha_calc.compute_all(market_df)
        df_full = self.ashare_calc.compute_all(df_alpha)
        # 高阶因子注册表 (另类/微观结构/遗传挖掘) 开关: 实测其弱特征会稀释信噪比，留出 A/B 开关
        if getattr(settings, "ENABLE_REGISTRY_FACTORS", True):
            df_full = FactorRegistry.compute_all_registered(df_full)

        fund_cols = [c for c in FUNDAMENTAL_FACTOR_NAMES if c in market_df.columns]
        missing_fund_cols = [c for c in fund_cols if c not in df_full.columns]
        if missing_fund_cols:
            df_full = df_full.merge(
                market_df[["symbol", "date"] + missing_fund_cols],
                on=["symbol", "date"], how="left"
            )
            logger.info(f"已并入基本面因子列: {missing_fund_cols}")
        elif fund_cols:
            logger.info(f"基本面因子列已由因子计算器带入: {fund_cols}")

        # 严格按 symbol 分组前向填充，严禁全因子盲目 fillna(0.0)！
        bool_lag_cols = {"IS_LIMIT_UP_LAG1", "IS_LIMIT_DOWN_LAG1"}
        valid_cols = [c for c in all_factor_cols if c in df_full.columns]
        if valid_cols:
            df_full[valid_cols] = df_full.groupby("symbol")[valid_cols].ffill()
            for col in bool_lag_cols:
                if col in df_full.columns:
                    df_full[col] = df_full[col].fillna(0.0)

            missing_series = df_full[valid_cols].isna().mean()
            self.feature_missing_ratio_by_factor = {k: round(float(v), 4) for k, v in missing_series.items()}
            self.feature_missing_ratio_total = round(float(missing_series.mean()), 4)

        df_standardized = self.cross_sectional_standardize(df_full, all_factor_cols)

        if enable_neutralization:
            df_standardized = self.neutralize_cross_section(
                df_standardized,
                factor_cols=[c for c in all_factor_cols if c not in ["LOG_CIRC_MV", "log_circ_mv"]],
                industry_col="industry" if "industry" in df_standardized.columns else None,
                market_cap_col="LOG_CIRC_MV" if "LOG_CIRC_MV" in df_standardized.columns else "log_circ_mv"
            )
            df_standardized = self.cross_sectional_standardize(df_standardized, all_factor_cols)

        logger.info(f"正在保存最终因子矩阵到 {factor_file} (总行数: {len(df_standardized)})...")
        df_standardized.to_parquet(factor_file, index=False, engine="pyarrow", compression="snappy")

        # 写入 Manifest (严格满足 ManifestType.FACTOR Schema 与父链)
        from backtest.audit import compute_canonical_runtime_config_hash
        factor_cols_present = [c for c in all_factor_cols if c in df_standardized.columns]
        h_series = pd.util.hash_pandas_object(df_standardized[factor_cols_present], index=False)
        dataset_sha256 = hashlib.sha256(h_series.values.tobytes()).hexdigest()
        parent_config_hash = compute_canonical_runtime_config_hash(settings)
        univ_manifest_file = settings.DATA_DIR / "universe" / "csi300" / "normalized" / "universe_pit_events.manifest.json"
        if univ_manifest_file.exists():
            try:
                with open(univ_manifest_file, "r", encoding="utf-8") as f:
                    u_text = f.read()
                parent_u_hash = hashlib.sha256(u_text.encode("utf-8")).hexdigest()
            except Exception:
                parent_u_hash = hashlib.sha256(b"unknown_universe_manifest").hexdigest()
        else:
            parent_u_hash = hashlib.sha256(b"none_universe_manifest").hexdigest()

        manifest_data = dict(current_fingerprint)
        manifest_data.update({
            "schema_version": "3.1",
            "dataset_name": "factor_matrix",
            "factor_columns": sorted(factor_cols_present),
            "dataset_sha256": dataset_sha256,
            "parent_runtime_config_hash": parent_config_hash,
            "parent_market_manifest_hash": input_market_manifest_hash,
            "parent_universe_manifest_hash": parent_u_hash,
            "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "industry_neutralization_enabled": self.industry_neutralization_enabled,
            "industry_coverage_ratio_mean": self.industry_coverage_ratio_mean,
            "industry_coverage_ratio_min": self.industry_coverage_ratio_min,
            "industry_neutralized_day_ratio": self.industry_neutralized_day_ratio,
            "industry_neutralized_days": self.industry_neutralized_days,
            "market_cap_only_days": self.market_cap_only_days,
            "feature_missing_ratio_total": self.feature_missing_ratio_total,
            "feature_missing_ratio_by_factor": self.feature_missing_ratio_by_factor,
            "price_adjustment_mode": self.price_adjustment_mode,
            "adjustment_point_in_time_safe": self.adjustment_point_in_time_safe,
            "future_adjustment_leakage_test_passed": self.future_adjustment_leakage_test_passed
        })
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        return df_standardized

    def _restore_audit_from_manifest(self, manifest: Dict[str, Any], df_cached: pd.DataFrame):
        """从 Manifest 还原真实审计指标 (P0-15 Fail-Closed 杜绝乐观默认)"""
        self.industry_neutralization_enabled = manifest.get("industry_neutralization_enabled", "DISABLED")
        self.industry_coverage_ratio_mean = manifest.get("industry_coverage_ratio_mean")
        self.industry_coverage_ratio_min = manifest.get("industry_coverage_ratio_min")
        self.industry_neutralized_day_ratio = manifest.get("industry_neutralized_day_ratio")
        self.industry_neutralized_days = manifest.get("industry_neutralized_days", 0)
        self.market_cap_only_days = manifest.get("market_cap_only_days", 0)
        self.feature_missing_ratio_total = manifest.get("feature_missing_ratio_total", 0.0)
        self.feature_missing_ratio_by_factor = manifest.get("feature_missing_ratio_by_factor", {})
        self.price_adjustment_mode = manifest.get("price_adjustment_mode", "unknown")
        self.adjustment_point_in_time_safe = bool(manifest.get("adjustment_point_in_time_safe", False))
        self.future_adjustment_leakage_test_passed = bool(manifest.get("future_adjustment_leakage_test_passed", False))

    def load_factor_matrix(self, expected_context: Optional[Dict[str, Any]] = None, strict: bool = False) -> pd.DataFrame:
        """加载因子矩阵缓存 (P0-8 严格指纹校验)"""
        factor_file = self.factor_dir / "factor_matrix.parquet"
        manifest_file = self.factor_dir / "factor_matrix.manifest.json"
        if not factor_file.exists():
            raise FileNotFoundError(f"未找到因子矩阵文件: {factor_file}，请先执行特征构建！")
        
        if not manifest_file.exists():
            if strict:
                raise ValueError("CacheIntegrityError: 因子矩阵 Manifest 文件缺失！")
        else:
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                
                if expected_context:
                    for k, v in expected_context.items():
                        if manifest.get(k) != v:
                            if strict:
                                raise ValueError(f"CacheIntegrityError: 因子缓存指纹项 {k} 不匹配 ({manifest.get(k)} != {v})")
                
                df = pd.read_parquet(factor_file)
                self._restore_audit_from_manifest(manifest, df)
                return df
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"读取因子 Manifest 异常: {e}")

        df = pd.read_parquet(factor_file)
        return df

    @classmethod
    def get_all_factor_cols(cls) -> List[str]:
        """获取全量因子特征名称集合 (Alpha158 + A股定制 + 另类高阶特征 + 基本面)"""
        cols = Alpha158Subset.get_factor_names() + AShareFactorCalculator.get_factor_names()
        if getattr(settings, "ENABLE_REGISTRY_FACTORS", True):
            for f in FactorRegistry.list_all_factors():
                if f not in cols:
                    cols.append(f)
        if getattr(settings, "ENABLE_FUNDAMENTALS", False):
            cols = cols + [f for f in FUNDAMENTAL_FACTOR_NAMES if f not in cols]
        return cols
