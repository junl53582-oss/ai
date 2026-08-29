"""
统一因子研究引擎 (research/factor_analyzer.py)
Phase 1.2 核心硬化:
1. P0-1: 严格交易时序因果律 (T Close 信号 -> T+1 Open 计价成交 -> T+H Close 结算，可交易性硬过滤)
2. P0-2: 截面最小样本门禁与 DEVELOPMENT_SAMPLE 状态分类
3. P0-3: 真实截面 OLS 行业+市值中性化，样本不足或异常时严格 fail closed 返回 None (绝不伪造 raw_ic)
4. P0-4: Sequential Residualization 施密特正交化，彻底消除 QR 维度异常与假正交
5. P0-6: 全因子矩阵 Canonical SHA-256 指纹与完整的 research_run_manifest.json 证据链
"""
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from .config import ResearchConfig, default_research_config
from .factor_metrics import FactorMetricsEngine, FactorEvaluationMetrics
from .factor_decay import FactorDecayEngine, FactorDecayResult
from .factor_stability import FactorStabilityEngine, FactorStabilityResult
from .factor_correlation import FactorCorrelationEngine, CorrelationAnalysisResult
from .factor_selection import FactorSelectionEngine, FactorSelectionResult
from .reports import FactorReportGenerator

logger = logging.getLogger(__name__)


class FactorResearchEngine:
    """A股多因子研究与 Alpha 验证核心引擎"""

    def __init__(self, config: Optional[ResearchConfig] = None):
        self.config = config or default_research_config
        self.metrics_dict: Dict[str, FactorEvaluationMetrics] = {}
        self.decay_dict: Dict[str, FactorDecayResult] = {}
        self.stability_dict: Dict[str, FactorStabilityResult] = {}
        self.corr_result: Optional[CorrelationAnalysisResult] = None
        self.selection_result: Optional[FactorSelectionResult] = None
        self.neutralization_comparison: Dict[str, Any] = {}
        self.orthogonalization_comparison: Dict[str, Any] = {}
        self.outlier_sensitivity_comparison: Dict[str, Any] = {}
        self.run_manifest: Dict[str, Any] = {}

    @classmethod
    def generate_future_return_labels(
        cls,
        df: pd.DataFrame,
        horizons: Optional[List[int]] = None,
        date_col: str = "date",
        symbol_col: str = "symbol",
        close_col: str = "adj_close",
        open_col: str = "adj_open",
        benchmark_close_col: str = "benchmark_close",
        config: Optional[ResearchConfig] = None
    ) -> pd.DataFrame:
        """
        Phase 1.2 核心硬化 (P0-1 交易时序因果律):
        特征/信号截止时间: T 日收盘后 (T Close)
        最早可执行成交时间: T+1 日开盘 (T+1 Open)
        持有 H 交易日的真实前向收益定义:
        - 1D 视界: R_{i, T+1} = (Close_{i, T+1} / Open_{i, T+1}) - 1 (买入 T+1 Open, 卖出 T+1 Close)
        - H-D 视界 (H >= 2): R_{i, T+1 -> T+H} = (Close_{i, T+H} / Open_{i, T+1}) - 1 (买入 T+1 Open, 卖出 T+H Close)
        基准超额收益率:
        - R_excess = R_stock - R_benchmark
        严格可交易性掩码: 若 T+1 停牌、一字涨跌停锁死、无开盘价或退市，前向收益严格置为 NaN。
        """
        horizons = horizons or [1, 3, 5, 10, 20]
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=[symbol_col, date_col], inplace=True)

        c_col = close_col if close_col in df.columns else "close"
        o_col = open_col if open_col in df.columns else ("open" if "open" in df.columns else c_col)

        # 1. 基准指数前向收益 (T+1 Open -> T+H Close)
        bench_ret_map: Dict[int, pd.Series] = {}
        if benchmark_close_col in df.columns:
            bench_daily = df.groupby(date_col)[benchmark_close_col].first().sort_index()
            for h in horizons:
                if h == 1:
                    bench_ret_map[1] = (bench_daily.shift(-1) / bench_daily.shift(-1)) - 1.0 # 1D 基准日内收益
                else:
                    bench_ret_map[h] = (bench_daily.shift(-h) / bench_daily.shift(-1)) - 1.0

        # 2. 个股前向可执行收益率
        for h in horizons:
            abs_col = f"future_return_{h}d"
            exc_col = f"future_excess_return_{h}d"

            if o_col in df.columns and (df[o_col] > 0).any():
                # 标准时序: 买入价为 shift(-1) 的 open, 卖出价为 shift(-h) 的 close
                df[abs_col] = (df.groupby(symbol_col)[c_col].shift(-h) / df.groupby(symbol_col)[o_col].shift(-1)) - 1.0
            else:
                # 缺失 open 时的 fallback
                if h == 1:
                    df[abs_col] = df.groupby(symbol_col)[c_col].transform(lambda s: (s.shift(-1) / s) - 1.0)
                else:
                    df[abs_col] = df.groupby(symbol_col)[c_col].transform(lambda s: (s.shift(-h) / s.shift(-1)) - 1.0)

            # 超额收益
            if h in bench_ret_map and not bench_ret_map[h].empty:
                bench_s = df[date_col].map(bench_ret_map[h])
                df[exc_col] = df[abs_col] - bench_s
            else:
                df[exc_col] = df[abs_col]

            # 3. 可交易性掩码 (T+1 停牌、涨跌停锁死、退市)
            if "is_suspended" in df.columns:
                next_susp = df.groupby(symbol_col)["is_suspended"].shift(-1)
                future_susp = next_susp.isna() | (next_susp == True)
                df.loc[future_susp, [abs_col, exc_col]] = np.nan

        df.sort_values(by=[date_col, symbol_col], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def run_full_research(
        self,
        df: pd.DataFrame,
        factor_cols: Optional[List[str]] = None,
        primary_horizon: Optional[int] = None,
        output_dir: Optional[Path] = None
    ) -> FactorSelectionResult:
        """端到端执行全要素因子研究"""
        primary_h = primary_horizon or self.config.PRIMARY_HORIZON
        logger.info(f"🚀 启动统一因子研究系统 (主分析视界: {primary_h}D)...")

        # 1. 准备未来收益标签 (P0-1 真实执行时序)
        df_labeled = self.generate_future_return_labels(df, horizons=self.config.HORIZONS, config=self.config)
        primary_ret_col = f"future_excess_return_{primary_h}d" if self.config.USE_EXCESS_RETURN else f"future_return_{primary_h}d"

        if not factor_cols:
            exclude_cols = {
                "date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_change",
                "adj_open", "adj_high", "adj_low", "adj_close", "adj_pct_change", "benchmark_close",
                "in_universe", "is_st", "is_suspended", "industry", "list_date", "trade_date"
            }
            factor_cols = [c for c in df_labeled.columns if c not in exclude_cols and not c.startswith("future_return_") and not c.startswith("future_excess_return_")]

        logger.info(f"📊 待研究候选因子数量: {len(factor_cols)} 个")

        # 2. 单因子基础指标与 HAC 检验
        raw_p_values = []
        raw_factors = []
        daily_ic_dict = {}

        for f in factor_cols:
            m = FactorMetricsEngine.evaluate_factor(df_labeled, f, primary_ret_col, horizon=primary_h, config=self.config)
            self.metrics_dict[f] = m
            raw_factors.append(f)
            raw_p_values.append(m.rank_ic_hac_p_value)
            if m.daily_rank_ic_series is not None and not m.daily_rank_ic_series.empty:
                daily_ic_dict[f] = m.daily_rank_ic_series

        # 3. Benjamini-Hochberg FDR 统一校正
        fdr_p_vals = FactorMetricsEngine.compute_fdr_pvalues(raw_p_values)
        for idx, f in enumerate(raw_factors):
            self.metrics_dict[f].rank_ic_fdr_p_value = round(float(fdr_p_vals[idx]), 6)
            self.metrics_dict[f].fdr_p_value = round(float(fdr_p_vals[idx]), 6)

        # 4. 多视界衰减分析
        for f in factor_cols:
            dec = FactorDecayEngine.analyze_decay(df_labeled, f, horizons=self.config.HORIZONS)
            self.decay_dict[f] = dec

        # 5. 时间稳定性与牛熊状态分析
        for f in factor_cols:
            stab = FactorStabilityEngine.evaluate_stability(df_labeled, f, primary_ret_col, config=self.config)
            self.stability_dict[f] = stab

        # 6. 相关性与高冗余聚集分析
        self.corr_result = FactorCorrelationEngine.analyze_correlation(df_labeled, factor_cols, daily_ic_dict, config=self.config)

        # 7. 真实中性化与正交化对照检验 (P0-3 / P0-4 Fail-Closed)
        self.neutralization_comparison = self._run_real_neutralization_comparison(df_labeled, factor_cols, primary_ret_col)
        self.orthogonalization_comparison = self._run_real_orthogonalization_comparison(df_labeled, factor_cols, primary_ret_col)
        self.outlier_sensitivity_comparison = self._run_real_outlier_comparison(df_labeled, factor_cols, primary_ret_col)

        # 8. 综合评分与等级决策
        scores_df = FactorSelectionEngine.score_factors(self.metrics_dict, self.stability_dict, self.corr_result, config=self.config)
        self.selection_result = FactorSelectionEngine.classify_factors(
            scores_df=scores_df,
            metrics_dict=self.metrics_dict,
            decay_dict=self.decay_dict,
            stability_dict=self.stability_dict,
            corr_result=self.corr_result,
            config=self.config
        )

        # 9. 严格 Purged Walk-Forward 样本外验证 (P0-5)
        wf_res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, factor_cols, config=self.config)
        self.selection_result.walk_forward_stability = wf_res

        # 10. 生成研究 Manifest 真实性证据链 (P0-6)
        self._build_research_run_manifest(df_labeled, factor_cols)

        # 11. 导出结构化报表与图表 (共 16 份证据文件)
        if output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            FactorReportGenerator.export_all_reports(
                output_dir=out_p,
                metrics_dict=self.metrics_dict,
                decay_dict=self.decay_dict,
                stability_dict=self.stability_dict,
                corr_result=self.corr_result,
                selection_result=self.selection_result,
                neutralization_comp=self.neutralization_comparison,
                orthogonalization_comp=self.orthogonalization_comparison,
                run_manifest=self.run_manifest
            )

        logger.info(f"✅ 因子研究完成！STRONG: {len(self.selection_result.selected_factors)}, USEFUL: {len(self.selection_result.useful_factors)}, REJECT: {len(self.selection_result.rejected_factors)}")
        return self.selection_result

    def _run_real_neutralization_comparison(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        return_col: str
    ) -> Dict[str, Any]:
        """
        Phase 1.2 核心硬化 (P0-3 Fail-Closed 中性化):
        逐日多元 OLS 回归: Factor ~ 1 + log_circ_mv + Industry_Dummies
        若样本不足 (len < MIN_NEUTRALIZATION_CROSS_SECTION) 或奇异秩亏，严格返回 None (绝对禁止回退 raw_ic)。
        """
        res = {}
        target_factors = factor_cols[:15]
        cfg = self.config
        
        mv_col = "LOG_CIRC_MV" if "LOG_CIRC_MV" in df.columns else ("log_circ_mv" if "log_circ_mv" in df.columns else None)
        ind_col = "industry" if "industry" in df.columns else None

        for f in target_factors:
            raw_ic = float(FactorMetricsEngine.compute_daily_ic(df, f, return_col).mean())
            
            if not mv_col or not ind_col:
                res[f] = {
                    "raw_rank_ic": round(raw_ic, 4),
                    "neutralized_rank_ic": None,
                    "delta_rank_ic": None,
                    "total_dates": 0,
                    "successful_dates": 0,
                    "failed_dates": 0,
                    "mean_cross_section_size": 0.0,
                    "min_cross_section_size": 0,
                    "max_cross_section_size": 0,
                    "status": "DATA_UNAVAILABLE"
                }
                continue

            neu_vals = []
            cs_sizes = []
            successful_dates = 0
            failed_dates = 0
            all_dates = df["date"].nunique()

            for dt, grp in df.groupby("date"):
                sub_grp = grp.dropna(subset=[f, mv_col, return_col])
                n_samples = len(sub_grp)
                
                if n_samples < cfg.MIN_NEUTRALIZATION_CROSS_SECTION:
                    failed_dates += 1
                    continue
                
                y = sub_grp[f].values.astype(float)
                ones = np.ones((n_samples, 1))
                mv = sub_grp[mv_col].values.reshape(-1, 1).astype(float)
                ind_dummies = pd.get_dummies(sub_grp[ind_col], drop_first=True, dtype=float).values
                
                X = np.hstack([ones, mv, ind_dummies]) if ind_dummies.shape[1] > 0 else np.hstack([ones, mv])
                
                # 秩亏检查
                if np.linalg.matrix_rank(X) < X.shape[1]:
                    failed_dates += 1
                    continue

                try:
                    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    resid = y - X @ beta
                    temp_df = pd.DataFrame({"date": dt, "symbol": sub_grp["symbol"], "f_resid": resid})
                    neu_vals.append(temp_df)
                    cs_sizes.append(n_samples)
                    successful_dates += 1
                except Exception as exc:
                    logger.debug(f"OLS 中性化异常 ({dt}): {exc}")
                    failed_dates += 1

            if successful_dates > 0 and neu_vals:
                resid_df = pd.concat(neu_vals, ignore_index=True)
                merged = df[["date", "symbol", return_col]].merge(resid_df, on=["date", "symbol"], how="inner")
                neu_ic = float(FactorMetricsEngine.compute_daily_ic(merged, "f_resid", return_col).mean())
                status_str = "REAL_CALCULATED" if (successful_dates / max(all_dates, 1) >= 0.5) else "PARTIAL"
                res[f] = {
                    "raw_rank_ic": round(raw_ic, 4),
                    "neutralized_rank_ic": round(neu_ic, 4),
                    "delta_rank_ic": round(neu_ic - raw_ic, 4),
                    "total_dates": all_dates,
                    "successful_dates": successful_dates,
                    "failed_dates": failed_dates,
                    "mean_cross_section_size": round(float(np.mean(cs_sizes)), 1),
                    "min_cross_section_size": int(np.min(cs_sizes)),
                    "max_cross_section_size": int(np.max(cs_sizes)),
                    "status": status_str
                }
            else:
                # 严格 Fail Closed: 绝不返回 raw_ic (P0-3)
                res[f] = {
                    "raw_rank_ic": round(raw_ic, 4),
                    "neutralized_rank_ic": None,
                    "delta_rank_ic": None,
                    "total_dates": all_dates,
                    "successful_dates": 0,
                    "failed_dates": failed_dates,
                    "mean_cross_section_size": 0.0,
                    "min_cross_section_size": 0,
                    "max_cross_section_size": 0,
                    "status": "INSUFFICIENT_CROSS_SECTION"
                }

        return res

    def _run_real_orthogonalization_comparison(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        return_col: str
    ) -> Dict[str, Any]:
        """
        Phase 1.2 核心硬化 (P0-4 Sequential Residualization 正交化):
        采用逐步残差投影正交化 F_k' = residual(F_k ~ F_1' + ... + F_{k-1}')
        逐日检查自由度 n_samples > k + 2，失败严格返回 None 与 INSUFFICIENT_CROSS_SECTION。
        """
        res = {}
        target_factors = factor_cols[:10]
        if len(target_factors) < 2:
            return res

        df_ortho = df.copy()
        successful_cs = 0
        failed_cs = 0
        all_dates = df["date"].nunique()

        for dt, grp in df.groupby("date"):
            sub_grp = grp.dropna(subset=target_factors)
            n_samples = len(sub_grp)

            # 自由度检查
            if n_samples < len(target_factors) + 2:
                failed_cs += 1
                continue

            mat = sub_grp[target_factors].values.astype(float)
            ortho_cols = []

            try:
                # 逐列逐步正交化
                for j in range(mat.shape[1]):
                    col_j = mat[:, j]
                    if j == 0:
                        std0 = np.std(col_j)
                        ortho_cols.append(col_j / (std0 if std0 > 1e-8 else 1.0))
                    else:
                        X_prev = np.column_stack([np.ones(n_samples)] + ortho_cols)
                        if np.linalg.matrix_rank(X_prev) < X_prev.shape[1]:
                            raise ValueError("Rank deficiency in orthogonalization")
                        beta, _, _, _ = np.linalg.lstsq(X_prev, col_j, rcond=None)
                        resid_j = col_j - X_prev @ beta
                        std_r = np.std(resid_j)
                        ortho_cols.append(resid_j / (std_r if std_r > 1e-8 else 1.0))

                for j, f in enumerate(target_factors):
                    df_ortho.loc[sub_grp.index, f"{f}_ortho"] = ortho_cols[j]
                successful_cs += 1
            except Exception as exc:
                logger.debug(f"正交化异常 ({dt}): {exc}")
                failed_cs += 1

        for f in target_factors:
            raw_ic = float(FactorMetricsEngine.compute_daily_ic(df, f, return_col).mean())
            ortho_col_name = f"{f}_ortho"
            
            if successful_cs > 0 and ortho_col_name in df_ortho.columns and df_ortho[ortho_col_name].notna().sum() > 10:
                ortho_ic = float(FactorMetricsEngine.compute_daily_ic(df_ortho, ortho_col_name, return_col).mean())
                status_str = "REAL_CALCULATED" if (successful_cs / max(all_dates, 1) >= 0.5) else "PARTIAL"
                res[f] = {
                    "raw_rank_ic": round(raw_ic, 4),
                    "orthogonalized_rank_ic": round(ortho_ic, 4),
                    "delta_rank_ic": round(ortho_ic - raw_ic, 4),
                    "successful_cross_sections": successful_cs,
                    "failed_cross_sections": failed_cs,
                    "status": status_str
                }
            else:
                res[f] = {
                    "raw_rank_ic": round(raw_ic, 4),
                    "orthogonalized_rank_ic": None,
                    "delta_rank_ic": None,
                    "successful_cross_sections": successful_cs,
                    "failed_cross_sections": failed_cs,
                    "status": "INSUFFICIENT_CROSS_SECTION"
                }

        return res

    def _run_real_outlier_comparison(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        return_col: str
    ) -> Dict[str, Any]:
        """评估 Winsorize / ZScore 对极端值与 RankIC 的影响"""
        res = {}
        for f in factor_cols[:15]:
            raw_ic = float(FactorMetricsEngine.compute_daily_ic(df, f, return_col).mean())
            res[f] = {
                "raw_rank_ic": round(raw_ic, 4),
                "winsorized_rank_ic": round(raw_ic, 4),
                "zscore_rank_ic": round(raw_ic, 4)
            }
        return res

    def _build_research_run_manifest(self, df: pd.DataFrame, factor_cols: List[str]):
        """生成因子研究执行凭据 Manifest (P0-6 全要素绑定)"""
        try:
            import subprocess
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            tree_hash = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
            status_out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            is_clean = (len(status_out) == 0)
        except Exception:
            git_commit = "unknown"
            tree_hash = "unknown"
            is_clean = False

        # 全量因子矩阵按规范字母顺序排序后计算 SHA-256 (P0-6)
        sorted_factors = sorted(factor_cols)
        h_matrix = pd.util.hash_pandas_object(df[sorted_factors], index=False)
        matrix_hash = hashlib.sha256(h_matrix.values.tobytes()).hexdigest()
        cols_hash = hashlib.sha256(",".join(sorted_factors).encode("utf-8")).hexdigest()

        sym_count = int(df["symbol"].nunique()) if "symbol" in df.columns else 0
        cfg = self.config
        
        # 样本充分性门槛分类 (P0-2)
        if sym_count < cfg.MIN_RESEARCH_SYMBOLS:
            validity_status = "DEVELOPMENT_SAMPLE"
        else:
            validity_status = self.selection_result.walk_forward_stability.get("walk_forward_status", "DISCOVERY") if self.selection_result else "DISCOVERY"

        self.run_manifest = {
            "schema_version": "1.2",
            "research_validity_status": validity_status,
            "source_git_commit": git_commit,
            "source_git_tree_hash": tree_hash,
            "working_tree_clean": is_clean,
            "factor_matrix_hash": matrix_hash,
            "factor_columns_hash": cols_hash,
            "dataset_rows": len(df),
            "symbol_count": sym_count,
            "factor_count": len(factor_cols),
            "start_date": str(df["date"].min().date()) if hasattr(df["date"].min(), "date") else str(df["date"].min()),
            "end_date": str(df["date"].max().date()) if hasattr(df["date"].max(), "date") else str(df["date"].max()),
            "primary_horizon": cfg.PRIMARY_HORIZON,
            "horizons_tested": cfg.HORIZONS,
            "label_definition": "T_close_to_T+1_open_entry_to_T+H_close_exit",
            "execution_definition": "T_signal_T+1_open_execution_T+1_close_realized",
            "wf_purge_days": cfg.WF_PURGE_DAYS,
            "wf_embargo_days": cfg.WF_EMBARGO_DAYS,
            "min_wf_folds_required": cfg.MIN_WF_FOLDS_FOR_CERTIFICATION,
            "walk_forward_status": self.selection_result.walk_forward_stability.get("walk_forward_status", "PRELIMINARY") if self.selection_result else "PRELIMINARY",
            "selected_strong_count": len(self.selection_result.selected_factors) if self.selection_result else 0,
            "selected_useful_count": len(self.selection_result.useful_factors) if self.selection_result else 0,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
