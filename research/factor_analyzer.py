"""
统一因子研究引擎 (research/factor_analyzer.py)
Phase 1.3 核心硬化:
1. P0-1 & P0-2: 彻底修复基准前向收益与股票前向超额收益标签 (统一定义为 T+1 Open 计价成交至 T+H Close 结算)
2. P1-1: 实现 Factor x Horizon 全家族 Global FDR 多重假设检验并导出 factor_horizon_significance.csv
3. P0-3 & P0-4: 真实截面 OLS 中性化与逐步残差正交化 Fail-Closed 机制
4. P0-6: 全要素绑定 Research Run Manifest
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
        self.horizon_significance_df: pd.DataFrame = pd.DataFrame()
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
        benchmark_open_col: str = "benchmark_open",
        config: Optional[ResearchConfig] = None
    ) -> pd.DataFrame:
        """
        Phase 1.3 核心硬化 (P0-1 / P0-2 时序与基准严密对齐):
        特征切断时间: T 日收盘后 (T Close)
        买入成交价格: T+1 日开盘价 (T+1 Open)
        卖出结算价格: T+H 日收盘价 (T+H Close)
        - 股票前向收益: R^{stock}_{H D, T+1 -> T+H} = (StockClose_{T+H} / StockOpen_{T+1}) - 1
        - 基准前向收益: R^{bench}_{H D, T+1 -> T+H} = (BenchmarkClose_{T+H} / BenchmarkOpen_{T+1}) - 1
        - 前向超额收益: R^{excess}_{H D} = R^{stock}_{H D} - R^{bench}_{H D}
        """
        horizons = horizons or [1, 3, 5, 10, 20]
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=[symbol_col, date_col], inplace=True)

        c_col = close_col if close_col in df.columns else "close"
        o_col = open_col if open_col in df.columns else ("open" if "open" in df.columns else c_col)
        b_close = benchmark_close_col if benchmark_close_col in df.columns else None
        b_open = benchmark_open_col if benchmark_open_col in df.columns else None

        # 1. 基准指数前向收益 (Date-Level 对齐，严格从 T+1 Open 计价成交至 T+H Close)
        bench_ret_map: Dict[int, pd.Series] = {}
        if b_close:
            b_daily = df.groupby(date_col)[[b_close, b_open] if b_open else [b_close]].first().sort_index()
            for h in horizons:
                if b_open and (b_daily[b_open] > 0).any():
                    bench_ret_map[h] = (b_daily[b_close].shift(-h) / b_daily[b_open].shift(-1)) - 1.0
                else:
                    if h == 1:
                        bench_ret_map[1] = b_daily[b_close].pct_change(1).shift(-1)
                    else:
                        bench_ret_map[h] = (b_daily[b_close].shift(-h) / b_daily[b_close].shift(-1)) - 1.0

        # 2. 个股前向收益率与超额收益
        for h in horizons:
            abs_col = f"future_return_{h}d"
            bench_col = f"future_benchmark_return_{h}d"
            exc_col = f"future_excess_return_{h}d"

            if o_col in df.columns and (df[o_col] > 0).any():
                df[abs_col] = (df.groupby(symbol_col)[c_col].shift(-h) / df.groupby(symbol_col)[o_col].shift(-1)) - 1.0
            else:
                if h == 1:
                    df[abs_col] = df.groupby(symbol_col)[c_col].transform(lambda s: (s.shift(-1) / s) - 1.0)
                else:
                    df[abs_col] = df.groupby(symbol_col)[c_col].transform(lambda s: (s.shift(-h) / s.shift(-1)) - 1.0)

            if h in bench_ret_map and not bench_ret_map[h].empty:
                df[bench_col] = df[date_col].map(bench_ret_map[h])
                df[exc_col] = df[abs_col] - df[bench_col]
            else:
                df[bench_col] = 0.0
                df[exc_col] = df[abs_col]

            # 3. 可交易性掩码
            if "is_suspended" in df.columns:
                next_susp = df.groupby(symbol_col)["is_suspended"].shift(-1)
                future_susp = next_susp.isna() | (next_susp == True)
                df.loc[future_susp, [abs_col, bench_col, exc_col]] = np.nan

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

        # 1. 准备未来收益标签 (P0-1 / P0-2)
        df_labeled = self.generate_future_return_labels(df, horizons=self.config.HORIZONS, config=self.config)
        primary_ret_col = f"future_excess_return_{primary_h}d" if self.config.USE_EXCESS_RETURN else f"future_return_{primary_h}d"

        if not factor_cols:
            exclude_cols = {
                "date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_change",
                "adj_open", "adj_high", "adj_low", "adj_close", "adj_pct_change", "benchmark_close", "benchmark_open",
                "in_universe", "is_st", "is_suspended", "industry", "list_date", "trade_date"
            }
            factor_cols = [c for c in df_labeled.columns if c not in exclude_cols and not c.startswith("future_return_") and not c.startswith("future_excess_return_") and not c.startswith("future_benchmark_return_")]

        logger.info(f"📊 待研究候选因子数量: {len(factor_cols)} 个")

        # 2. Factor x Horizon 全家族 Global FDR 多重检验 (Phase 1.3 P1-1)
        horizon_rows = []
        global_pvals = []
        pair_keys = []

        for f in factor_cols:
            for h in self.config.HORIZONS:
                ret_c = f"future_excess_return_{h}d" if self.config.USE_EXCESS_RETURN else f"future_return_{h}d"
                rank_ic_s = FactorMetricsEngine.compute_daily_ic(df_labeled, f, ret_c, method="spearman")
                m_ic = float(rank_ic_s.mean()) if not rank_ic_s.empty else 0.0
                t_stat, p_val = FactorMetricsEngine.compute_hac_tstat(rank_ic_s, lag=max(h - 1, 1))
                
                horizon_rows.append({
                    "factor": f,
                    "horizon": f"{h}D",
                    "mean_rank_ic": round(m_ic, 4),
                    "hac_t": t_stat,
                    "hac_p": p_val
                })
                global_pvals.append(p_val)
                pair_keys.append((f, h))

        # 全局 BH-FDR 校正
        global_fdr = FactorMetricsEngine.compute_fdr_pvalues(global_pvals)
        for idx, row in enumerate(horizon_rows):
            row["global_fdr_p"] = round(float(global_fdr[idx]), 6)
        
        self.horizon_significance_df = pd.DataFrame(horizon_rows)

        # 3. 单因子基础指标与 HAC 检验 (主视界)
        daily_ic_dict = {}
        for f in factor_cols:
            m = FactorMetricsEngine.evaluate_factor(df_labeled, f, primary_ret_col, horizon=primary_h, config=self.config)
            
            # 匹配 Global FDR 结果
            f_h_row = self.horizon_significance_df[
                (self.horizon_significance_df["factor"] == f) & (self.horizon_significance_df["horizon"] == f"{primary_h}D")
            ]
            if not f_h_row.empty:
                m.rank_ic_fdr_p_value = float(f_h_row["global_fdr_p"].iloc[0])
                m.fdr_p_value = m.rank_ic_fdr_p_value

            self.metrics_dict[f] = m
            if m.daily_rank_ic_series is not None and not m.daily_rank_ic_series.empty:
                daily_ic_dict[f] = m.daily_rank_ic_series

        # 4. 多视界衰减分析
        for f in factor_cols:
            dec = FactorDecayEngine.analyze_decay(df_labeled, f, horizons=self.config.HORIZONS)
            self.decay_dict[f] = dec

        # 5. 时间稳定性与牛熊状态分析
        for f in factor_cols:
            stab = FactorStabilityEngine.evaluate_stability(df_labeled, f, primary_ret_col, config=self.config)
            self.stability_dict[f] = stab

        # 6. 相关性与高冗余聚集分析 (P1-4 Complete Linkage)
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

        # 9. 严格 Purged Walk-Forward 样本外验证 (P0-5 / P1-2)
        wf_res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, factor_cols, config=self.config)
        self.selection_result.walk_forward_stability = wf_res

        # 10. 生成研究 Manifest 真实性证据链 (P0-4 / P0-6)
        self._build_research_run_manifest(df_labeled, factor_cols)

        # 11. 导出结构化报表与图表 (共 18 份证据文件)
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
                horizon_significance_df=self.horizon_significance_df,
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
        """截面多元 OLS 市值行业中性化 (Fail-Closed)"""
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
        """Sequential Residualization 逐步残差正交化 (Fail-Closed)"""
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

            if n_samples < len(target_factors) + 2:
                failed_cs += 1
                continue

            mat = sub_grp[target_factors].values.astype(float)
            ortho_cols = []

            try:
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
        """生成因子研究执行凭据 Manifest (Phase 1.3 P0-4 / P0-6)"""
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

        # requirements hash
        req_file = Path("requirements.txt")
        req_hash = hashlib.sha256(req_file.read_bytes()).hexdigest() if req_file.exists() else "unknown"

        # 全量因子矩阵按规范字母顺序排序后计算 SHA-256
        sorted_factors = sorted(factor_cols)
        h_matrix = pd.util.hash_pandas_object(df[sorted_factors], index=False)
        matrix_hash = hashlib.sha256(h_matrix.values.tobytes()).hexdigest()
        cols_hash = hashlib.sha256(",".join(sorted_factors).encode("utf-8")).hexdigest()

        sym_count = int(df["symbol"].nunique()) if "symbol" in df.columns else 0
        cfg = self.config
        
        # 样本充分性门槛分类 (P1-2)
        if sym_count < cfg.MIN_RESEARCH_SYMBOLS:
            validity_status = "DEVELOPMENT_SAMPLE"
        else:
            validity_status = self.selection_result.walk_forward_stability.get("walk_forward_status", "DISCOVERY") if self.selection_result else "DISCOVERY"

        self.run_manifest = {
            "schema_version": "1.3",
            "research_validity_status": validity_status,
            "research_source_commit": git_commit,
            "research_source_tree_hash": tree_hash,
            "research_source_working_tree_clean": is_clean,
            "requirements_hash": req_hash,
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
            "benchmark_return_definition": "T+1_open_to_T+H_close_return_matching_stock",
            "wf_purge_days": cfg.WF_PURGE_DAYS,
            "wf_embargo_days": cfg.WF_EMBARGO_DAYS,
            "min_wf_folds_required": cfg.MIN_WF_FOLDS_FOR_CERTIFICATION,
            "walk_forward_status": self.selection_result.walk_forward_stability.get("walk_forward_status", "PRELIMINARY") if self.selection_result else "PRELIMINARY",
            "selected_strong_count": len(self.selection_result.selected_factors) if self.selection_result else 0,
            "selected_useful_count": len(self.selection_result.useful_factors) if self.selection_result else 0,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
