"""
因子研究报告生成器 (research/reports.py)
Phase 1.5: 严密输出 20 份全要素结构化证据文件、selected_factors.json、research_run_manifest.json
以及全景 FACTOR_RESEARCH_REPORT.md 与可视化图表。
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .factor_metrics import FactorEvaluationMetrics
from .factor_decay import FactorDecayResult
from .factor_stability import FactorStabilityResult
from .factor_correlation import CorrelationAnalysisResult
from .factor_selection import FactorSelectionResult

logger = logging.getLogger(__name__)


class FactorReportGenerator:
    """报告与图表导出器"""

    @classmethod
    def export_all_reports(
        cls,
        output_dir: Path,
        metrics_dict: Dict[str, FactorEvaluationMetrics],
        decay_dict: Dict[str, FactorDecayResult],
        stability_dict: Dict[str, FactorStabilityResult],
        corr_result: CorrelationAnalysisResult,
        selection_result: FactorSelectionResult,
        neutralization_comp: Dict[str, Any],
        orthogonalization_comp: Dict[str, Any],
        horizon_significance_df: Optional[pd.DataFrame] = None,
        wf_horizon_significance_df: Optional[pd.DataFrame] = None,
        run_manifest: Optional[Dict[str, Any]] = None,
        benchmark_evidence: Optional[Dict[str, Any]] = None
    ):
        """导出全套 20 份报表、凭据与 Markdown 总结"""
        output_dir.mkdir(parents=True, exist_ok=True)
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        bench_valid = (benchmark_evidence.get("benchmark_timing_status") == "VALID") if benchmark_evidence else False

        # 1. factor_summary.csv
        summary_rows = []
        for name, m in metrics_dict.items():
            dec = decay_dict.get(name)
            stab = stability_dict.get(name)
            score_row = selection_result.factor_scores_df[selection_result.factor_scores_df["factor_name"] == name]
            score_val = float(score_row["selection_score"].iloc[0]) if not score_row.empty else 0.0

            summary_rows.append({
                "factor_name": name,
                "status": selection_result.status_summary.get(name, "REJECT"),
                "research_grade": selection_result.research_grade.get(name, "REJECT"),
                "selection_score": score_val,
                "recommended_direction": m.recommended_direction,
                "best_horizon": dec.best_horizon if dec else "20D",
                "mean_rank_ic": m.mean_rank_ic,
                "std_rank_ic": m.std_rank_ic,
                "rank_ic_ir": m.rank_ic_ir,
                "annualized_rank_icir": m.annualized_rank_icir,
                "rank_ic_t_stat": m.rank_ic_t_stat,
                "rank_ic_hac_t_stat": m.rank_ic_hac_t_stat,
                "rank_ic_p_value": m.rank_ic_p_value,
                "rank_ic_hac_p_value": m.rank_ic_hac_p_value,
                "rank_ic_fdr_p_value": m.rank_ic_fdr_p_value,
                "positive_rank_ic_ratio": m.positive_rank_ic_ratio,
                "mean_ic": m.mean_ic,
                "std_ic": m.std_ic,
                "ic_ir": m.ic_ir,
                "annualized_icir": m.annualized_icir,
                "positive_ic_ratio": m.positive_ic_ratio,
                "naive_t_stat": m.naive_t_stat,
                "hac_t_stat": m.hac_t_stat,
                "p_value": m.p_value,
                "hac_p_value": m.hac_p_value,
                "fdr_p_value": m.fdr_p_value,
                "monotonicity": m.monotonicity_score,
                "long_only_cagr": m.long_only_cagr,
                "long_only_gross_return": m.long_only_gross_return,
                "long_only_net_return": m.long_only_net_return,
                "long_only_sharpe": m.long_only_sharpe,
                "long_only_max_drawdown": m.long_only_max_drawdown,
                "long_only_win_rate": m.long_only_win_rate,
                "long_only_excess_annual_return": m.long_only_excess_annual_return if bench_valid else None,
                "long_only_excess_sharpe": m.long_only_excess_sharpe if bench_valid else None,
                "long_turnover": m.long_turnover,
                "diagnostic_spread_annual": m.diagnostic_spread_annual,
                "diagnostic_spread_sharpe": m.diagnostic_spread_sharpe,
                "short_turnover": m.short_turnover,
                "mean_turnover": m.mean_turnover,
                "sign_stability": stab.sign_consistency_ratio if (stab and stab.sign_consistency_ratio is not None) else None,
                "annual_stability_status": stab.annual_stability_status if stab else "INSUFFICIENT_DATA",
                "bull_rank_ic": stab.bull_rank_ic if stab else 0.0,
                "bear_rank_ic": stab.bear_rank_ic if stab else 0.0,
                "sideways_rank_ic": stab.sideways_rank_ic if stab else 0.0,
                "coverage_ratio": m.coverage_ratio,
                "missing_ratio": m.missing_ratio,
                "redundancy_group": corr_result.factor_to_group_id.get(name, 0)
            })

        df_summary = pd.DataFrame(summary_rows)
        df_summary.sort_values(by="selection_score", ascending=False, inplace=True)
        df_summary.to_csv(output_dir / "factor_summary.csv", index=False, encoding="utf-8-sig")

        # 2. factor_ic.csv & factor_rankic.csv
        df_summary[["factor_name", "mean_ic", "std_ic", "ic_ir", "annualized_icir", "positive_ic_ratio", "naive_t_stat", "hac_t_stat", "p_value", "hac_p_value", "fdr_p_value"]].to_csv(
            output_dir / "factor_ic.csv", index=False, encoding="utf-8-sig"
        )
        df_summary[["factor_name", "mean_rank_ic", "std_rank_ic", "rank_ic_ir", "annualized_rank_icir", "positive_rank_ic_ratio", "rank_ic_t_stat", "rank_ic_hac_t_stat", "rank_ic_p_value", "rank_ic_hac_p_value", "rank_ic_fdr_p_value", "recommended_direction"]].to_csv(
            output_dir / "factor_rankic.csv", index=False, encoding="utf-8-sig"
        )

        # 3. factor_decay.csv
        decay_rows = []
        for name, dec in decay_dict.items():
            row = {"factor_name": name, "best_horizon": dec.best_horizon, "half_life_days": dec.half_life_days}
            row.update(dec.rank_ic_by_horizon)
            decay_rows.append(row)
        pd.DataFrame(decay_rows).to_csv(output_dir / "factor_decay.csv", index=False, encoding="utf-8-sig")

        # 4. factor_quantile_returns.csv
        q_rows = []
        for name, m in metrics_dict.items():
            row = {"factor_name": name, "monotonicity_score": m.monotonicity_score}
            row.update(m.quantile_returns_5q)
            q_rows.append(row)
        pd.DataFrame(q_rows).to_csv(output_dir / "factor_quantile_returns.csv", index=False, encoding="utf-8-sig")

        # 5. factor_cost_sensitivity.csv
        cost_rows = []
        for name, m in metrics_dict.items():
            row = {"factor_name": name, "long_turnover": m.long_turnover, "long_gross_return": m.long_only_gross_return, "long_sharpe": m.long_only_sharpe}
            row.update(m.cost_sensitivity)
            cost_rows.append(row)
        pd.DataFrame(cost_rows).to_csv(output_dir / "factor_cost_sensitivity.csv", index=False, encoding="utf-8-sig")

        # 6. factor_stability.csv
        stab_rows = []
        for name, stab in stability_dict.items():
            row = {
                "factor_name": name,
                "overall_rank_ic": stab.overall_mean_rank_ic,
                "sign_consistency": stab.sign_consistency_ratio,
                "annual_stability_status": stab.annual_stability_status,
                "bull_rank_ic": stab.bull_rank_ic,
                "bear_rank_ic": stab.bear_rank_ic,
                "sideways_rank_ic": stab.sideways_rank_ic
            }
            row.update({f"ic_{yr}": v for yr, v in stab.annual_rank_ic.items()})
            stab_rows.append(row)
        pd.DataFrame(stab_rows).to_csv(output_dir / "factor_stability.csv", index=False, encoding="utf-8-sig")

        # 7. factor_correlation.csv & factor_ic_correlation.csv
        corr_result.factor_value_corr_matrix.to_csv(output_dir / "factor_correlation.csv", encoding="utf-8-sig")
        corr_result.factor_ic_corr_matrix.to_csv(output_dir / "factor_ic_correlation.csv", encoding="utf-8-sig")

        # 8. factor_selection.csv & selected_factors.json
        selection_result.factor_scores_df.to_csv(output_dir / "factor_selection.csv", index=False, encoding="utf-8-sig")

        selected_json_data = {
            "selected_strong_factors": selection_result.selected_factors,
            "useful_factors": selection_result.useful_factors,
            "weak_factors": selection_result.weak_factors,
            "rejected_factors": selection_result.rejected_factors,
            "factor_directions": selection_result.factor_directions,
            "best_horizons": selection_result.best_horizons,
            "research_grade": selection_result.research_grade,
            "rejection_reasons": selection_result.rejection_reasons,
            "walk_forward_stability": selection_result.walk_forward_stability
        }
        with open(output_dir / "selected_factors.json", "w", encoding="utf-8") as f:
            json.dump(selected_json_data, f, indent=2, ensure_ascii=False)

        # 9. factor_horizon_significance.csv
        if horizon_significance_df is not None and not horizon_significance_df.empty:
            horizon_significance_df.to_csv(output_dir / "factor_horizon_significance.csv", index=False, encoding="utf-8-sig")

        # 10. walk_forward_factor_horizon_significance.csv
        if wf_horizon_significance_df is not None and not wf_horizon_significance_df.empty:
            wf_horizon_significance_df.to_csv(output_dir / "walk_forward_factor_horizon_significance.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["fold_id", "factor", "horizon", "train_mean_rank_ic", "train_hac_t", "train_hac_p", "train_global_fdr_p", "selected", "selected_horizon", "train_direction", "validation_raw_rank_ic", "validation_aligned_rank_ic"]).to_csv(output_dir / "walk_forward_factor_horizon_significance.csv", index=False, encoding="utf-8-sig")

        # 11. trade_rejection_evidence.csv
        all_rejections = []
        for m in metrics_dict.values():
            if m.trade_rejections:
                all_rejections.extend(m.trade_rejections)
        if all_rejections:
            pd.DataFrame(all_rejections).drop_duplicates().to_csv(output_dir / "trade_rejection_evidence.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["signal_date", "entry_date", "earliest_exit_date", "symbol", "side", "reject_stage", "reject_reason"]).to_csv(output_dir / "trade_rejection_evidence.csv", index=False, encoding="utf-8-sig")

        # 12. research_run_manifest.json
        if run_manifest:
            with open(output_dir / "research_run_manifest.json", "w", encoding="utf-8") as f:
                json.dump(run_manifest, f, indent=2, ensure_ascii=False)

        # 13. walk_forward_folds.csv
        wf_folds = selection_result.walk_forward_stability.get("folds_detail", [])
        if wf_folds:
            pd.DataFrame(wf_folds).to_csv(output_dir / "walk_forward_folds.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["fold_id", "train_start", "train_end", "purge_start", "purge_end", "validation_start", "validation_end", "selected_factors"]).to_csv(output_dir / "walk_forward_folds.csv", index=False, encoding="utf-8-sig")

        # 14. neutralization_evidence.csv
        if neutralization_comp:
            neu_rows = [{"factor_name": k, **v} for k, v in neutralization_comp.items()]
            pd.DataFrame(neu_rows).to_csv(output_dir / "neutralization_evidence.csv", index=False, encoding="utf-8-sig")

        # 15. orthogonalization_evidence.csv
        if orthogonalization_comp:
            ortho_rows = [{"factor_name": k, **v} for k, v in orthogonalization_comp.items()]
            pd.DataFrame(ortho_rows).to_csv(output_dir / "orthogonalization_evidence.csv", index=False, encoding="utf-8-sig")

        # 16. daily_portfolio_pnl.csv (包含 delayed exit 与实际执行状态)
        top_factor = df_summary["factor_name"].iloc[0] if not df_summary.empty else None
        if top_factor and top_factor in metrics_dict and metrics_dict[top_factor].daily_pnl_df is not None:
            metrics_dict[top_factor].daily_pnl_df.to_csv(output_dir / "daily_portfolio_pnl.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["signal_date", "entry_date", "entry_price_type", "earliest_exit_date", "actual_exit_date", "exit_delay_days", "exit_attempt_count", "exit_status", "exit_price_type", "requested_long_count", "executed_long_count", "rejected_long_count", "long_gross_return", "benchmark_return", "long_excess_return", "entry_commission", "exit_commission", "stamp_duty", "slippage", "total_cost", "long_net_return", "long_equity_curve", "factor_diagnostic_spread"]).to_csv(output_dir / "daily_portfolio_pnl.csv", index=False, encoding="utf-8-sig")

        # 17. 可视化图表
        cls._generate_charts(charts_dir, metrics_dict, decay_dict, corr_result, selection_result)

        # 18. 全景 Markdown 报告 FACTOR_RESEARCH_REPORT.md
        cls._generate_markdown_report(output_dir / "FACTOR_RESEARCH_REPORT.md", df_summary, selection_result, corr_result, neutralization_comp, orthogonalization_comp, run_manifest, benchmark_evidence)
        cls._generate_markdown_report(output_dir.parent.parent / "FACTOR_RESEARCH_REPORT.md", df_summary, selection_result, corr_result, neutralization_comp, orthogonalization_comp, run_manifest, benchmark_evidence)

    @classmethod
    def _generate_charts(
        cls,
        charts_dir: Path,
        metrics_dict: Dict[str, FactorEvaluationMetrics],
        decay_dict: Dict[str, FactorDecayResult],
        corr_result: CorrelationAnalysisResult,
        selection_result: FactorSelectionResult
    ):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top_factors = selection_result.selected_factors[:10] or list(metrics_dict.keys())[:10]
            if top_factors:
                plt.figure(figsize=(10, 5))
                names = top_factors
                ics = [metrics_dict[n].mean_rank_ic for n in names]
                colors = ["#2ecc71" if x >= 0 else "#e74c3c" for x in ics]
                plt.barh(names, ics, color=colors)
                plt.axvline(0, color="black", linestyle="--", alpha=0.5)
                plt.title("Top Candidate Factors Mean RankIC (HAC Robust)")
                plt.xlabel("Mean RankIC")
                plt.tight_layout()
                plt.savefig(charts_dir / "top_factors_rank_ic.png", dpi=150)
                plt.close()

            plt.figure(figsize=(10, 5))
            for n in top_factors[:5]:
                dec = decay_dict.get(n)
                if dec and dec.decay_curve:
                    x = [item["horizon_days"] for item in dec.decay_curve]
                    y = [abs(item["mean_rank_ic"]) for item in dec.decay_curve]
                    plt.plot(x, y, marker="o", label=f"{n} (Best: {dec.best_horizon})")
            plt.title("Factor Decay Curves Across Prediction Horizons")
            plt.xlabel("Horizon (Trading Days)")
            plt.ylabel("|Mean RankIC|")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(charts_dir / "factor_decay_curves.png", dpi=150)
            plt.close()

        except Exception as e:
            logger.warning(f"生成图表异常: {e}")

    @classmethod
    def _generate_markdown_report(
        cls,
        report_path: Path,
        df_summary: pd.DataFrame,
        selection_result: FactorSelectionResult,
        corr_result: CorrelationAnalysisResult,
        neutralization_comp: Dict[str, Any],
        orthogonalization_comp: Dict[str, Any],
        run_manifest: Optional[Dict[str, Any]] = None,
        benchmark_evidence: Optional[Dict[str, Any]] = None
    ):
        top10_df = df_summary.head(10)
        wf_info = selection_result.walk_forward_stability
        manifest = run_manifest or {}
        bench = benchmark_evidence or {}

        val_status = manifest.get("research_validity_status", "DEVELOPMENT_SAMPLE")
        bench_status = bench.get("benchmark_timing_status", "BENCHMARK_DATA_INVALID")
        n_strong = len(selection_result.selected_factors)
        n_useful = len(selection_result.useful_factors)

        ranking_title = "Top 10 核心有效因子排行榜" if (n_strong + n_useful > 0) else "Top 10 探索性候选因子表现 (Exploratory Candidates)"

        lines = [
            "# A股多因子研究与 Alpha 真实性验证报告 (Phase 1.5 Production Tradability & Benchmark Closure)",
            "",
            f"> **研究证据级别 (Validity Status)**: `{val_status}` (数据样本数: {manifest.get('symbol_count', 0)} 标的, 行数: {manifest.get('dataset_rows', 0)})",
            "> **结算规则 (Settlement Rule)**: `A_SHARE_T_PLUS_1_NO_SAME_DAY_SELL` (严禁日内开仓平仓回转)",
            "> **执行模型 (Execution Definition)**: `Signal at T Close -> Long Entry at T+1 Open -> Earliest Exit at T+2 Open (Delayed Exit on Lock/Suspension)`",
            f"> **基准时序状态 (Benchmark Timing)**: `{bench_status}` (开盘覆盖率: {bench.get('benchmark_open_coverage_ratio', 0.0)*100:.1f}%, 收盘覆盖率: {bench.get('benchmark_close_coverage_ratio', 0.0)*100:.1f}%)",
            "",
            "## 1. 核心架构与真实性闭环要点 (Phase 1.5 Integrity Highlights)",
            "- **P0-1/P0-2 基准缺失严格 Fail-Closed**: 基准开盘价缺失或不达标时，所有超额收益标签严格置为 NaN，超额指标显示 `N/A` (`BENCHMARK_TIMING_INVALID`)，绝不进行假想平价或 0 回退；",
            "- **P0-3 严密缓存失效与必需列校验**: Factor / Market 缓存架构升级至 v3.2，严格要求 `[date, symbol, adj_open, adj_close, benchmark_open, benchmark_close, in_universe]`，残缺缓存自动触发重构；",
            "- **P0-4 生产交易 Schema 完全对齐**: Execution 引擎原生接入 `is_limit_up_locked`, `is_limit_down_locked`, `limit_up_price`, `limit_down_price`，准确拦截涨停买入与 ST；",
            "- **P0-5 真实 Delayed Exit 展期机制**: 当 $T+2$ 遇到跌停或停牌无法卖出时，持仓顺延至 $T+3..T+k$ 成交，交易成本严格发生在实际成交日 `actual_exit_date`；",
            "- **P1-1 真实物理父链 Manifest**: 绝不使用伪造哈希，缺失父链如实标记为 `null` / `MISSING`；",
            "- **P1-3 几何复合增长率 (CAGR)**: 纯多头复合收益率严格采用 `(final_equity / initial_equity)**(252/N) - 1`，彻底消除算术均值年化误差。",
            "",
            "## 2. 研究概览与因子分级统计",
            f"- **候选因子总数**: {len(df_summary)} 个",
            f"- **STRONG 核心有效因子**: {n_strong} 个",
            f"- **USEFUL 次级可用因子**: {n_useful} 个",
            f"- **WEAK 弱预测因子**: {len(selection_result.weak_factors)} 个",
            f"- **REJECT 淘汰因子**: {len(selection_result.rejected_factors)} 个",
            f"- **高相关冗余聚类群组**: {len(corr_result.redundancy_groups)} 组",
            f"- **Walk-Forward 验证状态**: `{wf_info.get('walk_forward_status', 'PRELIMINARY')}` (总 Fold 数: {wf_info.get('total_folds', 0)})",
            "",
            f"## 3. {ranking_title}",
            "",
            "| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 纯多头 CAGR | 纯多头夏普 | 日均换手 | 纯多头超额年化 |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        rank = 1
        for _, r in top10_df.iterrows():
            dir_str = "正向 (+1)" if r["recommended_direction"] == 1 else "反向 (-1)"
            cagr_str = f"{r['long_only_cagr']*100:.1f}%" if (pd.notna(r['long_only_cagr']) and r['long_only_cagr'] > -0.99) else "-99.0%"
            
            if bench_status == "VALID" and pd.notna(r['long_only_excess_annual_return']):
                excess_str = f"{r['long_only_excess_annual_return']*100:.1f}%"
            else:
                excess_str = "N/A (BENCHMARK_INVALID)"

            lines.append(
                f"| {rank} | `{r['factor_name']}` | `{r['status']}` | `{r['research_grade']}` | {dir_str} | {r['best_horizon']} | {r['mean_rank_ic']:.4f} | {r['rank_ic_hac_t_stat']:.2f} | {r['rank_ic_fdr_p_value']:.4f} | {cagr_str} | {r['long_only_sharpe']:.2f} | {r['long_turnover']*100:.1f}% | {excess_str} |"
            )
            rank += 1

        lines.extend([
            "",
            "## 4. 真实截面中性化与正交化实证证据",
            "",
            "| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | 真实正交化 RankIC | 中性化状态 | 正交化状态 |",
            "| :--- | :---: | :---: | :---: | :--- | :--- |"
        ])

        for f in df_summary["factor_name"].head(8):
            n_item = neutralization_comp.get(f, {})
            o_item = orthogonalization_comp.get(f, {})
            n_ic = f"{n_item.get('neutralized_rank_ic'):.4f}" if n_item.get('neutralized_rank_ic') is not None else "None"
            o_ic = f"{o_item.get('orthogonalized_rank_ic'):.4f}" if o_item.get('orthogonalized_rank_ic') is not None else "None"
            lines.append(f"| `{f}` | {df_summary[df_summary['factor_name']==f]['mean_rank_ic'].iloc[0]:.4f} | {n_ic} | {o_ic} | `{n_item.get('status', 'UNAVAILABLE')}` | `{o_item.get('status', 'UNAVAILABLE')}` |")

        lines.extend([
            "",
            "## 5. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)",
            ""
        ])

        folds_list = wf_info.get("folds_detail", [])
        if folds_list:
            for f_item in folds_list:
                lines.extend([
                    f"### 📍 Fold {f_item['fold_id']}",
                    f"- **训练区间**: `{f_item['train_start']}` ~ `{f_item['train_end']}` ({f_item['train_rows']} 样本, {f_item['train_symbols']} 标的)",
                    f"- **Purge 隔离区间**: `{f_item['purge_start']}` ~ `{f_item['purge_end']}` (硬性隔离，无标签重叠)",
                    f"- **验证区间 (OOS)**: `{f_item['validation_start']}` ~ `{f_item['validation_end']}` ({f_item['validation_rows']} 样本, {f_item['validation_symbols']} 标的)",
                    f"- **训练集选出因子数**: `{f_item['selected_factor_count']}` 个",
                    f"- **OOS 验证表现**: {f_item['oos_evaluation']}"
                ])
        else:
            lines.append("- 样本历史长度在严格 25 日 Purge 隔离下未达到 3 折硬门禁，状态如实标记为 `PRELIMINARY`。")

        lines.extend([
            "",
            "---",
            "*本报告由 `research/factor_analyzer.py` 自动生成，20 份结构化证据已同步归档至 `reports/factor_research/`。*"
        ])

        sep = chr(10)
        report_path.write_text(sep.join(lines), encoding="utf-8")
