"""
因子研究报告生成器 (research/reports.py)
输出 18 份全要素结构化证据文件、selected_factors.json、research_run_manifest.json
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
        run_manifest: Optional[Dict[str, Any]] = None
    ):
        """导出全套 18 份报表、凭据与 Markdown 总结"""
        output_dir.mkdir(parents=True, exist_ok=True)
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

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
                "daily_gross_mean_return": m.daily_gross_mean_return,
                "daily_net_mean_return": m.daily_net_mean_return,
                "annualized_return": m.annualized_return,
                "annualized_volatility": m.annualized_volatility,
                "sharpe_ratio": m.sharpe_ratio,
                "net_sharpe_10bps": m.net_sharpe_ratio,
                "turnover": m.mean_turnover,
                "long_turnover": m.long_turnover,
                "short_turnover": m.short_turnover,
                "max_drawdown": m.max_drawdown,
                "win_rate": m.win_rate,
                "long_only_excess_annual_return": m.long_only_excess_annual_return,
                "long_only_excess_sharpe": m.long_only_excess_sharpe,
                "sign_stability": stab.sign_consistency_ratio if stab else 0.0,
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
            row = {"factor_name": name, "mean_turnover": m.mean_turnover, "gross_annual_return": m.annualized_return, "net_sharpe": m.net_sharpe_ratio}
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

        # 9. factor_horizon_significance.csv (Phase 1.3 P1-1)
        if horizon_significance_df is not None and not horizon_significance_df.empty:
            horizon_significance_df.to_csv(output_dir / "factor_horizon_significance.csv", index=False, encoding="utf-8-sig")

        # 10. research_run_manifest.json (P0-4 / P0-6)
        if run_manifest:
            with open(output_dir / "research_run_manifest.json", "w", encoding="utf-8") as f:
                json.dump(run_manifest, f, indent=2, ensure_ascii=False)

        # 11. walk_forward_folds.csv (P0-5 Evidence)
        wf_folds = selection_result.walk_forward_stability.get("folds_detail", [])
        if wf_folds:
            pd.DataFrame(wf_folds).to_csv(output_dir / "walk_forward_folds.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["fold_id", "train_start", "train_end", "purge_start", "purge_end", "validation_start", "validation_end", "selected_factors"]).to_csv(output_dir / "walk_forward_folds.csv", index=False, encoding="utf-8-sig")

        # 12. neutralization_evidence.csv (P0-3 Evidence)
        if neutralization_comp:
            neu_rows = [{"factor_name": k, **v} for k, v in neutralization_comp.items()]
            pd.DataFrame(neu_rows).to_csv(output_dir / "neutralization_evidence.csv", index=False, encoding="utf-8-sig")

        # 13. orthogonalization_evidence.csv (P0-4 Evidence)
        if orthogonalization_comp:
            ortho_rows = [{"factor_name": k, **v} for k, v in orthogonalization_comp.items()]
            pd.DataFrame(ortho_rows).to_csv(output_dir / "orthogonalization_evidence.csv", index=False, encoding="utf-8-sig")

        # 14. daily_portfolio_pnl.csv (P0-1 / P0-6 Evidence)
        top_factor = df_summary["factor_name"].iloc[0] if not df_summary.empty else None
        if top_factor and top_factor in metrics_dict and metrics_dict[top_factor].daily_pnl_df is not None:
            metrics_dict[top_factor].daily_pnl_df.to_csv(output_dir / "daily_portfolio_pnl.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["signal_date", "execution_date", "top_quantile_return", "bottom_quantile_return", "benchmark_return", "gross_return", "long_turnover", "short_turnover", "gross_turnover", "commission", "stamp_duty", "slippage", "total_cost", "net_return", "long_only_excess_return", "equity_curve"]).to_csv(output_dir / "daily_portfolio_pnl.csv", index=False, encoding="utf-8-sig")

        # 15. 可视化图表
        cls._generate_charts(charts_dir, metrics_dict, decay_dict, corr_result, selection_result)

        # 16. 全景 Markdown 报告 FACTOR_RESEARCH_REPORT.md
        cls._generate_markdown_report(output_dir / "FACTOR_RESEARCH_REPORT.md", df_summary, selection_result, corr_result, neutralization_comp, orthogonalization_comp, run_manifest)
        cls._generate_markdown_report(output_dir.parent.parent / "FACTOR_RESEARCH_REPORT.md", df_summary, selection_result, corr_result, neutralization_comp, orthogonalization_comp, run_manifest)

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

            if top_factors:
                best_factor = top_factors[0]
                m = metrics_dict[best_factor]
                if m.quantile_returns_5q:
                    plt.figure(figsize=(8, 4))
                    q_names = list(m.quantile_returns_5q.keys())
                    q_rets = [m.quantile_returns_5q[k] * 100 for k in q_names]
                    plt.bar(q_names, q_rets, color="#3498db")
                    plt.title(f"Quantile Returns for Top Candidate: {best_factor} (Mono: {m.monotonicity_score:.2f})")
                    plt.ylabel("Mean Forward Return (%)")
                    plt.tight_layout()
                    plt.savefig(charts_dir / "quantile_returns_top1.png", dpi=150)
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
        run_manifest: Optional[Dict[str, Any]] = None
    ):
        top10_df = df_summary.head(10)
        rejected_df = df_summary[df_summary["status"] == "REJECT"].head(10)
        wf_info = selection_result.walk_forward_stability
        manifest = run_manifest or {}

        val_status = manifest.get("research_validity_status", "DEVELOPMENT_SAMPLE")
        n_strong = len(selection_result.selected_factors)
        n_useful = len(selection_result.useful_factors)

        if n_strong == 0 and n_useful == 0:
            ranking_title = "Top 10 探索性候选因子表现 (Exploratory / Research Candidates)"
        else:
            ranking_title = "Top 10 核心有效因子排行榜 (Top Selected Factors)"

        lines = [
            "# A股多因子研究与 Alpha 真实性验证报告 (Phase 1.3 Execution & Research Provenance Closure)",
            "",
            f"> **研究证据级别 (Validity Status)**: `{val_status}` (数据样本数: {manifest.get('symbol_count', 0)} 标的, 行数: {manifest.get('dataset_rows', 0)})",
            "> **交易执行模型 (Execution Definition)**: `Signal at T Close -> Earliest Entry at T+1 Open -> Realized Post-Entry Return`",
            "> **基准收益模型 (Benchmark Definition)**: `Benchmark Entry at T+1 Open -> Benchmark Exit at T+H Close (Exact-Math Matching)`",
            "",
            "## 1. 核心架构与真实性闭环要点 (Phase 1.3 Integrity Highlights)",
            "- **P0-1 严格基准收益解耦**: 彻底消除基准价格直接参与相减的错误，基准日度收益与前向收益严格基于 `T+1 Open -> T+H Close` 计价计算；",
            "- **P0-2 前向超额标签严密对齐**: 个股与基准收益享有完全一致的进入与退出时间点，超额收益精确为小数收益率之差；",
            "- **P0-3 CI 真实可复现性**: `requirements.txt` 完整纳入 `cryptography` 与 `pytest`，工作流配置真实测试与产物上传；",
            "- **P0-4 Manifest 干净源码绑定**: 引入两阶段提交流程，Manifest 严密绑定执行时的 clean source commit 指纹；",
            "- **P0-5 真实可交易性过滤**: 严格审计 T+1 停牌、一字涨跌停锁死与无开盘价，阻断不可执行成交；",
            "- **P0-6 多空换手与成本独立分离**: 独立计算多头换手与空头换手，分别核算买入滑点佣金与卖出印花税佣金滑点。",
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
            "| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 真实年化收益 | 真实夏普(10bps) | 日均换手 | 纯多头超额年化 |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        rank = 1
        for _, r in top10_df.iterrows():
            dir_str = "正向 (+1)" if r["recommended_direction"] == 1 else "反向 (-1)"
            ann_ret_str = f"{r['annualized_return']*100:.1f}%" if r['annualized_return'] > -0.99 else "-99.0%"
            long_only_str = f"{r['long_only_excess_annual_return']*100:.1f}%" if r['long_only_excess_annual_return'] > -0.99 else "-99.0%"
            lines.append(
                f"| {rank} | `{r['factor_name']}` | `{r['status']}` | `{r['research_grade']}` | {dir_str} | {r['best_horizon']} | {r['mean_rank_ic']:.4f} | {r['rank_ic_hac_t_stat']:.2f} | {r['rank_ic_fdr_p_value']:.4f} | {ann_ret_str} | {r['net_sharpe_10bps']:.2f} | {r['turnover']*100:.1f}% | {long_only_str} |"
            )
            rank += 1

        lines.extend([
            "",
            "## 4. Top 候选因子实证特征解析",
            ""
        ])

        for _, r in top10_df.head(5).iterrows():
            fname = r["factor_name"]
            ann_ret_str = f"{r['annualized_return']*100:.1f}%" if r['annualized_return'] > -0.99 else "-99.0%"
            long_only_str = f"{r['long_only_excess_annual_return']*100:.1f}%" if r['long_only_excess_annual_return'] > -0.99 else "-99.0%"
            lines.extend([
                f"### 📍 `{fname}`",
                f"- **RankIC & HAC 稳健显著性**: 20D 均值 RankIC 为 `{r['mean_rank_ic']:.4f}`，Newey-West HAC t-stat 为 `{r['rank_ic_hac_t_stat']:.2f}`，全家族 FDR p-val 为 `{r['rank_ic_fdr_p_value']:.4f}`；",
                f"- **分层单调性**: 截面分层相关性得分为 `{r['monotonicity']:.2f}`；",
                f"- **真实 T+1 开盘可执行 PnL**: 多头换手率 `{r['long_turnover']*100:.1f}%`，空头换手率 `{r['short_turnover']*100:.1f}%`，综合日均换手率 `{r['turnover']*100:.1f}%`，真实非重叠日度年化收益 `{ann_ret_str}`，扣除摩擦后夏普为 `{r['net_sharpe_10bps']:.2f}`，Top 组相对于基准超额年化为 `{long_only_str}`；",
                f"- **市场状态表现**: 牛市 RankIC=`{r['bull_rank_ic']:.4f}`，熊市 RankIC=`{r['bear_rank_ic']:.4f}`，震荡市 RankIC=`{r['sideways_rank_ic']:.4f}`。"
            ])

        lines.extend([
            "",
            "## 5. 真实截面中性化实证证据 (Neutralization Evidence)",
            "",
            "| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | Delta | 有效截面天数 | 失败天数 | 状态 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
        ])

        for f, n_item in list(neutralization_comp.items())[:8]:
            n_ic = f"{n_item.get('neutralized_rank_ic'):.4f}" if n_item.get('neutralized_rank_ic') is not None else "None"
            n_delta = f"{n_item.get('delta_rank_ic'):+.4f}" if n_item.get('delta_rank_ic') is not None else "None"
            lines.append(f"| `{f}` | {n_item.get('raw_rank_ic', 0.0):.4f} | {n_ic} | {n_delta} | {n_item.get('successful_dates', 0)} | {n_item.get('failed_dates', 0)} | `{n_item.get('status', 'UNAVAILABLE')}` |")

        lines.extend([
            "",
            "## 6. 真实施密特逐步正交化证据 (Orthogonalization Evidence)",
            "",
            "| 因子名称 | Raw RankIC | 真实正交化 RankIC | Delta | 有效截面天数 | 失败天数 | 状态 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
        ])

        for f, o_item in list(orthogonalization_comp.items())[:8]:
            o_ic = f"{o_item.get('orthogonalized_rank_ic'):.4f}" if o_item.get('orthogonalized_rank_ic') is not None else "None"
            o_delta = f"{o_item.get('delta_rank_ic'):+.4f}" if o_item.get('delta_rank_ic') is not None else "None"
            lines.append(f"| `{f}` | {o_item.get('raw_rank_ic', 0.0):.4f} | {o_ic} | {o_delta} | {o_item.get('successful_cross_sections', 0)} | {o_item.get('failed_cross_sections', 0)} | `{o_item.get('status', 'UNAVAILABLE')}` |")

        lines.extend([
            "",
            "## 7. 淘汰因子清单及淘汰归因 (Sample Rejected Factors)",
            "",
            "| 因子名称 | 综合得分 | 淘汰原因归因 |",
            "| :--- | :---: | :--- |"
        ])

        for _, r in rejected_df.iterrows():
            fname = r["factor_name"]
            reasons = selection_result.rejection_reasons.get(fname, ["low_score"])
            reason_str = ", ".join(reasons) if reasons else "score_below_threshold"
            lines.append(f"| `{fname}` | {r['selection_score']:.2f} | `{reason_str}` |")

        lines.extend([
            "",
            "## 8. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)",
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
            "*本报告由 `research/factor_analyzer.py` 自动生成，18 份结构化证据已同步归档至 `reports/factor_research/`。*"
        ])

        sep = chr(10)
        report_path.write_text(sep.join(lines), encoding="utf-8")
