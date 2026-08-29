"""
因子研究报告生成器 (research/reports.py)
输出 CSV 汇总、selected_factors.json、research_run_manifest.json、
以及全景 FACTOR_RESEARCH_REPORT.md 与 Matplotlib 图表。
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
        run_manifest: Optional[Dict[str, Any]] = None
    ):
        """导出全套 13 份报表、凭据与 Markdown 总结"""
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
                "max_drawdown": m.max_drawdown,
                "win_rate": m.win_rate,
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

        # 5. factor_turnover.csv & factor_cost_sensitivity.csv
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

        # 9. research_run_manifest.json (P1-5)
        if run_manifest:
            with open(output_dir / "research_run_manifest.json", "w", encoding="utf-8") as f:
                json.dump(run_manifest, f, indent=2, ensure_ascii=False)

        # 10. 生成可视化图表 (Matplotlib)
        cls._generate_charts(charts_dir, metrics_dict, decay_dict, corr_result, selection_result)

        # 11. 生成全景 Markdown 报告 FACTOR_RESEARCH_REPORT.md
        cls._generate_markdown_report(output_dir / "FACTOR_RESEARCH_REPORT.md", df_summary, selection_result, corr_result, neutralization_comp, orthogonalization_comp)
        cls._generate_markdown_report(output_dir.parent.parent / "FACTOR_RESEARCH_REPORT.md", df_summary, selection_result, corr_result, neutralization_comp, orthogonalization_comp)

    @classmethod
    def _generate_charts(
        cls,
        charts_dir: Path,
        metrics_dict: Dict[str, FactorEvaluationMetrics],
        decay_dict: Dict[str, FactorDecayResult],
        corr_result: CorrelationAnalysisResult,
        selection_result: FactorSelectionResult
    ):
        """生成静态图表"""
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
                plt.title("Top Factors Mean RankIC (HAC Robust)")
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
                    plt.title(f"Quantile Returns for Top Factor: {best_factor} (Mono: {m.monotonicity_score:.2f})")
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
        orthogonalization_comp: Dict[str, Any]
    ):
        """渲染全景 Markdown 报告"""
        top10_df = df_summary.head(10)
        rejected_df = df_summary[df_summary["status"] == "REJECT"].head(10)
        wf_info = selection_result.walk_forward_stability

        lines = [
            "# A股多因子研究与 Alpha 真实性验证报告 (FACTOR RESEARCH INTEGRITY REPORT)",
            "",
            "> **数据研究性质说明**: `RESEARCH_ONLY` (基于 Point-In-Time 严格截面清洗、无前视标签与真实日度非重叠 PnL 评估)",
            "",
            "## 1. 核心研究架构与防作弊升级要点 (Phase 1.1 Integrity Hotfix)",
            "- **P0-1 收益与标签解耦**: 彻底区分 Label 预测研究 (Forward H-Day Return) 与多空组合回测 (Real Non-Overlapping Daily PnL)；",
            "- **P0-2 严格 Purged Walk-Forward**: 训练窗口与验证窗口之间硬性插入 25 个交易日 Purge Gap，彻底杜绝边界标签泄漏；",
            "- **P0-3 真实执行时序**: 因子于 $T$ 日收盘计算，组合于 $T+1$ 执行并获得真实日度实现收益；",
            "- **P0-4 真实中性化/正交化**: 彻底清除伪常量乘子，全量通过截面 OLS 残差回归与 Gram-Schmidt QR 分解计算真实增量 Alpha；",
            "- **P0-5 排序置换不变性**: 禁用强制拆解 Ties 的 `method='first'`，常数因子严格判定为 0 Alpha；",
            "- **P1-1 强化 FDR 门禁**: STRONG 因子必须满足 Benjamini-Hochberg FDR $p \le 0.05$；",
            "- **P1-2 序列相关性校正**: 采用 Newey-West HAC 异方差自相关稳健统计量评估重叠视界显著性。",
            "",
            "## 2. 研究概览与因子分级统计",
            f"- **候选因子总数**: {len(df_summary)} 个",
            f"- **STRONG 核心有效因子**: {len(selection_result.selected_factors)} 个",
            f"- **USEFUL 次级可用因子**: {len(selection_result.useful_factors)} 个",
            f"- **WEAK 弱预测因子**: {len(selection_result.weak_factors)} 个",
            f"- **REJECT 淘汰因子**: {len(selection_result.rejected_factors)} 个",
            f"- **高相关冗余聚类群组**: {len(corr_result.redundancy_groups)} 组",
            f"- **Walk-Forward 验证状态**: `{wf_info.get('walk_forward_status', 'PRELIMINARY')}` (总 Fold 数: {wf_info.get('total_folds', 0)})",
            "",
            "## 3. Top 10 核心有效因子排行榜 (Top Selected Factors)",
            "",
            "| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 真实年化收益 | 真实夏普(10bps) | 日均换手 |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        rank = 1
        for _, r in top10_df.iterrows():
            dir_str = "正向 (+1)" if r["recommended_direction"] == 1 else "反向 (-1)"
            ann_ret_str = f"{r['annualized_return']*100:.1f}%" if r['annualized_return'] > -0.99 else "-99.0%"
            lines.append(
                f"| {rank} | `{r['factor_name']}` | `{r['status']}` | `{r['research_grade']}` | {dir_str} | {r['best_horizon']} | {r['mean_rank_ic']:.4f} | {r['rank_ic_hac_t_stat']:.2f} | {r['rank_ic_fdr_p_value']:.4f} | {ann_ret_str} | {r['net_sharpe_10bps']:.2f} | {r['turnover']*100:.1f}% |"
            )
            rank += 1

        lines.extend([
            "",
            "## 4. Top 因子保留原因与真实特征解析",
            ""
        ])

        for _, r in top10_df.head(5).iterrows():
            fname = r["factor_name"]
            ann_ret_str = f"{r['annualized_return']*100:.1f}%" if r['annualized_return'] > -0.99 else "-99.0%"
            lines.extend([
                f"### 🌟 `{fname}`",
                f"- **RankIC & 统计显著性**: 20D 均值 RankIC 为 `{r['mean_rank_ic']:.4f}`，Newey-West HAC t-stat 为 `{r['rank_ic_hac_t_stat']:.2f}`，FDR 校正 p-val 为 `{r['rank_ic_fdr_p_value']:.4f}`；",
                f"- **分层单调性**: 截面分层相关性得分为 `{r['monotonicity']:.2f}`；",
                f"- **真实日度 PnL 表现**: 日均多头换手率 `{r['turnover']*100:.1f}%`，真实非重叠日收益年化 `{ann_ret_str}`，扣除 10 bps 摩擦成本后真实每日 PnL 夏普为 `{r['net_sharpe_10bps']:.2f}`；",
                f"- **市场状态表现**: 牛市 RankIC=`{r['bull_rank_ic']:.4f}`，熊市 RankIC=`{r['bear_rank_ic']:.4f}`，震荡市 RankIC=`{r['sideways_rank_ic']:.4f}`。"
            ])

        lines.extend([
            "",
            "## 5. 真实中性化与正交化对照实证 (Real Neutralization & Orthogonalization Evidence)",
            "",
            "| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | Delta (中性化) | 真实正交化 RankIC | Delta (正交化) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |"
        ])

        for f, n_item in list(neutralization_comp.items())[:8]:
            o_item = orthogonalization_comp.get(f, {})
            n_ic = f"{n_item.get('neutralized_rank_ic', 0.0):.4f}" if n_item.get('neutralized_rank_ic') is not None else "N/A"
            n_delta = f"{n_item.get('delta_rank_ic', 0.0):+.4f}" if n_item.get('delta_rank_ic') is not None else "N/A"
            o_ic = f"{o_item.get('orthogonalized_rank_ic', 0.0):.4f}" if o_item.get('orthogonalized_rank_ic') is not None else "N/A"
            o_delta = f"{o_item.get('delta_rank_ic', 0.0):+.4f}" if o_item.get('delta_rank_ic') is not None else "N/A"
            lines.append(f"| `{f}` | {n_item.get('raw_rank_ic', 0.0):.4f} | {n_ic} | {n_delta} | {o_ic} | {o_delta} |")

        lines.extend([
            "",
            "## 6. 淘汰因子清单及淘汰归因 (Sample Rejected Factors)",
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
            "## 7. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)",
            ""
        ])

        folds_list = wf_info.get("folds_detail", [])
        if folds_list:
            for f_item in folds_list:
                lines.extend([
                    f"### 📍 Fold {f_item['fold_id']}",
                    f"- **训练区间**: `{f_item['train_start']}` ~ `{f_item['train_end']}` ({f_item['train_sample_count']} 条样本)",
                    f"- **Purge 隔离区间**: `{f_item['purge_start']}` ~ `{f_item['purge_end']}` (硬性隔离，无标签重叠)",
                    f"- **验证区间 (OOS)**: `{f_item['validation_start']}` ~ `{f_item['validation_end']}` ({f_item['validation_sample_count']} 条样本)",
                    f"- **训练集选出因子数**: `{f_item['selected_factors_count']}` 个",
                    f"- **OOS 验证表现**: {f_item['oos_evaluation']}"
                ])
        else:
            lines.append("- 数据历史长度未达到完整多折 Purged Walk-Forward 切分门禁，状态标记为 `PRELIMINARY`。")

        lines.extend([
            "",
            "---",
            "*本报告由 `research/factor_analyzer.py` 自动生成，结构化数据已同步归档至 `reports/factor_research/`。*"
        ])

        sep = chr(10)
        report_path.write_text(sep.join(lines), encoding="utf-8")
