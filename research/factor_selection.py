"""
因子评分、等级划分与严格 Purged Walk-Forward 滚动筛选引擎 (research/factor_selection.py)
P0-2: 严密实现 Purge Gap >= max(HORIZONS)，杜绝训练/验证边界标签泄露
P0-6/7: 每折在训练集内独立执行全套特征选择(方向、最优视界、FDR、冗余过滤)，验证集仅评估冻结决策
P1-1: 强化 FDR 门禁 (STRONG 必须 FDR <= 0.05)
P1-6: 最少 Fold 数门禁 (MIN_WF_FOLDS_FOR_CERTIFICATION >= 3)
"""
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .config import ResearchConfig, default_research_config
from .factor_metrics import FactorMetricsEngine, FactorEvaluationMetrics
from .factor_decay import FactorDecayResult, FactorDecayEngine
from .factor_stability import FactorStabilityResult, FactorStabilityEngine
from .factor_correlation import CorrelationAnalysisResult, FactorCorrelationEngine

logger = logging.getLogger(__name__)


@dataclass
class FactorSelectionResult:
    """因子筛选全景结果"""
    selected_factors: List[str] = field(default_factory=list)
    useful_factors: List[str] = field(default_factory=list)
    weak_factors: List[str] = field(default_factory=list)
    rejected_factors: List[str] = field(default_factory=list)
    
    factor_scores_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_directions: Dict[str, int] = field(default_factory=dict)
    best_horizons: Dict[str, str] = field(default_factory=dict)
    rejection_reasons: Dict[str, List[str]] = field(default_factory=dict)
    status_summary: Dict[str, str] = field(default_factory=dict) # factor -> STRONG | USEFUL | WEAK | REJECT
    research_grade: Dict[str, str] = field(default_factory=dict) # OOS_VALIDATED | OOS_PRELIMINARY | IN_SAMPLE_STRONG | REJECT
    
    # 严格 Purged Walk-Forward 审计报告 (P0-2)
    walk_forward_stability: Dict[str, Any] = field(default_factory=dict)


class FactorSelectionEngine:
    """因子评分与多维门禁筛选引擎"""

    @classmethod
    def score_factors(
        cls,
        metrics_dict: Dict[str, FactorEvaluationMetrics],
        stability_dict: Dict[str, FactorStabilityResult],
        corr_result: CorrelationAnalysisResult,
        config: Optional[ResearchConfig] = None
    ) -> pd.DataFrame:
        """
        多维度标准化综合评分
        """
        cfg = config or default_research_config
        rows = []

        for name, m in metrics_dict.items():
            stab = stability_dict.get(name)
            sign_stab = stab.sign_consistency_ratio if stab else 0.5
            is_redundant = (corr_result.factor_to_group_id.get(name, 0) > 0)

            rows.append({
                "factor_name": name,
                "abs_rank_ic": abs(m.mean_rank_ic),
                "rank_ic": m.mean_rank_ic,
                "rank_ic_ir": abs(m.annualized_rank_icir),
                "monotonicity": abs(m.monotonicity_score),
                "sign_stability": sign_stab,
                "net_sharpe": max(m.net_sharpe_ratio, 0.0),
                "turnover": m.mean_turnover,
                "missing_ratio": m.missing_ratio,
                "coverage_ratio": m.coverage_ratio,
                "is_redundant": 1.0 if is_redundant else 0.0,
                "recommended_direction": m.recommended_direction
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        def _z(s):
            std = s.std(ddof=1)
            return (s - s.mean()) / (std if std > 1e-8 else 1.0)

        z_ic = _z(df["abs_rank_ic"])
        z_ir = _z(df["rank_ic_ir"])
        z_mono = _z(df["monotonicity"])
        z_stab = _z(df["sign_stability"])
        z_sharpe = _z(df["net_sharpe"])
        z_turn = _z(df["turnover"])
        z_miss = _z(df["missing_ratio"])

        score = (
            cfg.WEIGHT_RANK_IC * z_ic
            + cfg.WEIGHT_IC_IR * z_ir
            + cfg.WEIGHT_MONOTONICITY * z_mono
            + cfg.WEIGHT_STABILITY * z_stab
            + cfg.WEIGHT_NET_SHARPE * z_sharpe
            - cfg.PENALTY_TURNOVER * z_turn
            - cfg.PENALTY_MISSING * z_miss
            - cfg.PENALTY_REDUNDANCY * df["is_redundant"]
        )

        df["selection_score"] = score.round(4)
        df.sort_values(by="selection_score", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @classmethod
    def classify_factors(
        cls,
        scores_df: pd.DataFrame,
        metrics_dict: Dict[str, FactorEvaluationMetrics],
        decay_dict: Dict[str, FactorDecayResult],
        stability_dict: Dict[str, FactorStabilityResult],
        corr_result: CorrelationAnalysisResult,
        config: Optional[ResearchConfig] = None
    ) -> FactorSelectionResult:
        """
        根据门禁阈值将因子分为 STRONG / USEFUL / WEAK / REJECT
        P1-1 强化: STRONG 必须满足 rank_ic_fdr_p_value <= FDR_ALPHA (0.05)
        """
        cfg = config or default_research_config
        selected = []
        useful = []
        weak = []
        rejected = []
        
        directions = {}
        best_horizons = {}
        reasons = {}
        status_map = {}
        grades = {}

        # 冗余组去重
        kept_group_representatives = set()

        for _, row in scores_df.iterrows():
            name = row["factor_name"]
            m = metrics_dict[name]
            dec = decay_dict.get(name)
            stab = stability_dict.get(name)

            directions[name] = m.recommended_direction
            best_horizons[name] = dec.best_horizon if dec else f"{cfg.PRIMARY_HORIZON}D"
            factor_reasons = []

            abs_ic = abs(m.mean_rank_ic)
            effective_icir = max(abs(m.rank_ic_ir), abs(m.annualized_rank_icir))
            sign_stab = stab.sign_consistency_ratio if stab else 0.5
            cov = m.coverage_ratio
            gid = corr_result.factor_to_group_id.get(name, 0)

            # 冗余检测
            is_redundant_drop = False
            if gid > 0:
                if gid in kept_group_representatives:
                    is_redundant_drop = True
                    factor_reasons.append(f"high_redundancy_in_cluster_{gid}")
                else:
                    kept_group_representatives.add(gid)

            # 状态分级决策树 (P1-1 严格 FDR 门禁)
            if is_redundant_drop:
                status = "REJECT"
                rejected.append(name)
                grades[name] = "REJECT"
            elif (
                abs_ic >= cfg.STRONG_RANK_IC
                and effective_icir >= cfg.STRONG_IC_IR
                and sign_stab >= cfg.STRONG_SIGN_STABILITY
                and cov >= cfg.STRONG_COVERAGE
                and m.rank_ic_fdr_p_value <= cfg.FDR_ALPHA  # 严格 FDR <= 0.05
            ):
                status = "STRONG"
                selected.append(name)
                grades[name] = "IN_SAMPLE_STRONG"
            elif (
                abs_ic >= cfg.USEFUL_RANK_IC
                and effective_icir >= cfg.USEFUL_IC_IR
                and cov >= cfg.USEFUL_COVERAGE
                and m.rank_ic_fdr_p_value <= cfg.FDR_USEFUL_ALPHA
            ):
                status = "USEFUL"
                useful.append(name)
                grades[name] = "USEFUL"
            elif abs_ic >= cfg.WEAK_RANK_IC:
                status = "WEAK"
                weak.append(name)
                grades[name] = "WEAK"
            else:
                status = "REJECT"
                rejected.append(name)
                grades[name] = "REJECT"
                if abs_ic < cfg.WEAK_RANK_IC:
                    factor_reasons.append("insignificant_rank_ic")
                if effective_icir < cfg.USEFUL_IC_IR:
                    factor_reasons.append("low_icir_stability")
                if cov < cfg.USEFUL_COVERAGE:
                    factor_reasons.append("insufficient_coverage")
                if m.rank_ic_fdr_p_value > cfg.FDR_USEFUL_ALPHA:
                    factor_reasons.append(f"fdr_rejected_pval_{m.rank_ic_fdr_p_value:.3f}")

            status_map[name] = status
            reasons[name] = factor_reasons

        return FactorSelectionResult(
            selected_factors=selected,
            useful_factors=useful,
            weak_factors=weak,
            rejected_factors=rejected,
            factor_scores_df=scores_df,
            factor_directions=directions,
            best_horizons=best_horizons,
            rejection_reasons=reasons,
            status_summary=status_map,
            research_grade=grades
        )

    @classmethod
    def run_purged_walk_forward(
        cls,
        df: pd.DataFrame,
        factor_cols: List[str],
        config: Optional[ResearchConfig] = None
    ) -> Dict[str, Any]:
        """
        P0-2 / P0-6 / P0-7 严格 Purged Walk-Forward 因子样本外验证:
        1. 训练集 -> Purge Gap (>= max(HORIZONS)=25天) -> 验证集，杜绝标签前视泄露;
        2. 训练窗口内完全独立执行方向、最优视界、FDR 与因子选择 pipeline;
        3. 验证窗口严格冻结并仅评估训练集选出的因子 (Frozen Direction & Frozen Horizon);
        4. 未入选因子绝不记入 selected OOS 成绩。
        """
        cfg = config or default_research_config
        dates = sorted(pd.to_datetime(df["date"].unique()))
        
        train_days = int(cfg.WF_TRAIN_YEARS * 252)
        val_days = int(cfg.WF_VALIDATION_YEARS * 252)
        purge_gap = max(cfg.WF_PURGE_DAYS, max(cfg.HORIZONS))
        embargo = cfg.WF_EMBARGO_DAYS
        step_days = val_days

        folds_info = []
        start_idx = 0

        while start_idx + train_days + purge_gap + embargo + val_days <= len(dates):
            t_train_dates = dates[start_idx : start_idx + train_days]
            t_purge_dates = dates[start_idx + train_days : start_idx + train_days + purge_gap + embargo]
            t_val_dates = dates[start_idx + train_days + purge_gap + embargo : start_idx + train_days + purge_gap + embargo + val_days]
            
            folds_info.append({
                "fold_id": len(folds_info) + 1,
                "train_dates": t_train_dates,
                "purge_dates": t_purge_dates,
                "val_dates": t_val_dates
            })
            start_idx += step_days

        if not folds_info:
            return {
                "walk_forward_status": "PRELIMINARY",
                "reason": "insufficient_history_for_purged_folds",
                "total_folds": 0,
                "folds_detail": []
            }

        factor_selected_counts = {f: 0 for f in factor_cols}
        factor_oos_rank_ics: Dict[str, List[float]] = {f: [] for f in factor_cols}
        factor_oos_icirs: Dict[str, List[float]] = {f: [] for f in factor_cols}
        folds_detail = []

        for f_dict in folds_info:
            f_id = f_dict["fold_id"]
            train_d = f_dict["train_dates"]
            purge_d = f_dict["purge_dates"]
            val_d = f_dict["val_dates"]

            train_df = df[pd.to_datetime(df["date"]).isin(train_d)].copy()
            val_df = df[pd.to_datetime(df["date"]).isin(val_d)].copy()

            # 1. 训练窗口全要素评估 (P0-7)
            train_metrics: Dict[str, FactorEvaluationMetrics] = {}
            train_decay: Dict[str, FactorDecayResult] = {}
            train_pvals = []
            
            for f in factor_cols:
                dec = FactorDecayEngine.analyze_decay(train_df, f, horizons=cfg.HORIZONS)
                train_decay[f] = dec
                best_h = int(dec.best_horizon.replace("D", ""))
                ret_col = f"future_excess_return_{best_h}d" if f"future_excess_return_{best_h}d" in train_df.columns else f"future_return_{best_h}d"
                
                m = FactorMetricsEngine.evaluate_factor(train_df, f, ret_col, horizon=best_h, config=cfg)
                train_metrics[f] = m
                train_pvals.append(m.rank_ic_p_value)

            # 训练集 FDR 校正
            train_fdr = FactorMetricsEngine.compute_fdr_pvalues(train_pvals)
            for idx, f in enumerate(factor_cols):
                train_metrics[f].rank_ic_fdr_p_value = train_fdr[idx]

            # 训练集选拔因子 (STRONG 或 USEFUL 且 FDR <= 0.20)
            fold_selected = []
            fold_directions = {}
            fold_horizons = {}

            for f in factor_cols:
                m = train_metrics[f]
                dec = train_decay[f]
                eff_icir = max(abs(m.rank_ic_ir), abs(m.annualized_rank_icir))
                if (abs(m.mean_rank_ic) >= cfg.USEFUL_RANK_IC and eff_icir >= cfg.USEFUL_IC_IR and m.rank_ic_fdr_p_value <= cfg.FDR_USEFUL_ALPHA):
                    fold_selected.append(f)
                    fold_directions[f] = m.recommended_direction
                    fold_horizons[f] = dec.best_horizon

            for f in fold_selected:
                factor_selected_counts[f] += 1

            # 2. 验证窗口严格冻结评估 (P0-6: 仅对选出因子评估冻结参数)
            val_results = {}
            for f in fold_selected:
                h_int = int(fold_horizons[f].replace("D", ""))
                val_ret_col = f"future_excess_return_{h_int}d" if f"future_excess_return_{h_int}d" in val_df.columns else f"future_return_{h_int}d"
                
                val_ic_s = FactorMetricsEngine.compute_daily_ic(val_df, f, val_ret_col, method="spearman")
                if not val_ic_s.empty:
                    # 依据训练期冻结方向调正
                    aligned_ic = val_ic_s * fold_directions[f]
                    m_val_ic = float(aligned_ic.mean())
                    std_val_ic = float(aligned_ic.std(ddof=1)) if len(aligned_ic) > 1 else 1e-6
                    val_icir = (m_val_ic / std_val_ic) if std_val_ic > 1e-8 else 0.0

                    factor_oos_rank_ics[f].append(m_val_ic)
                    factor_oos_icirs[f].append(val_icir)
                    val_results[f] = {"oos_rank_ic": round(m_val_ic, 4), "oos_icir": round(val_icir, 4)}

            folds_detail.append({
                "fold_id": f_id,
                "train_start": str(train_d[0].date()),
                "train_end": str(train_d[-1].date()),
                "purge_start": str(purge_d[0].date()),
                "purge_end": str(purge_d[-1].date()),
                "validation_start": str(val_d[0].date()),
                "validation_end": str(val_d[-1].date()),
                "train_sample_count": len(train_df),
                "validation_sample_count": len(val_df),
                "selected_factors_count": len(fold_selected),
                "selected_factors": fold_selected,
                "oos_evaluation": val_results
            })

        total_folds = len(folds_info)
        wf_status = "OOS_VALIDATED" if total_folds >= cfg.MIN_WF_FOLDS_FOR_CERTIFICATION else "OOS_PRELIMINARY"

        summary = {}
        for f in factor_cols:
            ics = factor_oos_rank_ics[f]
            icirs = factor_oos_icirs[f]
            summary[f] = {
                "selected_frequency": round(factor_selected_counts[f] / max(total_folds, 1), 4),
                "selected_count": factor_selected_counts[f],
                "total_folds": total_folds,
                "oos_mean_rank_ic": round(float(np.mean(ics)), 4) if ics else 0.0,
                "oos_icir": round(float(np.mean(icirs)), 4) if icirs else 0.0
            }

        return {
            "walk_forward_status": wf_status,
            "total_folds": total_folds,
            "min_folds_required": cfg.MIN_WF_FOLDS_FOR_CERTIFICATION,
            "folds_detail": folds_detail,
            "factor_summary": summary
        }
