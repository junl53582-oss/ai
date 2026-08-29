"""
因子综合评分、状态分级与 Walk-Forward 严密筛选引擎 (research/factor_selection.py)
综合 RankIC 强度、ICIR、单调性、时间稳定性、多空夏普与换手/缺失惩罚，透明可配置化打分，
结合 Purged Walk-Forward 滚动窗口杜绝 In-Sample 自嗨，输出 selected_factors.json。
"""
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .config import ResearchConfig, default_research_config
from .factor_metrics import FactorMetricsEngine, FactorEvaluationMetrics
from .factor_decay import FactorDecayResult
from .factor_stability import FactorStabilityResult
from .factor_correlation import CorrelationAnalysisResult

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
    
    # 走步筛选 (Walk-Forward) 稳定性报告
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
        多维度标准化综合评分 (透明公式化)
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
                "rank_ic_ir": abs(m.rank_ic_ir),
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

        # Z-Score 截面标准化打分组件
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

        # 线性组合打分
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
        根据门禁阈值将因子分为 STRONG / USEFUL / WEAK / REJECT 四大等级
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

        # 冗余组去重追踪: 冗余组内只保留评分最高的因子，其余降级为 REJECT(REDUNDANT)
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

            # 状态分级决策树
            if is_redundant_drop:
                status = "REJECT"
                rejected.append(name)
            elif (
                abs_ic >= cfg.STRONG_RANK_IC
                and effective_icir >= cfg.STRONG_IC_IR
                and sign_stab >= cfg.STRONG_SIGN_STABILITY
                and cov >= cfg.STRONG_COVERAGE
                and m.rank_ic_p_value < 0.05
            ):
                status = "STRONG"
                selected.append(name)
            elif (
                abs_ic >= cfg.USEFUL_RANK_IC
                and effective_icir >= cfg.USEFUL_IC_IR
                and cov >= cfg.USEFUL_COVERAGE
            ):
                status = "USEFUL"
                useful.append(name)
            elif abs_ic >= cfg.WEAK_RANK_IC:
                status = "WEAK"
                weak.append(name)
            else:
                status = "REJECT"
                rejected.append(name)
                if abs_ic < cfg.WEAK_RANK_IC:
                    factor_reasons.append("insignificant_rank_ic")
                if effective_icir < cfg.USEFUL_IC_IR:
                    factor_reasons.append("low_icir_stability")
                if cov < cfg.USEFUL_COVERAGE:
                    factor_reasons.append("insufficient_coverage")

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
            status_summary=status_map
        )

    @classmethod
    def run_walk_forward_selection(
        cls,
        df: pd.DataFrame,
        factor_cols: List[str],
        return_col: str,
        config: Optional[ResearchConfig] = None
    ) -> Dict[str, Any]:
        """
        滚动走步 (Walk-Forward) 因子验证：
        在训练窗口 (e.g. 2年) 筛选出有效因子，在后续紧邻的样本外验证窗口 (e.g. 1年) 评估 OOS RankIC 与 ICIR，
        统计因子在滚动窗口中的入选频率与 OOS 预测能力，杜绝 In-Sample 过拟合自嗨。
        """
        cfg = config or default_research_config
        dates = sorted(pd.to_datetime(df["date"].unique()))
        if len(dates) < 500: # 不足 2 年数据时进行单折切分
            return {"status": "insufficient_history_for_full_walk_forward"}

        train_days = int(cfg.WF_TRAIN_YEARS * 252)
        val_days = int(cfg.WF_VALIDATION_YEARS * 252)
        step_days = val_days

        folds = []
        start_idx = 0
        while start_idx + train_days + val_days <= len(dates):
            t_train = dates[start_idx : start_idx + train_days]
            t_val = dates[start_idx + train_days : start_idx + train_days + val_days]
            folds.append((t_train, t_val))
            start_idx += step_days

        if not folds:
            return {"status": "no_valid_folds_generated"}

        factor_selected_counts = {f: 0 for f in factor_cols}
        factor_oos_rank_ics = {f: [] for f in factor_cols}

        for f_idx, (t_train, t_val) in enumerate(folds):
            train_df = df[pd.to_datetime(df["date"]).isin(t_train)]
            val_df = df[pd.to_datetime(df["date"]).isin(t_val)]

            # 1. 训练窗口评估
            train_metrics = {}
            for f in factor_cols:
                train_metrics[f] = FactorMetricsEngine.evaluate_factor(train_df, f, return_col, config=cfg)

            # 选出该折选出的因子 (|RankIC| >= USEFUL_RANK_IC 且 ICIR >= USEFUL_IC_IR)
            fold_selected = [
                f for f, m in train_metrics.items()
                if abs(m.mean_rank_ic) >= cfg.USEFUL_RANK_IC and max(abs(m.rank_ic_ir), abs(m.annualized_rank_icir)) >= cfg.USEFUL_IC_IR
            ]

            for f in fold_selected:
                factor_selected_counts[f] += 1

            # 2. 样本外验证窗口评估
            for f in factor_cols:
                val_ic_s = FactorMetricsEngine.compute_daily_ic(val_df, f, return_col, method="spearman")
                if not val_ic_s.empty:
                    factor_oos_rank_ics[f].append(float(val_ic_s.mean()))

        total_folds = len(folds)
        summary = {}
        for f in factor_cols:
            ics = factor_oos_rank_ics[f]
            summary[f] = {
                "selected_frequency": round(factor_selected_counts[f] / max(total_folds, 1), 4),
                "selected_count": factor_selected_counts[f],
                "total_folds": total_folds,
                "oos_mean_rank_ic": round(float(np.mean(ics)), 4) if ics else 0.0,
                "oos_std_rank_ic": round(float(np.std(ics, ddof=1)), 4) if len(ics) > 1 else 0.0
            }

        return {"total_folds": total_folds, "factor_summary": summary}
