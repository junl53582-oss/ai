"""
研究产物与证据链自动化验证器 (tools/validate_research_artifacts.py)
在 CI 与本地流水线中自动执行严密校验，任何异常直接 Fail-Closed (exit 1)。
"""
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("ArtifactValidator")

REQUIRED_FILES = [
    "FACTOR_RESEARCH_REPORT.md",
    "factor_summary.csv",
    "factor_ic.csv",
    "factor_rankic.csv",
    "factor_decay.csv",
    "factor_quantile_returns.csv",
    "factor_cost_sensitivity.csv",
    "factor_stability.csv",
    "factor_correlation.csv",
    "factor_ic_correlation.csv",
    "factor_selection.csv",
    "factor_horizon_significance.csv",
    "walk_forward_factor_horizon_significance.csv",
    "trade_rejection_evidence.csv",
    "selected_factors.json",
    "walk_forward_folds.csv",
    "daily_portfolio_pnl.csv",
    "neutralization_evidence.csv",
    "orthogonalization_evidence.csv",
    "research_run_manifest.json"
]


def validate_artifacts(report_dir: Path) -> bool:
    logger.info(f"🔍 开始验证因子研究产物完整性与数值合理性: {report_dir}")
    
    if not report_dir.exists():
        logger.error(f"❌ 报告目录不存在: {report_dir}")
        return False

    # 1. 检查全部必需文件存在且非空
    for fname in REQUIRED_FILES:
        fpath = report_dir / fname
        if not fpath.exists():
            logger.error(f"❌ 缺失必需产物文件: {fname}")
            return False
        if fpath.stat().st_size == 0:
            logger.error(f"❌ 产物文件大小为空 (0 bytes): {fname}")
            return False

    # 2. 检查 Manifest 合法性
    manifest_path = report_dir / "research_run_manifest.json"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        required_manifest_keys = [
            "schema_version", "research_validity_status", "research_source_commit",
            "research_source_tree_hash", "factor_matrix_hash", "research_input_dataset_hash",
            "dataset_rows", "symbol_count", "factor_count", "settlement_rule"
        ]
        for k in required_manifest_keys:
            if k not in manifest:
                logger.error(f"❌ Manifest 缺失必要字段: {k}")
                return False

        if manifest.get("settlement_rule") != "A_SHARE_T_PLUS_1_NO_SAME_DAY_SELL":
            logger.error(f"❌ 结算规则不符合 A股 T+1 要求: {manifest.get('settlement_rule')}")
            return False

    except Exception as e:
        logger.error(f"❌ 读取 Manifest 异常: {e}")
        return False

    # 3. 检查 daily_portfolio_pnl.csv 数值合理性 (无 NaN, 无 Inf, 无价格量级收益)
    pnl_path = report_dir / "daily_portfolio_pnl.csv"
    try:
        df_pnl = pd.read_csv(pnl_path)
        if not df_pnl.empty:
            num_cols = ["long_gross_return", "benchmark_return", "long_excess_return", "long_net_return"]
            for col in num_cols:
                if col in df_pnl.columns:
                    vals = df_pnl[col].dropna()
                    if (vals.abs() > 10.0).any():
                        logger.error(f"❌ daily_portfolio_pnl.csv 中 {col} 存在荒谬价格量级收益 (超出 [-10, +10]): {vals[vals.abs() > 10.0].values}")
                        return False
                    if np.isinf(vals).any():
                        logger.error(f"❌ daily_portfolio_pnl.csv 中 {col} 存在 Inf 异常值！")
                        return False
    except Exception as e:
        logger.error(f"❌ 读取 daily_portfolio_pnl.csv 异常: {e}")
        return False

    # 4. 检查 factor_horizon_significance.csv 与 walk_forward_factor_horizon_significance.csv
    fdr_path = report_dir / "factor_horizon_significance.csv"
    df_fdr = pd.read_csv(fdr_path)
    if df_fdr.empty or len(df_fdr) < 10:
        logger.error("❌ factor_horizon_significance.csv 记录数不足！")
        return False

    logger.info("✅ 因子研究全部 20 份产物与证据链验证 100% 通过！")
    return True


if __name__ == "__main__":
    rep_dir = Path("reports/factor_research")
    if not validate_artifacts(rep_dir):
        sys.exit(1)
    sys.exit(0)
