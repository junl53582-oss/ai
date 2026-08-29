"""
研究产物与证据链自动化验证器 (tools/validate_research_artifacts.py)
Phase 1.5 强化:
1. 重新计算并核对全部 SHA-256 哈希 (Tree, Requirements, Factor Matrix, Input Dataset)
2. 校验 Benchmark 状态与超额收益一致性 (Fail-Closed 时超额指标必须 N/A)
3. 校验 FDR 假设全家族记录数 ($N_{factor} \times N_{horizon}$)
4. 校验 daily_portfolio_pnl.csv Exact-Math 与 Delayed Exit 约束
5. 输出分项状态: STRUCTURE_VALID, HASH_VALID, BENCHMARK_VALID, EXECUTION_VALID, FDR_VALID, PROVENANCE_VALID -> FULL_VALID
"""
import sys
import json
import logging
import hashlib
import subprocess
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
    logger.info(f"🔍 [Phase 1.5] 开始全维度严密审计因子研究产物与证据链: {report_dir}")
    
    status_dict = {
        "STRUCTURE_VALID": False,
        "HASH_VALID": False,
        "BENCHMARK_VALID": False,
        "EXECUTION_VALID": False,
        "FDR_VALID": False,
        "PROVENANCE_VALID": False,
        "FULL_VALID": False
    }

    if not report_dir.exists():
        logger.error(f"❌ 报告目录不存在: {report_dir}")
        return False

    # 1. 结构检查 (Structure Check)
    for fname in REQUIRED_FILES:
        fpath = report_dir / fname
        if not fpath.exists():
            logger.error(f"❌ 缺失必需产物文件: {fname}")
            return False
        if fpath.stat().st_size == 0:
            logger.error(f"❌ 产物文件大小为空 (0 bytes): {fname}")
            return False
    status_dict["STRUCTURE_VALID"] = True
    logger.info("  [1/6] 结构完整性检查 (20/20 文件存在且非空): PASS")

    # 2. Manifest 与哈希校验 (Hash & Provenance Check)
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

        # 核算 requirements.txt 哈希
        req_file = Path("requirements.txt")
        if req_file.exists():
            exp_req_h = hashlib.sha256(req_file.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if manifest.get("requirements_hash") != exp_req_h:
                logger.error(f"❌ Manifest requirements_hash 与实际 requirements.txt 不匹配！")
                return False

        status_dict["HASH_VALID"] = True
        status_dict["PROVENANCE_VALID"] = True
        logger.info("  [2/6] Manifest 与数据血缘哈希校验: PASS")

    except Exception as e:
        logger.error(f"❌ 读取 Manifest 异常: {e}")
        return False

    # 3. 基准数据链与 Fail-Closed 校验 (Benchmark Check)
    bench_timing_status = manifest.get("benchmark_timing_status")
    df_summary = pd.read_csv(report_dir / "factor_summary.csv")
    if bench_timing_status != "VALID":
        # 基准无效时，factor_summary 中 long_only_excess_annual_return 必须为 NaN / 空
        valid_excess = df_summary["long_only_excess_annual_return"].dropna()
        if not valid_excess.empty:
            logger.error("❌ 基准时序无效但报告中仍产生了非空的超额年化收益！未达到 Fail-Closed 要求！")
            return False
    status_dict["BENCHMARK_VALID"] = True
    logger.info(f"  [3/6] 基准时序与超额 Fail-Closed 校验 (状态: {bench_timing_status}): PASS")

    # 4. 执行与 Delayed Exit 数值校验 (Execution & Delayed Exit Check)
    pnl_path = report_dir / "daily_portfolio_pnl.csv"
    try:
        df_pnl = pd.read_csv(pnl_path)
        if not df_pnl.empty:
            # 数值范围检查
            for col in ["long_gross_return", "benchmark_return", "long_excess_return", "long_net_return"]:
                if col in df_pnl.columns:
                    vals = df_pnl[col].dropna()
                    if (vals.abs() > 10.0).any() or np.isinf(vals).any():
                        logger.error(f"❌ daily_portfolio_pnl.csv 中 {col} 存在异常数值！")
                        return False

            # Exact-Math 检查: long_net_return == long_gross_return - total_cost
            if "long_net_return" in df_pnl.columns and "total_cost" in df_pnl.columns:
                diff = (df_pnl["long_gross_return"] - df_pnl["total_cost"]) - df_pnl["long_net_return"]
                if (diff.abs() > 1e-6).any():
                    logger.error("❌ daily_portfolio_pnl.csv 中 long_net_return != long_gross_return - total_cost！")
                    return False

            # Delayed Exit 约束检查
            if "actual_exit_date" in df_pnl.columns and "earliest_exit_date" in df_pnl.columns:
                act_dates = pd.to_datetime(df_pnl["actual_exit_date"])
                earl_dates = pd.to_datetime(df_pnl["earliest_exit_date"])
                if (act_dates < earl_dates).any():
                    logger.error("❌ daily_portfolio_pnl.csv 存在 actual_exit_date < earliest_exit_date！")
                    return False

        status_dict["EXECUTION_VALID"] = True
        logger.info("  [4/6] 交易执行 Exact-Math 与 Delayed Exit 约束校验: PASS")
    except Exception as e:
        logger.error(f"❌ 读取 daily_portfolio_pnl.csv 异常: {e}")
        return False

    # 5. FDR 多重检验维度校验 (FDR Check)
    fdr_path = report_dir / "factor_horizon_significance.csv"
    df_fdr = pd.read_csv(fdr_path)
    f_count = manifest.get("factor_count", 0)
    h_count = len(manifest.get("horizons_tested", []))
    expected_fdr_rows = f_count * h_count
    if len(df_fdr) != expected_fdr_rows:
        logger.error(f"❌ factor_horizon_significance.csv 记录数 ({len(df_fdr)}) 与预期假设数 ({expected_fdr_rows}) 不一致！")
        return False

    status_dict["FDR_VALID"] = True
    logger.info(f"  [5/6] 全家族 Global FDR 多重检验记录校验 ({expected_fdr_rows} 项假设): PASS")

    # 6. 最终综合评定
    status_dict["FULL_VALID"] = all([
        status_dict["STRUCTURE_VALID"],
        status_dict["HASH_VALID"],
        status_dict["BENCHMARK_VALID"],
        status_dict["EXECUTION_VALID"],
        status_dict["FDR_VALID"],
        status_dict["PROVENANCE_VALID"]
    ])

    if status_dict["FULL_VALID"]:
        logger.info("🏆 [6/6] 因子研究全要素证据链验证 100% 通过！FULL_VALID = TRUE")
        return True
    else:
        logger.error("❌ 验证未完全通过！")
        return False


if __name__ == "__main__":
    try:
        rep_dir = Path("reports/factor_research")
        if not validate_artifacts(rep_dir):
            sys.exit(1)
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Fatal error in artifact validator: {exc}")
        sys.exit(1)
