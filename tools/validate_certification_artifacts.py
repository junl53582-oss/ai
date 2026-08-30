"""
CI Certification Artifact Validator (tools/validate_certification_artifacts.py)
只读验证 Phase 2.0.2 产物完整性、密码学哈希、门禁矩阵与换手率证据。
在 GitHub Actions 或本地快速门禁中执行。
"""
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from models.certification_logic import (
    build_research_protocol_config,
    build_model_full_config,
    compute_research_protocol_config_hash,
    compute_model_full_config_hash
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("artifact_validator")


def validate_artifacts() -> bool:
    reports_dir = settings.REPORTS_DIR / "model_research"
    if not reports_dir.exists():
        logger.error(f"Reports directory {reports_dir} does not exist.")
        return False

    # 1. 验证 latest.json 指针
    latest_file = reports_dir / "latest.json"
    if not latest_file.exists():
        logger.error("latest.json does not exist.")
        return False
    latest_data = json.loads(latest_file.read_text(encoding="utf-8"))
    for req_key in ["latest_run_id", "model_evidence_source_commit", "research_protocol_config_hash", "model_full_config_hash"]:
        if req_key not in latest_data:
            logger.error(f"latest.json missing required key: {req_key}")
            return False

    # 2. 验证配置哈希
    expected_protocol_hash = compute_research_protocol_config_hash()
    expected_full_hash = compute_model_full_config_hash()
    if latest_data.get("research_protocol_config_hash") != expected_protocol_hash:
        logger.error(f"research_protocol_config_hash mismatch: {latest_data.get('research_protocol_config_hash')} vs {expected_protocol_hash}")
        return False
    if latest_data.get("model_full_config_hash") != expected_full_hash:
        logger.error(f"model_full_config_hash mismatch: {latest_data.get('model_full_config_hash')} vs {expected_full_hash}")
        return False

    # 3. 验证 Trading Fold 与换手率证据
    fold_file = reports_dir / "trading_fold_stability_verified.csv"
    if not fold_file.exists():
        logger.error("trading_fold_stability_verified.csv does not exist.")
        return False
    df_folds = pd.read_csv(fold_file)
    if len(df_folds) != 20:
        logger.error(f"Expected 20 folds, got {len(df_folds)}")
        return False
    if set(df_folds["fold"]) != set(range(1, 21)):
        logger.error("Fold numbers are not 1..20")
        return False
    if not (pd.to_datetime(df_folds["test_start"]) <= pd.to_datetime(df_folds["test_end"])).all():
        logger.error("Fold test dates are invalid")
        return False
    
    # 换手率与费用一致性
    if not (df_folds["ranker_annualized_turnover"] > 0).all():
        logger.error("Ranker annualized turnover has non-positive values")
        return False
    if not (df_folds["baseline_annualized_turnover"] > 0).all():
        logger.error("Baseline annualized turnover has non-positive values")
        return False
    if not (df_folds["ranker_filled_trades"] > 0).all():
        logger.error("Ranker filled trades has non-positive values")
        return False
    if not (df_folds["ranker_total_costs"] > 0).all():
        logger.error("Ranker total costs has non-positive values")
        return False

    # 差值与胜负逻辑
    expected_deltas = df_folds["ranker_cost_adjusted_excess_return"] - df_folds["baseline_cost_adjusted_excess_return"]
    if not np.isclose(df_folds["delta_excess_return"], expected_deltas, atol=0.02).all():
        logger.error("Delta excess return math mismatch")
        return False
    if not (df_folds["ranker_win"] == (df_folds["ranker_cost_adjusted_excess_return"] > df_folds["baseline_cost_adjusted_excess_return"])).all():
        logger.error("Ranker win logic mismatch")
        return False

    # 4. 验证模型对比报告
    comp_file = reports_dir / "model_comparison_certified.csv"
    if not comp_file.exists():
        logger.error("model_comparison_certified.csv does not exist.")
        return False
    df_comp = pd.read_csv(comp_file, keep_default_na=False)
    if (df_comp["common_ranking_rows"].astype(int) != 221019).any():
        logger.error("Common ranking rows mismatch")
        return False
    if (df_comp["common_oos_dates"].astype(int) != 744).any():
        logger.error("Common OOS dates mismatch")
        return False
    
    # 非分类 AUC 必须为 N/A
    non_clf = df_comp[df_comp["task_type"] != "classification"]
    if not (non_clf["auc"] == "N/A").all():
        logger.error(f"Non-classification models must have AUC == 'N/A', got {non_clf['auc'].tolist()}")
        return False

    # 5. 验证门禁矩阵
    matrix_file = reports_dir / "certification_gate_matrix.json"
    if not matrix_file.exists():
        logger.error("certification_gate_matrix.json does not exist.")
        return False
    gate_data = json.loads(matrix_file.read_text(encoding="utf-8"))
    
    required_passes = [
        "SOURCE_PROVENANCE",
        "RESEARCH_PROTOCOL_CONFIG_HASH_VALIDITY",
        "MODEL_FULL_CONFIG_HASH_VALIDITY",
        "ARTIFACT_REUSE_COMPATIBILITY",
        "SEED_PROPAGATION",
        "PREDICTION_CHAMPION_SEED_ROBUSTNESS",
        "COMMON_OOS_POOL",
        "NW20_CERTIFICATION",
        "BOOTSTRAP_VALIDITY",
        "SELF_COMPARISON_GUARD",
        "REPORT_SEMANTICS",
        "TRADING_FOLD_EVIDENCE_VALIDITY",
        "TURNOVER_EVIDENCE_VALIDITY",
        "PYTEST"
    ]
    for g in required_passes:
        if gate_data.get(g) != "PASS":
            logger.error(f"Gate {g} is {gate_data.get(g)}, expected PASS")
            return False

    if gate_data.get("TRADING_CANDIDATE_SEED_ROBUSTNESS") != "NOT_CERTIFIED":
        logger.error(f"TRADING_CANDIDATE_SEED_ROBUSTNESS is {gate_data.get('TRADING_CANDIDATE_SEED_ROBUSTNESS')}, expected NOT_CERTIFIED")
        return False

    logger.info("==> [SUCCESS] All Phase 2.0.2 certification artifacts and evidence are valid and truthful!")
    return True


if __name__ == "__main__":
    if validate_artifacts():
        sys.exit(0)
    else:
        sys.exit(1)
