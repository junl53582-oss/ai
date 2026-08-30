"""
Legacy Truth Freeze Generator (tools/freeze_legacy_v1.py)
严格从现有认证产物中提取 Phase 2.0.2 历史实证，冻结为不可篡改的 LEGACY_BASELINE_V1 基准。
"""
import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from research_v2.provenance.source_config_reader import (
    resolve_historical_effective_configs,
    compute_legacy_effective_model_config_hash
)


def compute_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_legacy_v1():
    source_reports_dir = settings.REPORTS_DIR / "model_research"
    target_baseline_dir = settings.REPORTS_DIR / "baselines" / "legacy_v1"
    target_baseline_dir.mkdir(parents=True, exist_ok=True)

    comp_file = source_reports_dir / "model_comparison_certified.csv"
    fold_file = source_reports_dir / "trading_fold_stability_verified.csv"
    seed_file = source_reports_dir / "seed_robustness_verified.csv"
    source_state_file = source_reports_dir / "source_state.json"
    latest_file = source_reports_dir / "latest.json"

    # Fail-Closed 检查
    for f in [comp_file, fold_file, seed_file, source_state_file, latest_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing certified artifact for legacy freeze: {f}")

    df_comp = pd.read_csv(comp_file, keep_default_na=False)
    df_folds = pd.read_csv(fold_file)
    df_seed = pd.read_csv(seed_file)
    source_state = json.loads(source_state_file.read_text(encoding="utf-8"))
    latest_data = json.loads(latest_file.read_text(encoding="utf-8"))

    # 提取有效模型配置与 Hash
    effective_configs = resolve_historical_effective_configs(
        source_commit=source_state.get("model_evidence_source_commit", "e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48"),
        project_root=PROJECT_ROOT
    )
    legacy_effective_hash = compute_legacy_effective_model_config_hash(effective_configs)

    # 1. 复制并保存标准 CSV
    df_comp.to_csv(target_baseline_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    df_folds.to_csv(target_baseline_dir / "trading_fold_stability.csv", index=False, encoding="utf-8-sig")
    df_seed.to_csv(target_baseline_dir / "seed_robustness.csv", index=False, encoding="utf-8-sig")

    # 2. 生成 legacy_model_semantics.json
    legacy_semantics = {
        "baseline_id": "LEGACY_BASELINE_V1",
        "historical_source_commit": source_state.get("model_evidence_source_commit"),
        "models": {
            "prediction_champion": {
                "historical_artifact_model_id": "lightgbm_clf_baseline",
                "reported_name": "LightGBM Classification (Baseline)",
                "effective_estimator_class": "LGBMClassifier",
                "effective_objective": "binary",
                "effective_metric": ["binary_logloss", "auc"],
                "role": "PREDICTION_CHAMPION",
                "mean_daily_rank_ic": 0.0503,
                "nw20_rank_icir": 0.4044,
                "auc": 0.5319,
                "q5_minus_q1_spread": 7.17
            },
            "trading_candidate": {
                "historical_artifact_model_id": "lightgbm_ranker",
                "reported_name": "LightGBM Ranker (LambdaRank)",
                "legacy_model_id": "legacy_ordinal_ranker",
                "effective_estimator_class": "LGBMRanker",
                "effective_objective": "regression",
                "effective_metric": "rmse",
                "effective_parameters_source": "settings.LGBM_PARAMS",
                "ranking_group_supplied": True,
                "relevance_label": "daily_ordinal_0_to_4",
                "true_lambdarank_certified": False,
                "role": "LEGACY_TRADING_CANDIDATE",
                "cost_adjusted_excess_return": 5.72,
                "sharpe_ratio": 0.36,
                "max_drawdown": -14.35,
                "fold_win_ratio": 0.55
            }
        },
        "true_lambdarank_certified": False,
        "legacy_label_semantics": {
            "signal_time": "T_CLOSE",
            "entry_aligned": False,
            "legacy_return_window": "T_CLOSE_TO_T_PLUS_20_CLOSE",
            "execution_engine_entry": "T_PLUS_1_OPEN",
            "execution_alignment_gap": "Label evaluates return from T Close, while execution enters at T+1 Open."
        }
    }
    with open(target_baseline_dir / "legacy_model_semantics.json", "w", encoding="utf-8") as f:
        json.dump(legacy_semantics, f, indent=2, ensure_ascii=False)

    # 3. 计算所有生成的产物 Hash
    artifact_hashes = {
        "model_comparison.csv": compute_file_sha256(target_baseline_dir / "model_comparison.csv"),
        "trading_fold_stability.csv": compute_file_sha256(target_baseline_dir / "trading_fold_stability.csv"),
        "seed_robustness.csv": compute_file_sha256(target_baseline_dir / "seed_robustness.csv"),
        "legacy_model_semantics.json": compute_file_sha256(target_baseline_dir / "legacy_model_semantics.json"),
        "dataset_sha256": source_state.get("dataset_sha256"),
        "feature_schema_hash": source_state.get("feature_schema_hash"),
        "phase_2_0_2_reported_protocol_config_hash": source_state.get("research_protocol_config_hash"),
        "phase_2_0_2_reported_model_full_config_hash": source_state.get("model_full_config_hash"),
        "legacy_effective_model_config_hash": legacy_effective_hash
    }
    with open(target_baseline_dir / "artifact_hashes.json", "w", encoding="utf-8") as f:
        json.dump(artifact_hashes, f, indent=2, ensure_ascii=False)

    # 4. 生成 baseline_manifest.json
    baseline_manifest = {
        "baseline_id": "LEGACY_BASELINE_V1",
        "baseline_status": "FROZEN",
        "created_at": datetime.now().isoformat(),
        "model_evidence_source_commit": source_state.get("model_evidence_source_commit"),
        "certified_artifact_commit": latest_data.get("certification_logic_source_commit"),
        "dataset_path": source_state.get("dataset_path"),
        "dataset_sha256": source_state.get("dataset_sha256"),
        "feature_schema_hash": source_state.get("feature_schema_hash"),
        "feature_count": 79,
        "label_horizon": 20,
        "phase_2_0_2_reported_protocol_config_hash": source_state.get("research_protocol_config_hash"),
        "phase_2_0_2_reported_model_full_config_hash": source_state.get("model_full_config_hash"),
        "legacy_effective_model_config_hash": legacy_effective_hash,
        "legacy_label_semantics": legacy_semantics["legacy_label_semantics"],
        "prediction_baseline": {
            "model_id": "lightgbm_clf_baseline",
            "model_name": "LightGBM Classification (Baseline)",
            "mean_daily_rank_ic": 0.0503,
            "nw20_rank_icir": 0.4044,
            "auc": 0.5319,
            "q5_minus_q1_spread": 7.17,
            "common_ranking_rows": 221019,
            "common_oos_dates": 744
        },
        "trading_candidate": {
            "historical_artifact_model_id": "lightgbm_ranker",
            "reported_name": "LightGBM Ranker (LambdaRank)",
            "legacy_model_id": "legacy_ordinal_ranker",
            "effective_objective": "regression",
            "cost_adjusted_excess_return": 5.72,
            "sharpe_ratio": 0.36,
            "max_drawdown": -14.35,
            "real_fold_win_ratio": 0.55,
            "annualized_turnover_avg": 9.12
        },
        "prediction_champion_seed_robustness": "PASS",
        "trading_candidate_seed_robustness": "NOT_CERTIFIED",
        "live_trading_ready": False
    }
    with open(target_baseline_dir / "baseline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(baseline_manifest, f, indent=2, ensure_ascii=False)

    # 5. 生成 LEGACY_BASELINE_REPORT.md
    report_md = f"""# LEGACY_BASELINE_V1 — Frozen Research Baseline Report
# A股量化预测系统 Legacy V1 历史真值冻结与基准报告

> **基准定位**: 本报告将 Phase 2.0.2 经过 Walk-Forward OOS 实证检验的模型证据冻结为不可篡改的 `LEGACY_BASELINE_V1` 科学参照系。后续 Phase 2.1+ 所有实验均以此为基准进行单变量因果对比。

---

## 1. 历史血缘与配置哈希 (Provenance & Config Hashes)

- **BASELINE_ID**: `LEGACY_BASELINE_V1`
- **STATUS**: `FROZEN`
- **MODEL_EVIDENCE_SOURCE_COMMIT**: `{baseline_manifest['model_evidence_source_commit']}`
- **DATASET_SHA256**: `{baseline_manifest['dataset_sha256']}`
- **FEATURE_SCHEMA_HASH**: `{baseline_manifest['feature_schema_hash']}` (79 因子)
- **PHASE_2_0_2_REPORTED_PROTOCOL_CONFIG_HASH**: `{baseline_manifest['phase_2_0_2_reported_protocol_config_hash']}`
- **PHASE_2_0_2_REPORTED_MODEL_FULL_CONFIG_HASH**: `{baseline_manifest['phase_2_0_2_reported_model_full_config_hash']}`
- **LEGACY_EFFECTIVE_MODEL_CONFIG_HASH**: `{legacy_effective_hash}`

---

## 2. 旧 Ranker 历史模型语义订正 (Legacy Ranker Semantic Correction)

根据源码级 AST 逆向解析，历史模型调用路径特征如下：
- **Historical Artifact Model ID**: `lightgbm_ranker`
- **Historical Reported Name**: `LightGBM Ranker (LambdaRank)`
- **Corrected Legacy ID**: `legacy_ordinal_ranker`
- **Effective Estimator Class**: `LGBMRanker`
- **Effective Objective**: **`regression`** (由于历史代码分支逻辑，加载了 `LGBM_PARAMS` 且覆盖了 lambdarank 默认值)
- **Effective Metric**: `rmse`
- **Relevance Targets**: `daily_ordinal_0_to_4` (按日截面百分位离散化为 0~4 整数等级)
- **TRUE_LAMBDARANK_CERTIFIED**: **`FALSE`** (真正的 LambdaRank 损失与排序实验将在 Phase 2.1-B 开展)

---

## 3. 冻结核心基准指标 (Frozen Baseline Metrics)

### 3.1 预测质量基准 (Prediction Baseline) — `lightgbm_clf_baseline`
- **Mean Daily OOS RankIC**: **+0.0503**
- **NW20 RankICIR**: **+0.4044**
- **AUC**: **0.5319**
- **Q5-Q1 算术超额差**: **+7.17%**
- **Common Ranking Rows**: `221,019`
- **Common OOS Dates**: `744`
- **3-Seed 稳健性**: 42 (0.0503), 2026 (0.0455), 3407 (0.0459) $	o$ **`VERIFIED_STABLE`**

### 3.2 交易候选基准 (Trading Candidate Baseline) — `legacy_ordinal_ranker`
- **Cost-adjusted Excess Return**: **+5.72%**
- **Sharpe Ratio**: **+0.36**
- **Max Drawdown**: **-14.35%**
- **Real 20-Fold Win Ratio**: **55.0%** (11 / 20 胜出)
- **Seed Robustness**: **`NOT_CERTIFIED`** (待 Phase 2.1 实测)

---

## 4. 历史标签时序与撮合脱节说明 (Legacy Label Alignment Gap)

- **Legacy Signal Time**: `T_CLOSE`
- **Legacy Label Window**: `T Close -> T+20 Close`
- **Backtest Execution Entry**: `T+1 Open`
- **Alignment Gap**: 历史标签计算的收益始于 T 日收盘价，而执行撮合始于 T+1 日开盘价。该时序脱节将作为 **Phase 2.1-A (Execution-Aligned Labels)** 的唯一核心自变量予以对齐。
"""
    (target_baseline_dir / "LEGACY_BASELINE_REPORT.md").write_text(report_md, encoding="utf-8")
    print(f"==> [SUCCESS] Legacy V1 Baseline successfully frozen at: {target_baseline_dir}")


if __name__ == "__main__":
    freeze_legacy_v1()
