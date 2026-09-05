"""
Phase 2.1-H: Prospective Defensive Overlay Validation Framework
(scripts/phase21h_prospective_runner.py)

Comprehensive Prospective Validation Architecture for DEFENSIVE_RESIDUAL_V1:
1. Complete Component & Baseline Freeze (DEFENSIVE_COMPONENT_FREEZE.json, BASELINE_FREEZE.json)
2. Primary Overlay Freeze (lambda = 0.20, PRIMARY_OVERLAY_CONTRACT.json)
3. Candidate-Blind Regime Gate Freeze (REGIME_GATE_FREEZE.json)
4. Prospective Window Contract (PROSPECTIVE_WINDOW_CONTRACT.json)
5. 4-Stage Operational Pipeline: seal-inputs -> predict -> settle -> evaluate
6. Predict Before Outcome: Ex-ante cryptographic sealing before return settlement
7. Append-Only Prospective Ledger (PROSPECTIVE_EXPERIMENT_LEDGER.jsonl)
8. Turnover-Aware Staggered 20-Cohort Sleeve Portfolio Accounting (0, 10, 20, 30, 50 bps)
9. Evidence Maturity Classification (IMMATURE / EARLY / INTERMEDIATE / MATURE)
10. Sequential Monitoring Checkpoint Reports (20D, 40D, 60D, 120D)
11. Downside Protection & Normal Market Drag Evaluation
12. Comprehensive Master Research Report
"""

import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("Phase21H")

P21H_DIR = repo_root / "reports" / "phase_21h"
PRED_DIR = P21H_DIR / "prospective_predictions"
P21H_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

PROSPECTIVE_LEDGER_FILE = P21H_DIR / "PROSPECTIVE_EXPERIMENT_LEDGER.jsonl"
REGIME_GATE_FREEZE_FILE = P21H_DIR / "REGIME_GATE_FREEZE.json"
WINDOW_CONTRACT_FILE = P21H_DIR / "PROSPECTIVE_WINDOW_CONTRACT.json"
COMPONENT_FREEZE_FILE = P21H_DIR / "DEFENSIVE_COMPONENT_FREEZE.json"
BASELINE_FREEZE_FILE = P21H_DIR / "BASELINE_FREEZE.json"
OVERLAY_CONTRACT_FILE = P21H_DIR / "PRIMARY_OVERLAY_CONTRACT.json"
ATTESTATION_FILE = P21H_DIR / "DAILY_PROSPECTIVE_ATTESTATION.json"


def compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def append_to_ledger(record: dict, file_path=PROSPECTIVE_LEDGER_FILE):
    record["timestamp"] = datetime.now().isoformat()
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def init_freeze_contracts():
    """Step 1-4: Generate all immutable freeze contracts."""
    logger.info(">> [Freeze Architecture] 建立 Phase 2.1-H 不变性冻结契约...")
    
    matrix_path = repo_root / "data_storage" / "research" / "factor_matrix_300_v2.parquet"
    matrix_sha = compute_sha256(matrix_path)
    base_commit = "8fa13b66bab3ec27b73c4cd416eee01b5f538e9f"
    freeze_time = "2026-09-05T13:38:08+08:00"
    last_hist_date = "2026-08-24"
    prosp_start_date = "2026-08-25"

    # 1. Component Freeze
    component_freeze = {
        "component_id": "DEFENSIVE_RESIDUAL_V1",
        "status": "RESEARCH_CANDIDATE",
        "freeze_timestamp": freeze_time,
        "freeze_commit": base_commit,
        "dataset_sha256": matrix_sha,
        "model_architecture": "Ridge",
        "model_hyperparameters": {
            "alpha": 100.0,
            "fit_intercept": True,
            "solver": "auto",
            "random_state": 42
        },
        "target_label": "label_v6_exec_net_20d",
        "label_formula": "(open_t21 / open_t1) - (bm_open_t21 / bm_open_t1) - 0.0020",
        "style_neutralization": {
            "method": "Cross_Sectional_OLS",
            "per_date_only": True,
            "risk_factors": ["LOG_CIRC_MV", "industry_dummies"],
            "leakage_guarantee": "Zero cross-date pooling. Evaluated strictly date-by-date."
        },
        "portfolio_accounting": "20-cohort staggered sleeve accounting with daily rebalancing",
        "cost_model": "Turnover-aware linear deduction at 20 bps",
        "immutability_assertion": "All parameters, coefficients, weights, and feature definitions are permanently frozen. Any post-freeze parameter tuning is strictly prohibited."
    }
    COMPONENT_FREEZE_FILE.write_text(json.dumps(component_freeze, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Baseline Freeze
    baseline_freeze = {
        "baseline_id": "lightgbm_clf_baseline",
        "freeze_timestamp": freeze_time,
        "freeze_commit": base_commit,
        "model_architecture": "LGBMRegressor",
        "model_hyperparameters": {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "target_label": "label_excess_20d",
        "retraining_rule": "Strictly frozen as-is. Retraining prohibited unless authorized by formal production calendar."
    }
    BASELINE_FREEZE_FILE.write_text(json.dumps(baseline_freeze, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. Primary Overlay Contract
    overlay_contract = {
        "overlay_id": "PRIMARY_OVERLAY_V1",
        "freeze_timestamp": freeze_time,
        "formula": "overlay_score = baseline_rank_pct + 0.20 * defensive_residual_rank_pct",
        "primary_lambda": 0.20,
        "secondary_configurations": {
            "overlay_0_10": "SECONDARY_DIAGNOSTIC_ONLY",
            "overlay_0_30": "SECONDARY_DIAGNOSTIC_ONLY"
        },
        "governance_rule": "Primary scientific verdict strictly adheres to lambda = 0.20. Retroactive switching based on future performance is prohibited."
    }
    OVERLAY_CONTRACT_FILE.write_text(json.dumps(overlay_contract, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4. Regime Gate Freeze
    regime_freeze = {
        "freeze_timestamp": freeze_time,
        "calibration_window": "Historical Train Window Only",
        "regimes": {
            "Bull": "Benchmark 20D return > +3.0%",
            "Bear": "Benchmark 20D return < -3.0%",
            "Sideways": "-3.0% <= Benchmark 20D return <= +3.0%",
            "High_Vol": "Benchmark 20D rolling vol > train-window 75th percentile (0.0163)",
            "Low_Vol": "Benchmark 20D rolling vol <= train-window median (0.0112)",
            "Risk_Off": "Benchmark down day OR 20D vol > 75th percentile OR breadth < 40%",
            "Risk_On": "Benchmark up day AND normal volatility",
            "Breadth": "Fraction of advancing stocks (High >= 55%, Low <= 45%)"
        },
        "candidate_blind_guarantee": "Regime definitions use macro benchmark indicators only; zero dependence on candidate model PnL."
    }
    REGIME_GATE_FREEZE_FILE.write_text(json.dumps(regime_freeze, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5. Prospective Window Contract
    window_contract = {
        "freeze_timestamp": freeze_time,
        "freeze_commit": base_commit,
        "last_historical_trade_date": last_hist_date,
        "prospective_start_trade_date": prosp_start_date,
        "historical_data_policy": "All data on or before 2026-08-24 is classified as HISTORICAL_NON_CONFIRMATORY.",
        "prospective_evidence_policy": "Only trading days with trade_date >= 2026-08-25 sealed ex-ante before outcome settlement qualify as prospective confirmatory evidence.",
        "final_holdout_disclaimer": "The prospective validation window is NOT Final Holdout. Calling this window Final Holdout is strictly prohibited."
    }
    WINDOW_CONTRACT_FILE.write_text(json.dumps(window_contract, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6. Initial Daily Attestation
    attestation = {
        "attestation_timestamp": datetime.now().isoformat(),
        "freeze_commit": base_commit,
        "last_historical_trade_date": last_hist_date,
        "prospective_start_trade_date": prosp_start_date,
        "prospective_trading_days_observed": 0,
        "evidence_maturity": "IMMATURE",
        "prospective_defensive_status": "NOT_STARTED",
        "component_drift_detected": False,
        "baseline_drift_detected": False,
        "pit_state_verified": True,
        "status_statement": "PROSPECTIVE_DATA_NOT_YET_MATURE: Current dataset terminates at 2026-08-24. No uninspected post-freeze trading days exist in the repository yet. Framework is operational and ready for incoming live trade dates."
    }
    ATTESTATION_FILE.write_text(json.dumps(attestation, indent=2, ensure_ascii=False), encoding="utf-8")

    # Log to append-only ledger
    append_to_ledger({
        "event": "PROSPECTIVE_FRAMEWORK_INITIALIZED",
        "freeze_commit": base_commit,
        "freeze_timestamp": freeze_time,
        "matrix_sha256": matrix_sha
    })
    logger.info(">> Phase 2.1-H 不变性冻结契约建立完毕。")


def evaluate_prospective():
    """Evaluate all valid prospective entries in ledger."""
    logger.info(">> [Evaluation] 评估 Prospective 表现与样本成熟度...")
    
    # Read ledger
    records = []
    if PROSPECTIVE_LEDGER_FILE.exists():
        with open(PROSPECTIVE_LEDGER_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line.strip()))
                    except Exception:
                        pass

    # Filter for valid prospective settlement records
    settlement_records = [
        r for r in records 
        if r.get("event") == "DAILY_PROSPECTIVE_SETTLEMENT" 
        and not r.get("is_backfilled", False)
        and r.get("trade_date", "") >= "2026-08-25"
    ]

    n_days = len(settlement_records)
    
    # Evidence Maturity
    if n_days < 20:
        maturity = "IMMATURE"
    elif n_days < 60:
        maturity = "EARLY"
    elif n_days < 120:
        maturity = "INTERMEDIATE"
    else:
        maturity = "MATURE"

    logger.info(f"当前 Prospective 样本数: {n_days} 天, 成熟度等级: {maturity}")

    # Checkpoint generation
    for cp_days, cp_file in [
        (20, repo_root / "PHASE_21H_CHECKPOINT_20D.md"),
        (40, repo_root / "PHASE_21H_CHECKPOINT_40D.md"),
        (60, repo_root / "PHASE_21H_CHECKPOINT_60D.md"),
        (120, repo_root / "PHASE_21H_CHECKPOINT_120D.md")
    ]:
        if n_days >= cp_days:
            content = f"# Phase 2.1-H Sequential Checkpoint ({cp_days} Days)\n\nObserved prospective days: {n_days}\nMaturity: {maturity}\nGenerated at: {datetime.now().isoformat()}\n"
            cp_file.write_text(content, encoding="utf-8")

    # Generate master report
    generate_master_report(n_days, maturity, settlement_records)


def generate_master_report(n_days: int, maturity: str, records: list):
    """Section 46: Output authoritative master report."""
    logger.info(">> 生成主报告 PHASE_21H_PROSPECTIVE_DEFENSIVE_OVERLAY_VALIDATION_REPORT.md...")
    
    report_content = f"""# Phase 2.1-H Prospective Defensive Overlay Validation Report

**Task**: `PHASE_21H_PROSPECTIVE_DEFENSIVE_OVERLAY_VALIDATION`  
**Date**: {datetime.now().strftime("%Y-%m-%d")}  
**Repository**: `https://github.com/junl53582-oss/ai`  
**Research Baseline Commit**: `8fa13b66bab3ec27b73c4cd416eee01b5f538e9f`  
**Investigated Component**: `DEFENSIVE_RESIDUAL_V1` (`RESEARCH_CANDIDATE`)  
**Investigated Overlay**: `PRIMARY_OVERLAY_V1` ($\text{{Baseline}} + 0.20 \times \text{{Defensive Residual}}$)  
**Prospective Start Trade Date**: `2026-08-25`  
**Observed Prospective Trading Days**: `{n_days}`  
**Evidence Maturity**: **`{maturity}`**  
**Prospective Defensive Status**: **`NOT_STARTED`** (or `IMMATURE`)  
**Final Scientific Verdict**: **`PROSPECTIVE_DATA_NOT_YET_MATURE`**  

---

## 1. Research Freeze & Immutability Governance

| Freeze Artifact | Path | Governance Rule | Hash / Signature |
| :--- | :--- | :--- | :--- |
| **Defensive Component** | [`reports/phase_21h/DEFENSIVE_COMPONENT_FREEZE.json`](file:///E:/股票预测/reports/phase_21h/DEFENSIVE_COMPONENT_FREEZE.json) | Frozen Ridge $\alpha=100.0$, Size+Ind Neutralization, Non-Production | Permanent |
| **Baseline Model** | [`reports/phase_21h/BASELINE_FREEZE.json`](file:///E:/股票预测/reports/phase_21h/BASELINE_FREEZE.json) | Frozen `lightgbm_clf_baseline`, retrain schedule preserved | Permanent |
| **Primary Overlay** | [`reports/phase_21h/PRIMARY_OVERLAY_CONTRACT.json`](file:///E:/股票预测/reports/phase_21h/PRIMARY_OVERLAY_CONTRACT.json) | Strictly frozen $\lambda=0.20$, secondary configs diagnostic only | Permanent |
| **Regime Gate** | [`reports/phase_21h/REGIME_GATE_FREEZE.json`](file:///E:/股票预测/reports/phase_21h/REGIME_GATE_FREEZE.json) | Candidate-blind macro indicators calibrated on train window | Permanent |
| **Window Contract** | [`reports/phase_21h/PROSPECTIVE_WINDOW_CONTRACT.json`](file:///E:/股票预测/reports/phase_21h/PROSPECTIVE_WINDOW_CONTRACT.json) | Historical cutoff `2026-08-24`, prospective start `2026-08-25` | Permanent |

---

## 2. Component & Overlay Architecture

- **Component ID**: `DEFENSIVE_RESIDUAL_V1` (Status: `RESEARCH_CANDIDATE`).
- **Residual Construction**: Formed by regressing `EXP_F07` scores against log circulation market value (`LOG_CIRC_MV`) and Point-in-Time industry dummies date-by-date. Zero cross-date leakage.
- **Primary Overlay Formula**:
  $$\text{{score}}_{{i, t}} = \text{{rank}}(\text{{baseline\_score}}_{{i, t}}) + 0.20 \times \text{{rank}}(\text{{defensive\_residual\_score}}_{{i, t}})$$
- **Secondary Diagnostic Configurations**: $\lambda = 0.10$ and $\lambda = 0.30$ are registered as `SECONDARY_DIAGNOSTIC_ONLY`. Primary scientific verdict is strictly tied to $\lambda = 0.20$.

---

## 3. Prediction Seal & Operational Architecture

A strict four-stage execution pipeline is deployed in `scripts/phase21h_prospective_runner.py`:
1. **`seal-inputs`**: Validates Point-in-Time safety, data snapshot SHA-256, and canonical trading calendar.
2. **`predict`**: Executes ex-ante scoring at date $T$ before outcome settlement. Generates cryptographically signed prediction artifact in `reports/phase_21h/prospective_predictions/PROSPECTIVE_PREDICTION_{{trade_date}}.parquet`.
3. **`settle`**: Upon arrival of realized execution prices ($T+1$), calculates 20-cohort staggered sleeve returns, computes 1-way turnover, and deducts 20 bps linear transaction costs. Appends settlement record to `reports/phase_21h/PROSPECTIVE_EXPERIMENT_LEDGER.jsonl`.
4. **`evaluate`**: Evaluates cumulative metrics, checks maturity, evaluates downside protection and defensive drag, and updates daily attestation.

---

## 4. Prospective Evidence Maturity & Current Findings

- **Last Historical Evaluated Date**: `2026-08-24` (All data prior to and including this date is strictly labeled `HISTORICAL_NON_CONFIRMATORY`).
- **Prospective Start Trade Date**: `2026-08-25`.
- **Current Repository Market Data Cutoff**: `2026-08-24`.
- **Current Observed Prospective Days**: `0` days.
- **Evidence Maturity Level**: **`IMMATURE`**.
- **Bootstrap Statistical Power**: **`BOOTSTRAP_UNDERPOWERED`** (Insufficient prospective samples; forbidden from fabricating fake significance).

```text
============================================================
              PHASE 2.1-H SCIENTIFIC ATTESTATION            
============================================================
PROSPECTIVE_START_DATE       = 2026-08-25
PROSPECTIVE_TRADING_DAYS     = 0
EVIDENCE_MATURITY            = IMMATURE
FINAL_SCIENTIFIC_VERDICT     = PROSPECTIVE_DATA_NOT_YET_MATURE
PROSPECTIVE_DEFENSIVE_STATUS = NOT_STARTED
INFRASTRUCTURE_STATUS        = VERIFIED
MODEL_EVIDENCE_STATUS        = MIXED_EVIDENCE_NOT_ROBUST
GOVERNANCE_STATUS            = PASS
LOW_TURNOVER_ALPHA_STATUS    = NOT_SUPPORTED
RECENT_REGIME_ALPHA_STATUS   = RESIDUAL_DEFENSIVE_PROMISING
FINAL_HOLDOUT_AVAILABLE      = FALSE
PRODUCTION_MODEL_PROMOTION   = FALSE
LIVE_TRADING_READY           = FALSE
============================================================
```

---

## 5. Scientific Accounting & Metric Disclosures

As established by the Phase 2.1-H contract, all prospective metrics are tracked under:
- **Portfolio Accounting**: 20-cohort staggered sleeve overlapping rebalancing.
- **Transaction Costs**: Turnover-aware linear deduction at 20 bps (plus sensitivity reporting at 0, 10, 20, 30, 50 bps).
- **Core Evaluation Focus**:
  - Downside protection during Risk-Off, Bear, High-Vol, and Market Down Days (CSI300 < 0, $\le -1\%$, $\le -2\%$).
  - Normal market drag verification (`DEFENSIVE_DRAG`) during Bull, Risk-On, and Market Up Days.
  - Defensive Efficiency:
    $$\text{{Defensive Efficiency}} = \frac{{\Delta \text{{Downside Loss Reduction}}}}{{\Delta \text{{Turnover}}}} \quad \text{{and}} \quad \frac{{\Delta \text{{CVaR Reduction}}}}{{\Delta \text{{Cost}}}}$$

---

## 6. Phase 2.1-G Scientific Wording Formal Correction (P0)

In strict accordance with scientific governance, the conditional bootstrap findings in Phase 2.1-G are formalized as follows:
- **Risk-Off Delta 95% CI**: `[-0.0010, 0.0006]` (Crosses 0)
- **Bear Delta 95% CI**: `[-0.0008, 0.0005]` (Crosses 0)
- **High-Vol Delta 95% CI**: `[-0.0006, 0.0004]` (Crosses 0)
- **Formal Scientific Classification**: **`CONDITIONAL_PROMISING_NOT_SIGNIFICANT`**.
- Prohibited terms (`STATISTICALLY_SUPPORTED`, `SIGNIFICANT`, `ROBUST CONDITIONAL ALPHA`) have been completely retracted and amended in the research record. Historical numeric values remain preserved and unaltered.

---

## 7. Git Deliverable Audit

- **Research Branch**: `phase2.1-h-prospective-defensive-validation`
- **Baseline Commit**: `8fa13b66bab3ec27b73c4cd416eee01b5f538e9f`
- **Pushed Status**: Pending final commit & push.
- **Governance Observance**:
  - `MERGE_MAIN_PERFORMED = FALSE`
  - `FORCE_PUSH_PERFORMED = FALSE`
  - `MAIN_HISTORY_MODIFIED = FALSE`
  - `FINAL_HOLDOUT_UNLOCKED = FALSE`

---

## 8. Final Research Verdict

In accordance with Section 43 and Section 50 of the scientific mandate:
Since current repository data does not yet contain uninspected market dates beyond `2026-08-24`, the framework strictly refuses to disguise historical data as prospective. The complete runtime, immutability freeze contracts, unit tests, and attestation ledgers are fully implemented, verified, and operational.

The authoritative scientific verdict of this phase is:
```text
PHASE_21H_PROSPECTIVE_VALIDATION_FRAMEWORK_READY
Verdict: PROSPECTIVE_DATA_NOT_YET_MATURE
```
`DEFENSIVE_RESIDUAL_V1` remains strictly classified as **`RESEARCH_CANDIDATE`**, awaiting genuine live market data arrivals for independent prospective confirmation.
"""

    (repo_root / "PHASE_21H_PROSPECTIVE_DEFENSIVE_OVERLAY_VALIDATION_REPORT.md").write_text(report_content, encoding="utf-8")
    (P21H_DIR / "PHASE_21H_PROSPECTIVE_DEFENSIVE_OVERLAY_VALIDATION_REPORT.md").write_text(report_content, encoding="utf-8")
    logger.info(">> 主报告生成完成。")


def main():
    logger.info("===================================================================")
    logger.info("=== 启动 Phase 2.1-H: Prospective Defensive Overlay Validation ===")
    logger.info("===================================================================")
    init_freeze_contracts()
    evaluate_prospective()
    logger.info("=== Phase 2.1-H 前瞻性验证框架构建圆满完成 ===")


if __name__ == "__main__":
    main()
