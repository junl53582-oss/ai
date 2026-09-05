# Phase 2.1-H Prospective Defensive Overlay Validation Report

**Task**: `PHASE_21H_PROSPECTIVE_DEFENSIVE_OVERLAY_VALIDATION`  
**Date**: 2026-09-05  
**Repository**: `https://github.com/junl53582-oss/ai`  
**Research Baseline Commit**: `8fa13b66bab3ec27b73c4cd416eee01b5f538e9f`  
**Investigated Component**: `DEFENSIVE_RESIDUAL_V1` (`RESEARCH_CANDIDATE`)  
**Investigated Overlay**: `PRIMARY_OVERLAY_V1` ($	ext{Baseline} + 0.20 	imes 	ext{Defensive Residual}$)  
**Prospective Start Trade Date**: `2026-08-25`  
**Observed Prospective Trading Days**: `0`  
**Evidence Maturity**: **`IMMATURE`**  
**Prospective Defensive Status**: **`NOT_STARTED`** (or `IMMATURE`)  
**Final Scientific Verdict**: **`PROSPECTIVE_DATA_NOT_YET_MATURE`**  

---

## 1. Research Freeze & Immutability Governance

| Freeze Artifact | Path | Governance Rule | Hash / Signature |
| :--- | :--- | :--- | :--- |
| **Defensive Component** | [`reports/phase_21h/DEFENSIVE_COMPONENT_FREEZE.json`](file:///E:/股票预测/reports/phase_21h/DEFENSIVE_COMPONENT_FREEZE.json) | Frozen Ridge $lpha=100.0$, Size+Ind Neutralization, Non-Production | Permanent |
| **Baseline Model** | [`reports/phase_21h/BASELINE_FREEZE.json`](file:///E:/股票预测/reports/phase_21h/BASELINE_FREEZE.json) | Frozen `lightgbm_clf_baseline`, retrain schedule preserved | Permanent |
| **Primary Overlay** | [`reports/phase_21h/PRIMARY_OVERLAY_CONTRACT.json`](file:///E:/股票预测/reports/phase_21h/PRIMARY_OVERLAY_CONTRACT.json) | Strictly frozen $\lambda=0.20$, secondary configs diagnostic only | Permanent |
| **Regime Gate** | [`reports/phase_21h/REGIME_GATE_FREEZE.json`](file:///E:/股票预测/reports/phase_21h/REGIME_GATE_FREEZE.json) | Candidate-blind macro indicators calibrated on train window | Permanent |
| **Window Contract** | [`reports/phase_21h/PROSPECTIVE_WINDOW_CONTRACT.json`](file:///E:/股票预测/reports/phase_21h/PROSPECTIVE_WINDOW_CONTRACT.json) | Historical cutoff `2026-08-24`, prospective start `2026-08-25` | Permanent |

---

## 2. Component & Overlay Architecture

- **Component ID**: `DEFENSIVE_RESIDUAL_V1` (Status: `RESEARCH_CANDIDATE`).
- **Residual Construction**: Formed by regressing `EXP_F07` scores against log circulation market value (`LOG_CIRC_MV`) and Point-in-Time industry dummies date-by-date. Zero cross-date leakage.
- **Primary Overlay Formula**:
  $$	ext{score}_{i, t} = 	ext{rank}(	ext{baseline\_score}_{i, t}) + 0.20 	imes 	ext{rank}(	ext{defensive\_residual\_score}_{i, t})$$
- **Secondary Diagnostic Configurations**: $\lambda = 0.10$ and $\lambda = 0.30$ are registered as `SECONDARY_DIAGNOSTIC_ONLY`. Primary scientific verdict is strictly tied to $\lambda = 0.20$.

---

## 3. Prediction Seal & Operational Architecture

A strict four-stage execution pipeline is deployed in `scripts/phase21h_prospective_runner.py`:
1. **`seal-inputs`**: Validates Point-in-Time safety, data snapshot SHA-256, and canonical trading calendar.
2. **`predict`**: Executes ex-ante scoring at date $T$ before outcome settlement. Generates cryptographically signed prediction artifact in `reports/phase_21h/prospective_predictions/PROSPECTIVE_PREDICTION_{trade_date}.parquet`.
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
    $$	ext{Defensive Efficiency} = rac{\Delta 	ext{Downside Loss Reduction}}{\Delta 	ext{Turnover}} \quad 	ext{and} \quad rac{\Delta 	ext{CVaR Reduction}}{\Delta 	ext{Cost}}$$

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
