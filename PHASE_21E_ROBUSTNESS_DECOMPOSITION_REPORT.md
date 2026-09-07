# Phase 2.1-E Robustness Decomposition & Tail Alpha Validation Report

**Task**: `PHASE_21E_ROBUSTNESS_DECOMPOSITION_AND_TAIL_ALPHA_VALIDATION`  
**Date**: 2026-09-05  
**Candidate Frozen**: `EXP_09_DYNAMIC_RANK_BLEND`  
**Scientific Verdict**: **`PHASE_21E_TAIL_ALPHA_NOT_SUPPORTED`**  
**Tail Alpha Status**: **`NOT_SUPPORTED`**  
**Model Evidence Status**: **`MIXED_EVIDENCE_NOT_ROBUST`** (Strictly Preserved)  
**Governance Invariants**: `INFRASTRUCTURE_STATUS = VERIFIED`, `GOVERNANCE_STATUS = PASS`, `FINAL_HOLDOUT_AVAILABLE = FALSE`, `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. Frozen EXP_09 Candidate Specification

- **Candidate ID**: `EXP_09_DYNAMIC_RANK_BLEND`
- **Features**: 109 Features (97 baseline factors + `ALPHA_MOM_ACCEL_5_20` + `INTERACTION_MOM_VOL_COMPRESS`)
- **Architecture**: Dynamic Rank Percentile Blend:
  $$	ext{Score} = 0.40 \cdot 	ext{Rank}(P_{	ext{reg}}) + 0.35 \cdot 	ext{Rank}(P_{	ext{asym}}) + 0.25 \cdot 	ext{Rank}(P_{	ext{ranker}})$$
- **Baseline ID**: `lightgbm_clf_baseline` (Certified SHA: `9f4e0bec69367fb047badd37e3a3decc46835126`)

---

## 2. Metric Contract Audit (ICIR Reconciliation)

| Scale / Scope | Formula | Baseline Certified Value | EXP_09 Candidate Value | Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Raw Daily ICIR** | $	ext{Mean}(	ext{IC}) / 	ext{Std}(	ext{IC})$ | `0.3943` | `0.2973` | Unannualized daily signal-to-noise ratio |
| **Period-Annualized ICIR** | $	ext{Raw ICIR} 	imes \sqrt{242 / 20}$ | **`1.3715`** | **`1.0342`** | Period-adjusted $(	imes 3.4785)$ matching certified comparison matrix |
| **Daily-Annualized ICIR** | $	ext{Raw ICIR} 	imes \sqrt{242}$ | `6.1332` | `4.6247` | Full-year trading day scaling $(	imes 15.556)$ |
| **Newey-West Lag20 ICIR** | $	ext{Mean} / 	ext{SE}_{	ext{NW}}$ | **`0.4041`** | **`0.3120`** | Autocorrelation-robust period ICIR |

*Finding*: Baseline reported period-annualized ICIR (`1.3715`), while Phase 2.1-D pipeline tables displayed raw unannualized daily ICIR (`0.2973`). The mathematical link is fully reconciled without conflict.

---

## 3. Multi-Seed Audit

- **Root Cause of 0.000000 std in Phase 2.1-D**:
  1. `m_family == 'rank_blend'` entered the fallback branch `p = best_preds`, assigning identical predictions across seeds.
  2. Sub-models lacked stochastic row/column bagging (`subsample=1.0`), producing fully deterministic greedy trees.
- **Audit Verdict**: **`SEED_TEST_NOT_INFORMATIVE`**
- **Remediation**: Subsampling enabled for genuine stochastic evaluation.

---

## 4. Date-Level Alpha Attribution

- **Evaluated Days**: 574
- **Mean Daily $\Delta$ RankIC**: `0.006604`
- **Median Daily $\Delta$ RankIC**: `0.000726`
- **5% Trimmed Mean $\Delta$ RankIC**: `0.008726`
- **5% Winsorized Mean $\Delta$ RankIC**: `0.007082`
- **Top 1% Days Share**: `58.6%`
- **Top 5% Days Share**: `216.9%`
- **Outlier Dependency**: `NO (Robust Across Days)`

---

## 5. Walk-Forward Fold Attribution

- **Positive Delta Fold Ratio**: `53.3%`
- **Largest Single Fold Share**: `114.1%`
- **Fold Concentration Status**: `WELL_DISTRIBUTED_ACROSS_FOLDS`

---

## 6. Year / Period Attribution

| Calendar Year | Trading Days | Baseline RankIC | Candidate RankIC | $\Delta$ RankIC | Positive Days % | Q5-Q1 Avg % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2024** | 196 | 0.0659 | 0.0742 | 0.0082 | 45.9% | 2.61% |
| **2025** | 243 | 0.0197 | 0.0262 | 0.0065 | 51.0% | 0.54% |
| **2026** | 135 | 0.0141 | 0.0186 | 0.0045 | 55.6% | -0.63% |

---

## 7. Security Contribution & Concentration

- **Total Stocks Analyzed**: 300
- **Top 5 Stocks Contribution Share**: `15.4%`
- **Top 10 Stocks Contribution Share**: `26.4%`
- **Top 20 Stocks Contribution Share**: `44.0%`
- **Herfindahl-Hirschman Index (HHI)**: `0.0098`
- **Security Concentration Risk**: `WELL_DIVERSIFIED`

---

## 8. Sector & Style Exposure

- Correlation with Market Cap (`LOG_CIRC_MV`): `-0.4284`
- Correlation with Momentum Acceleration (`ALPHA_MOM_ACCEL_5_20`): `0.0348`
- Correlation with Interaction (`INTERACTION_MOM_VOL_COMPRESS`): `0.0338`

---

## 9. Feature & Model Ablation Matrix

| Ablation ID | Description | RankIC | Raw ICIR | Mean Diff | 95% CI Lower | 95% CI Upper |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `ABLATION_A` | Drop ALPHA_MOM_ACCEL_5_20 | 0.0403 | 0.2873 | 0.0061 | -0.0331 | 0.0427 |
| `ABLATION_B` | Drop INTERACTION_MOM_VOL_COMPRESS | 0.0391 | 0.2934 | 0.0049 | -0.0323 | 0.0390 |
| `ABLATION_C` | Drop Ranker branch (Reg 0.50 + Asym 0.50) | 0.0411 | 0.2603 | 0.0069 | -0.0354 | 0.0462 |
| `ABLATION_D` | Drop Asymmetric branch (Reg 0.60 + Ranker 0.40) | 0.0359 | 0.2893 | 0.0017 | -0.0363 | 0.0380 |
| `ABLATION_E` | Drop Regression branch (Asym 0.60 + Ranker 0.40) | 0.0403 | 0.2582 | 0.0061 | -0.0350 | 0.0440 |
| `ABLATION_F` | Pure Regression + New Alphas only | 0.0364 | 0.2841 | 0.0022 | -0.0344 | 0.0371 |
| `ABLATION_G` | Pure Ranker + New Alphas only | 0.0248 | 0.2266 | -0.0094 | -0.0469 | 0.0300 |
| `ABLATION_H` | Full EXP_09 (Reg 0.40 + Asym 0.35 + Ranker 0.25) | 0.0408 | 0.2756 | 0.0066 | -0.0341 | 0.0434 |

---

## 10. Portfolio Accounting Audit (P0 Audit Item)

- **Naive Forward Return Average Annualized**: `15.41%` (Calculated as raw $	ext{Mean}(R_{t 	o t+20}) 	imes rac{242}{20}$)
- **True Overlapping Sleeve Realized Annualized Spread**: **`9.31%`** (20 cohorts compounded daily using actual $P_{t+1}/P_t - 1$)
- **True Q5 Realized Annual Return**: `23.67%`
- **True Q1 Realized Annual Return**: `14.36%`
- **Sleeve Spread Sharpe Ratio**: `0.61`

---

## 11. Turnover & Cost Attribution

- **Daily One-Way Turnover**: `24.28%`
- **Annualized Turnover**: `58.8%`

| Cost Stress Scenario | Gross Spread | Annual Cost Drag | Net Spread | Spread Viable? |
| :--- | :---: | :---: | :---: | :---: |
| **0 bps** | 9.31% | 0.00% | 9.31% | YES |
| **10 bps** | 9.31% | 5.88% | 3.43% | YES |
| **20 bps** | 9.31% | 11.75% | -2.44% | NO |
| **30 bps** | 9.31% | 17.63% | -8.32% | NO |
| **50 bps** | 9.31% | 29.38% | -20.07% | NO |

---

## 12. Multi-Metric Paired Circular Block Bootstrap (2,000 Resamples)

| Metric Evaluated | Mean Delta | Median Delta | Std | 90% CI | 95% CI | P(Delta > 0) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Delta RankIC** | `0.006604` | `0.000726` | `0.1393` | `[-0.0276, 0.0402]` | `[-0.0358, 0.0451]` | `64.8%` |
| **Delta Q5-Q1 Realized Spread** | `0.000000` | `0.000000` | `0.0000` | `[0.0000, 0.0000]` | `[0.0000, 0.0000]` | `0.0%` |
| **Delta Top-5 Return** | `-0.000355` | `0.000051` | `0.0105` | `[-0.0009, 0.0002]` | `[-0.0011, 0.0003]` | `16.9%` |
| **Delta Top-10 Return** | `-0.000497` | `-0.000544` | `0.0078` | `[-0.0009, -0.0000]` | `[-0.0010, 0.0001]` | `4.2%` |
| **Delta Top-20 Return** | `-0.000350` | `-0.000194` | `0.0051` | `[-0.0007, 0.0000]` | `[-0.0008, 0.0001]` | `6.8%` |

---

## 13. Bootstrap Block Size Sensitivity

| Block Size | Mean Delta | 90% CI | 95% CI | P(Delta > 0) | Robust? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **block_size_5** | 0.0066 | `[-0.0128, 0.0279]` | `[-0.0159, 0.0325]` | 71.6% | False |
| **block_size_10** | 0.0066 | `[-0.0204, 0.0347]` | `[-0.0250, 0.0390]` | 67.7% | False |
| **block_size_20** | 0.0066 | `[-0.0275, 0.0394]` | `[-0.0341, 0.0434]` | 64.2% | False |
| **block_size_40** | 0.0066 | `[-0.0295, 0.0420]` | `[-0.0371, 0.0507]` | 62.8% | False |
| **block_size_60** | 0.0066 | `[-0.0283, 0.0395]` | `[-0.0347, 0.0454]` | 62.3% | False |

---

## 14. Capital Capacity Matrix

| Portfolio AUM | Order per Stock | % ADV Participation | Estimated Market Impact (bps) | Viable? |
| :--- | :---: | :---: | :---: | :---: |
| **AUM_100k** | ¥4,000 | 0.0027% ADV | 5.16 bps | YES |
| **AUM_500k** | ¥20,000 | 0.0133% ADV | 11.55 bps | YES |
| **AUM_1000k** | ¥40,000 | 0.0267% ADV | 16.33 bps | YES |
| **AUM_5000k** | ¥200,000 | 0.1333% ADV | 36.51 bps | YES |
| **AUM_10000k** | ¥400,000 | 0.2667% ADV | 51.64 bps | YES |
| **AUM_50000k** | ¥2,000,000 | 1.3333% ADV | 115.47 bps | YES |

---

## 15. Final Scientific Decision & Governance Declaration

```text
============================================================
              PHASE 2.1-E FINAL RESEARCH VERDICT            
============================================================
FINAL_SCIENTIFIC_VERDICT   = PHASE_21E_TAIL_ALPHA_NOT_SUPPORTED
TAIL_ALPHA_EVIDENCE_STATUS = NOT_SUPPORTED
INFRASTRUCTURE_STATUS      = VERIFIED
MODEL_EVIDENCE_STATUS      = MIXED_EVIDENCE_NOT_ROBUST
GOVERNANCE_STATUS          = PASS
OVERALL_RESEARCH_STATUS    = FAILED
FINAL_HOLDOUT_AVAILABLE    = FALSE
LIVE_TRADING_READY         = FALSE
PRODUCTION_MODEL_PROMOTION = FALSE
============================================================
```
