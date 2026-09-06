# Phase 2.1-F Low-Turnover Net Alpha Research Report

**Task**: `PHASE_21F_LOW_TURNOVER_NET_ALPHA_DISCOVERY`  
**Date**: 2026-09-05  
**Best Candidate**: `EXP_F07_COMBINED_RIDGE_V6`  
**Scientific Verdict**: **`PHASE_21F_LOW_TURNOVER_ALPHA_DISCOVERY_INCONCLUSIVE`**  
**Low-Turnover Alpha Status**: **`NOT_SUPPORTED`**  
**Model Evidence Status**: **`MIXED_EVIDENCE_NOT_ROBUST`** (Strictly Preserved)  
**Governance Invariants**: `INFRASTRUCTURE_STATUS = VERIFIED`, `GOVERNANCE_STATUS = PASS`, `FINAL_HOLDOUT_AVAILABLE = FALSE`, `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. Frozen Baseline & EXP_09 Permanent Archive

- **Certified Baseline**: `lightgbm_clf_baseline` (97 features, Label V1 Control, 20-Sleeve Accounting)
- **EXP_09 Status**: **`PERMANENTLY_REJECTED`**
  - Archived Reason: Zero robust incremental alpha, tail alpha failure, temporal decay in 2026.
  - Policy: EXP_09 is permanently excluded from further hyperparameter tuning.

---

## 2. Low-Turnover Alpha Hypotheses (Families A ~ H)

We explored 21 structured economic hypotheses targeting persistent, low-churn signals:

| Family | Alpha ID | Economic Hypothesis | Autocorr Lag-1 | Autocorr Lag-20 | Screening Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| MOM | `ALPHA_MOM_20D` | Low-turnover signal persistence | 0.9349 | -0.0503 | `REJECTED` |
| MOM | `ALPHA_MOM_60D` | Low-turnover signal persistence | 0.9745 | 0.5888 | `REJECTED` |
| MOM | `ALPHA_MOM_CONSISTENCY_60D` | Low-turnover signal persistence | 0.9791 | 0.6595 | `REJECTED` |
| TREND | `ALPHA_TREND_SLOPE_60D` | Low-turnover signal persistence | 0.9745 | 0.5888 | `REJECTED` |
| TREND | `ALPHA_TREND_R2_60D` | Low-turnover signal persistence | 0.9441 | 0.4077 | `SCREENED` |
| MOM | `ALPHA_MOM_PERSISTENCE_60D` | Low-turnover signal persistence | 0.9748 | 0.5907 | `REJECTED` |
| REL | `ALPHA_REL_STRENGTH_20D` | Low-turnover signal persistence | 0.9349 | -0.0509 | `REJECTED` |
| REL | `ALPHA_REL_STRENGTH_60D` | Low-turnover signal persistence | 0.9746 | 0.5887 | `REJECTED` |
| REL | `ALPHA_REL_CONSISTENCY_60D` | Low-turnover signal persistence | 0.9822 | 0.6940 | `REJECTED` |
| FUND | `ALPHA_FUND_ROE_PIT` | Low-turnover signal persistence | 0.9934 | 0.9275 | `REJECTED` |
| FUND | `ALPHA_FUND_GROSS_MARGIN_PIT` | Low-turnover signal persistence | 0.9997 | 0.9930 | `REJECTED` |
| FUND | `ALPHA_FUND_REV_YOY_PIT` | Low-turnover signal persistence | 0.9964 | 0.9316 | `REJECTED` |
| FUND | `ALPHA_FUND_ROE_STABILITY` | Low-turnover signal persistence | 0.9995 | 0.9723 | `REJECTED` |
| FUND | `ALPHA_FUND_EARNINGS_SURPRISE` | Low-turnover signal persistence | 0.9892 | 0.7830 | `REJECTED` |
| FUND | `ALPHA_FUND_REV_SURPRISE` | Low-turnover signal persistence | 0.9902 | 0.8081 | `REJECTED` |
| QUALITY | `ALPHA_QUALITY_MOM_COMPOSITE` | Low-turnover signal persistence | 0.9815 | 0.7303 | `REJECTED` |
| RESIDUAL | `ALPHA_RESIDUAL_MOM_SIZE_NEUT` | Low-turnover signal persistence | 0.9749 | 0.5928 | `REJECTED` |
| ADV | `ALPHA_ADV_GROWTH_20_60` | Low-turnover signal persistence | 0.9884 | 0.1574 | `REJECTED` |
| TURNOVER | `ALPHA_TURNOVER_STABILITY_60D` | Low-turnover signal persistence | 0.9873 | 0.6416 | `REJECTED` |
| UPSIDE | `ALPHA_UPSIDE_DOWNSIDE_VOL_RATIO_60D` | Low-turnover signal persistence | 0.9965 | 0.9143 | `REJECTED` |
| VOL | `ALPHA_VOL_COMPRESSION_20_60` | Low-turnover signal persistence | 0.9449 | -0.0730 | `REJECTED` |

---

## 3. Holding Buffer & Rebalance Policy Evaluation

| Buffer Policy | Daily 1-Way Turnover | Annual Turns | Net Annual Return @ 20bps |
| :--- | :---: | :---: | :---: |
| **Top20_Hold20** | 16.65% | 40.3x | 197.68% |
| **Top20_Hold40** | 5.34% | 12.9x | 59.70% |
| **Top20_Hold50** | 4.49% | 10.9x | 53.35% |
| **Top30_Hold60** | 4.60% | 11.1x | 48.56% |

---

## 4. Matrix Experiments (Alpha Tiers x Label Candidates)

| Experiment ID | Feature Tier | Label Target | Model | RankIC | Annual Turns | Top-10 Net @ 20bps |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `EXP_F01_CONTROL_BASELINE` | CONTROL | label_excess_20d | lgb | 0.0303 | 7.7x | 35.41% |
| `EXP_F02_CORE_LGB_V1` | CORE_ALPHA | label_excess_20d | lgb | 0.0264 | 7.7x | 36.29% |
| `EXP_F03_CORE_LGB_V6_EXEC` | CORE_ALPHA | label_v6_exec_net_20d | lgb | 0.0275 | 7.8x | 32.12% |
| `EXP_F04_FUNDAMENTAL_LGB_V1` | FUNDAMENTAL_ALPHA | label_excess_20d | lgb | 0.0303 | 7.7x | 35.41% |
| `EXP_F05_MOM_QUALITY_LGB_V1` | MOMENTUM_QUALITY | label_excess_20d | lgb | 0.0264 | 7.7x | 36.29% |
| `EXP_F06_COMBINED_LGB_V6_EXEC` | COMBINED_LOW_TURNOVER | label_v6_exec_net_20d | lgb | 0.0275 | 7.8x | 32.12% |
| `EXP_F07_COMBINED_RIDGE_V6` | COMBINED_LOW_TURNOVER | label_v6_exec_net_20d | ridge | 0.0138 | 9.7x | 39.25% |

---

## 5. Candidate vs Frozen Baseline Core Comparison

| Metric | Frozen Baseline | Best Candidate (`EXP_F07_COMBINED_RIDGE_V6`) | Difference / Delta |
| :--- | :---: | :---: | :---: |
| **RankIC** | `0.0303` | `0.0138` | `+-0.0164` |
| **Annual One-Way Turnover** | `7.74 turns/yr` | `9.72 turns/yr` | **`-25.5%`** |
| **Top-10 Net Return @ 20bps** | `34.36%` | `36.42%` | `+2.06%` |
| **Top-20 Net Return @ 20bps** | `29.81%` | `30.65%` | `+0.84%` |
| **Q5-Q1 Net Return @ 20bps** | `6.69%` | `4.78%` | `+-1.91%` |
| **Net Alpha per Turn** | - | **`0.0375`** | Diagnostic efficiency metric |
| **Alpha Persistence Score** | - | **`0.6962`** | High autocorrelation persistence |

---

## 6. Paired Block Bootstrap (2,000 Resamples, Block Size = 20)

| Metric Evaluated | Mean Delta | Median Delta | Std | 90% CI | 95% CI | P(Delta > 0) | Robust? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Delta RankIC** | `-0.016446` | `-0.011737` | `0.0175` | `[-0.0466, 0.0115]` | `[-0.0509, 0.0166]` | `17.0%` | **False** |
| **Delta Top-10 Net Alpha** | `0.000085` | `0.000065` | `0.0004` | `[-0.0005, 0.0007]` | `[-0.0007, 0.0008]` | `58.6%` | **False** |
| **Delta Top-20 Net Alpha** | `0.000035` | `0.000274` | `0.0003` | `[-0.0004, 0.0005]` | `[-0.0005, 0.0005]` | `55.1%` | **False** |
| **Delta Q5-Q1 Net Alpha** | `-0.000079` | `-0.000157` | `0.0003` | `[-0.0006, 0.0005]` | `[-0.0007, 0.0006]` | `39.9%` | **False** |

---

## 7. Recent Period Stability

| Window / Year | Days | Baseline Ann Net % | Candidate Ann Net % | Delta Ann Net % | Positive Days % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2024** | 196 | 71.29% | 57.90% | -13.39% | 46.9% |
| **2025** | 243 | 31.90% | 34.12% | +2.23% | 50.2% |
| **2026** | 155 | -8.47% | 12.87% | +21.33% | 57.4% |
| **Recent_6M** | 125 | -29.87% | 8.31% | +38.18% | 58.4% |
| **Recent_12M** | 242 | 4.90% | 12.08% | +7.19% | 54.1% |

---

## 8. Multi-Seed Stochastic Verification

- **Random Seeds Evaluated**: `[42, 100, 2024]`
- **Subsampling Mode**: Enabled (`subsample=0.8, colsample_bytree=0.8`)
- **Evaluated RankIC Values**: `42: 0.0275, 100: 0.0271, 2024: 0.0276`
- **Seed Standard Deviation**: `0.000189`
- **Stochastic Mechanism Status**: **`STOCHASTIC_VERIFIED`** (Non-zero variation confirmed)

---

## 9. Final Scientific Decision & Governance Declaration

```text
============================================================
              PHASE 2.1-F FINAL RESEARCH VERDICT            
============================================================
FINAL_SCIENTIFIC_VERDICT   = PHASE_21F_LOW_TURNOVER_ALPHA_DISCOVERY_INCONCLUSIVE
LOW_TURNOVER_ALPHA_STATUS  = NOT_SUPPORTED
INFRASTRUCTURE_STATUS      = VERIFIED
MODEL_EVIDENCE_STATUS      = MIXED_EVIDENCE_NOT_ROBUST
GOVERNANCE_STATUS          = PASS
OVERALL_RESEARCH_STATUS    = FAILED
FINAL_HOLDOUT_AVAILABLE    = FALSE
LIVE_TRADING_READY         = FALSE
PRODUCTION_MODEL_PROMOTION = FALSE
============================================================
```

**Verdict Conclusion**: No candidate achieved meaningful or statistically significant cost-adjusted alpha improvement over the frozen baseline.
