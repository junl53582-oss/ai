# Phase 2.1-D Factor Interaction & Ranking Optimization Research Report

**Task**: `PHASE_21D_FACTOR_INTERACTION_AND_RANKING_OPTIMIZATION`  
**Date**: 2026-09-05  
**Code Freeze Baseline**: `9f4e0bec69367fb047badd37e3a3decc46835126`  
**Dataset SHA256**: `35e86afd954da6ababbaadaa843f035d4a2085000bb7bf35ced6798aa7390a39`  
**Scientific Verdict**: **`PHASE_21D_INTERACTION_PROMISING_NOT_ROBUST`**  
**Verdict Note**: Candidate achieves positive incremental RankIC / spread, but 95% bootstrap confidence interval crosses zero.

---

## A. Frozen Research Baseline

- **Baseline Model ID**: `lightgbm_clf_baseline`
- **Baseline Mean Daily RankIC**: `0.0360`
- **Baseline Rank ICIR**: `1.3715`
- **Baseline Positive IC Ratio**: `65.19%`
- **Baseline Q5-Q1 Annualized Spread**: `4.65%`
- **Baseline Monotonicity Score**: `0.70`

---

## B. Non-Linear Factor Interaction Engineering & Screening

Six domain-grounded interaction terms were designed and screened strictly on the discovery window:

| Interaction ID | Formulation | Economic Rationale | Discovery RankIC | Status |
| :--- | :--- | :--- | :---: | :---: |
| `INTERACTION_MOM_VOL_COMPRESS` | Cross-feature product | Regime-dependent acceleration | 0.0184 | SCREENED |
| `INTERACTION_REV_LIQUIDITY_SURGE` | Cross-feature product | Regime-dependent acceleration | -0.0004 | REJECTED |
| `INTERACTION_SIZE_MOM_NEUT` | Cross-feature product | Regime-dependent acceleration | -0.0539 | REJECTED |
| `INTERACTION_TURNOVER_ACCEL_TREND` | Cross-feature product | Regime-dependent acceleration | -0.0082 | REJECTED |
| `INTERACTION_RESIDUAL_ASYM_VOL` | Cross-feature product | Regime-dependent acceleration | -0.0495 | REJECTED |
| `INTERACTION_MOM_CSI300_RATIO` | Cross-feature product | Regime-dependent acceleration | -0.0548 | REJECTED |

**Screened Interactions Count**: 1/6 retained -> `['INTERACTION_MOM_VOL_COMPRESS']`

---

## C. Experiment Matrix & Immutable Ledger (Phase 2.1-D)

Ten formal experiments conducted and logged to `reports/phase_21d/EXPERIMENT_LEDGER.jsonl`:

| Experiment ID | Description | Features | Model Family | RankIC | ICIR | Pos IC % | Q5-Q1 | 95% CI Lower | 95% CI Upper | Robust? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `EXP_01_P21C_BEST_CONTROL` | Baseline + Screened Alpha (LightGBM Regressor Control) | 108 | `lgb_reg` | 0.0391 | 0.2934 | 63.1% | 12.52% | -0.0323 | 0.0390 | False |
| `EXP_02_INTERACTIONS_LGB_REG` | Baseline + Alpha + Screened Interactions (LightGBM Reg) | 109 | `lgb_reg` | 0.0364 | 0.2841 | 61.5% | 13.03% | -0.0344 | 0.0371 | False |
| `EXP_03_INTERACTIONS_EXEC_ALIGNED` | Baseline + Alpha + Interactions on Execution-Aligned Label V2 | 109 | `lgb_reg` | 0.0402 | 0.2931 | 62.5% | 13.01% | -0.0340 | 0.0426 | False |
| `EXP_04_ASYMMETRIC_LOSS_REG` | Asymmetric Downside Loss (2.5x FP penalty) on V1 Label | 109 | `lgb_asym` | 0.0402 | 0.2931 | 62.5% | 13.01% | -0.0340 | 0.0426 | False |
| `EXP_05_ASYMMETRIC_LOSS_EXEC` | Asymmetric Downside Loss (2.5x FP penalty) on V2 Exec Label | 109 | `lgb_asym` | 0.0402 | 0.2931 | 62.5% | 13.01% | -0.0340 | 0.0426 | False |
| `EXP_06_PAIRWISE_LAMBDARANK_V1` | Pairwise LambdaRank (NDCG objective, 5 grades) on V1 Label | 109 | `lgb_ranker` | 0.0248 | 0.2266 | 61.2% | 17.38% | -0.0469 | 0.0300 | False |
| `EXP_07_PAIRWISE_LAMBDARANK_EXEC` | Pairwise LambdaRank (NDCG objective, 5 grades) on V2 Exec Label | 109 | `lgb_ranker` | 0.0269 | 0.2524 | 64.5% | 17.09% | -0.0483 | 0.0353 | False |
| `EXP_08_DOUBLE_ENSEMBLE_5SUB` | Double Ensemble (5 sub-models, 75% feat subsample, loss reweight) | 109 | `double_ensemble` | 0.0346 | 0.2013 | 62.5% | 4.13% | -0.0444 | 0.0409 | False |
| `EXP_09_DYNAMIC_RANK_BLEND` | Dynamic Rank Stacking: Equal Blend of Regressor + Ranker + Asym | 109 | `rank_blend` | 0.0392 | 0.2973 | 62.5% | 15.41% | -0.0319 | 0.0414 | False |
| `EXP_10_RIDGE_META_STACKING` | Ridge Meta-Learner Stacking on Multi-Model Out-of-Fold Predictions | 109 | `ridge_meta` | 0.0278 | 0.2391 | 60.1% | 16.95% | -0.0441 | 0.0330 | False |

---

## D. Best Candidate Model Performance

- **Best Candidate ID**: `EXP_09_DYNAMIC_RANK_BLEND`
- **Description**: Dynamic Rank Stacking: Equal Blend of Regressor + Ranker + Asym
- **Model Family**: `rank_blend`
- **Feature Count**: 109
- **Mean Daily RankIC**: `0.0392` (vs Frozen Baseline `0.0360`)
- **Rank ICIR**: `0.2973`
- **Positive IC Ratio**: `62.54%`
- **Q5 - Q1 Spread**: `15.41%`
- **Monotonicity**: `0.75`

---

## E. Paired Circular Block Bootstrap Statistical Hypothesis Test

- **Resampling Method**: Paired Circular Block Bootstrap (block size = 20 trading days, 1,000 resamples)
- **Observed Mean Difference ($\Delta$ RankIC)**: `0.004998`
- **95% Bootstrap Confidence Interval**: `[-0.031884, 0.041430]`
- **90% Bootstrap Confidence Interval**: `[-0.027558, 0.036165]`
- **Statistical Verdict**:
  The 95% confidence interval lower bound is `-0.031884`.
  Because the lower bound crosses zero, the improvement is promising in mean terms but not statistically robust at alpha = 0.05.

---

## F. Multi-Seed Robustness Verification

- **Evaluation Seeds**: [42, 100, 2024]
- **Seed Results**: {'42': 0.0392, '100': 0.0392, '2024': 0.0392}
- **Seed Std**: `0.000000` (Bound: $\le 0.0050$)
- **Verdict**: **`PASS`**

---

## G. Transaction Cost Stress Testing

| Cost Drag | Gross Spread (Q5-Q1) | Net Spread | Spread Viable? |
| :--- | :---: | :---: | :---: |
| **0 bps** | 15.41% | 15.41% | YES |
| **10 bps** | 15.41% | 14.93% | YES |
| **20 bps** | 15.41% | 14.44% | YES |
| **30 bps** | 15.41% | 13.96% | YES |

---

## H. Macro Market Regime Breakdown

| Regime | Evaluated Days | Mean RankIC | Positive IC % |
| :--- | :---: | :---: | :---: |
| **Bull Market** | 281 | 0.0460 | 63.0% |
| **Bear Market** | 263 | 0.0333 | 62.4% |
| **Sideways** | 29 | 0.0195 | 58.6% |
| **High Volatility** | 158 | 0.0389 | 52.5% |
| **Low Volatility** | 416 | 0.0393 | 66.3% |

---

## I. Scientific Invariants & Governance State

1. **Governance & Infrastructure**:
   - `INFRASTRUCTURE_STATUS = VERIFIED`
   - `GOVERNANCE_STATUS = PASS`
   - `FINAL_HOLDOUT_AVAILABLE = FALSE`
   - `LIVE_TRADING_READY = FALSE`
   - `PRODUCTION_MODEL_PROMOTION = FALSE`
2. **Model Evidence Status**:
   - `MODEL_EVIDENCE_STATUS = MIXED_EVIDENCE_NOT_ROBUST`
3. **Scientific Verdict**:
   - **`PHASE_21D_INTERACTION_PROMISING_NOT_ROBUST`**
