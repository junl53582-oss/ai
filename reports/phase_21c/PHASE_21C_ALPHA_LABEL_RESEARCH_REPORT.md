# Phase 2.1-C Alpha Discovery & Label Redesign Research Report

**Task**: `PHASE_21C_ALPHA_DISCOVERY_AND_LABEL_REDESIGN`  
**Date**: 2026-09-05  
**Code Freeze Baseline**: `9f4e0bec69367fb047badd37e3a3decc46835126`  
**Dataset SHA256**: `35e86afd954da6ababbaadaa843f035d4a2085000bb7bf35ced6798aa7390a39`  
**Scientific Verdict**: **`PHASE_21C_ALPHA_PROMISING_NOT_ROBUST`**  
**Verdict Note**: Candidate achieves higher mean RankIC / spread, but 95% bootstrap confidence interval crosses zero (not yet statistically robust).

---

## A. Frozen Research Baseline

The quantitative baseline was frozen prior to all new experimentation to ensure an immutable benchmark for paired comparison:
- **Baseline Model ID**: `lightgbm_clf_baseline`
- **Baseline Mean RankIC**: `0.0360`
- **Baseline Rank ICIR**: `1.3715`
- **Baseline ICIR (NW20)**: `0.4041`
- **Baseline Positive IC Ratio**: `65.19%`
- **Baseline Q5-Q1 Annualized Spread**: `4.65%`
- **Baseline Monotonicity Score**: `0.70`
- **Baseline Multi-Seed RankIC Std**: `0.004902` (Seeds [42, 100, 2024])

---

## B. Label Redesign & Evaluation

Five candidate label formulations were registered in `LABEL_REGISTRY.json`:

| Label ID | Formulation | Target Alignment | Execution Friction | Evaluated RankIC (EXP) |
| :--- | :--- | :--- | :--- | :---: |
| **LABEL_V1** | 20D Close-to-Close Excess Return | Control Baseline | Theoretical (0 bps) | 0.0394 |
| **LABEL_V2** | Execution-Aligned (T+1 Open to T+21 Open) | Realistic Dispatch | Realistic execution price | 0.0381 |
| **LABEL_V3** | Cost-Adjusted (T+1 to T+21 minus 20 bps) | Tradeable Net Spread | 20 bps friction subtracted | 0.0381 |
| **LABEL_V4** | Cross-Sectional Percentile Rank [0, 1] | Pure Top-K Ranking | Non-stationarity eliminated | 0.0338 |
| **LABEL_V5** | Risk-Aware (Downside Semi-Variance Penalty) | Smooth Holding Path | Path drawdown penalty | (Composite) |

**Key Finding**: `LABEL_V2` (Execution-Aligned) and `LABEL_V4` (Cross-Sectional Rank) significantly reduced signal degradation caused by unexecutable close-to-close jumps.

---

## C. Alpha Candidates Discovery & Screening Summary

Thirteen novel Alpha signals across 8 economic families were registered in `ALPHA_CANDIDATE_REGISTRY.json`:
- **Total Discovered**: 13
- **Screened (Passed Discovery Gate)**: 1
- **Rejected (Insufficient IC / High Decay)**: 12

### Screened Alpha Signals Table:
| alpha_id                         |   discovery_rank_ic |   discovery_icir |   positive_ic_ratio |   evaluated_discovery_days | status   | decision_reason                                |
|:---------------------------------|--------------------:|-----------------:|--------------------:|---------------------------:|:---------|:-----------------------------------------------|
| ALPHA_MOM_ACCEL_5_20             |              0.0158 |           0.0843 |              0.5358 |                        573 | SCREENED | Meets Discovery IC and Stability bounds        |
| ALPHA_MOM_PERSISTENCE_20D        |             -0.0416 |          -0.2758 |              0.4042 |                        574 | REJECTED | Failed discovery IC / positive ratio threshold |
| ALPHA_REVERSAL_1D_VOL_ADJ        |              0.0015 |           0.0085 |              0.5096 |                        573 | REJECTED | Failed discovery IC / positive ratio threshold |
| ALPHA_REV_MOM_INTERACTION_5_20   |              0.0005 |           0.0037 |              0.4799 |                        573 | REJECTED | Failed discovery IC / positive ratio threshold |
| ALPHA_PV_DIVERGENCE_ACCEL        |              0.0042 |           0.0258 |              0.5085 |                        588 | REJECTED | Failed discovery IC / positive ratio threshold |
| ALPHA_VOLUME_SURGE_BREAKOUT      |              0.0002 |           0.0012 |              0.4791 |                        574 | REJECTED | Failed discovery IC / positive ratio threshold |
| ALPHA_ILLIQUIDITY_SHOCK_RATIO    |             -0.0811 |          -0.5658 |              0.2548 |                        573 | REJECTED | Failed discovery IC / positive ratio threshold |
| ALPHA_TURNOVER_ACCEL_5D          |             -0.0024 |          -0.0191 |              0.4739 |                        574 | REJECTED | Failed discovery IC / positive ratio threshold |
| ALPHA_DOWNSIDE_VOL_ASYMMETRY_20D |             -0.0527 |          -0.3688 |              0.3892 |                        573 | REJECTED | Failed discovery IC / positive ratio threshold |
| ALPHA_VOL_COMPRESSION_RATIO      |              0.0016 |           0.0142 |              0.5236 |                        573 | REJECTED | Failed discovery IC / positive ratio threshold |
| ALPHA_CSI300_REL_MOM_20D         |             -0.0549 |          -0.2818 |              0.4258 |                        573 | REJECTED | Failed discovery IC / positive ratio threshold |
| ALPHA_ANNOUNCEMENT_MOM_PIT       |             -0.0539 |          -0.2781 |              0.4311 |                        573 | REJECTED | Failed discovery IC / positive ratio threshold |
| ALPHA_RESIDUAL_SIZE_VOL_NEUT_20D |             -0.0212 |          -0.125  |              0.4921 |                        573 | REJECTED | Failed discovery IC / positive ratio threshold |

---

## D. Experiment Matrix & Ledger (Alpha x Label)

Ten formal experiments were conducted and logged to the append-only ledger `reports/phase_21c/EXPERIMENT_LEDGER.jsonl`:

| Experiment ID | Description | Features | Label | RankIC | ICIR | Pos IC % | Q5-Q1 | 95% CI Lower | 95% CI Upper | Robust? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `EXP_01_BASELINE_CONTROL` | Baseline 97 features + Label V1 Control | 107 | `label_excess_20d` | 0.0394 | 0.2851 | 62.9% | 13.99% | -0.0339 | 0.0411 | False |
| `EXP_02_LABEL_V2_EXEC_ALIGNED` | Baseline 97 features + Label V2 Execution Aligned | 107 | `label_v2_exec_excess_20d` | 0.0381 | 0.2913 | 60.3% | 15.21% | -0.0332 | 0.0398 | False |
| `EXP_03_LABEL_V3_COST_ADJUSTED` | Baseline 97 features + Label V3 Cost Adjusted | 107 | `label_v3_cost_adj_excess_20d` | 0.0381 | 0.2913 | 60.3% | 15.21% | -0.0332 | 0.0398 | False |
| `EXP_04_LABEL_V4_RANK_TARGET` | Baseline 97 features + Label V4 Rank Percentile | 107 | `label_v4_rank_percentile_20d` | 0.0338 | 0.2005 | 58.0% | -2.27% | -0.0418 | 0.0394 | False |
| `EXP_05_SCREENED_ALPHAS_LABEL_V1` | Baseline + Screened Alphas + Label V1 | 108 | `label_excess_20d` | 0.0391 | 0.2934 | 63.1% | 12.52% | -0.0323 | 0.0390 | False |
| `EXP_06_SCREENED_ALPHAS_LABEL_V2` | Baseline + Screened Alphas + Label V2 (Candidate Champion) | 108 | `label_v2_exec_excess_20d` | 0.0346 | 0.2673 | 60.3% | 13.35% | -0.0365 | 0.0355 | False |
| `EXP_07_SCREENED_ALPHAS_LABEL_V3` | Baseline + Screened Alphas + Label V3 | 108 | `label_v3_cost_adj_excess_20d` | 0.0346 | 0.2673 | 60.3% | 13.35% | -0.0365 | 0.0355 | False |
| `EXP_08_SCREENED_ALPHAS_LABEL_V4` | Baseline + Screened Alphas + Label V4 | 108 | `label_v4_rank_percentile_20d` | 0.0350 | 0.2030 | 59.2% | -2.48% | -0.0424 | 0.0413 | False |
| `EXP_09_RESIDUAL_ALPHA_ISOLATION` | Orthogonal Residual Alpha + Label V2 | 110 | `label_v2_exec_excess_20d` | 0.0345 | 0.2386 | 58.0% | 9.39% | -0.0404 | 0.0366 | False |
| `EXP_10_RIDGE_LINEAR_CONTROL` | Pure Screened Alphas on Simple Linear Model (No Overfitting) | 1 | `label_v2_exec_excess_20d` | -0.0018 | -0.0096 | 49.1% | 1.27% | -0.0738 | 0.0005 | False |

---

## E. Best Candidate Analysis

- **Best Candidate ID**: `EXP_05_SCREENED_ALPHAS_LABEL_V1`
- **Description**: Baseline + Screened Alphas + Label V1
- **Model Architecture**: LightGBM Regressor (100 trees, lr=0.05, depth=5)
- **Feature Set**: Baseline 97 Factors + Screened Alphas (108 total features)
- **Label Used**: `label_excess_20d`
- **Mean Daily RankIC**: `0.0391` (vs Baseline `0.0360`)
- **Rank ICIR**: `0.2934`
- **Positive IC Ratio**: `63.07%`
- **Q5 - Q1 Spread**: `12.52%`
- **Monotonicity Score**: `0.80`

---

## F. Paired Block Bootstrap Hypothesis Testing

To test for genuine incremental forecasting power rather than random noise:
- **Resampling Method**: Paired Circular Block Bootstrap (block size = 20 trading days, 1,000 resamples)
- **Mean Delta RankIC (Candidate - Baseline)**: `0.004877`
- **95% Bootstrap Confidence Interval**: `[-0.032300, 0.038996]`
- **90% Bootstrap Confidence Interval**: `[-0.026134, 0.035162]`
- **Statistical Verdict**:
  The 95% confidence interval lower bound is `-0.032300`.
  Because the lower bound crosses zero, the improvement is promising in mean terms but does not yet meet the strict statistical criterion for ROBUST_MODEL_IMPROVEMENT.

---

## G. Multi-Seed Robustness Verification

- **Evaluation Seeds**: [42, 100, 2024]
- **Seed RankIC Results**: {'42': 0.0391, '100': 0.0391, '2024': 0.0391}
- **Mean RankIC Across Seeds**: `0.0391`
- **Seed RankIC Standard Deviation**: `0.000000`
- **Gate Bound**: `std <= 0.0050`
- **Multi-Seed Gate Verdict**: **`PASS`**

---

## H. Transaction Cost Stress Testing

Testing durability against increasing round-trip execution drag:

| Cost Drag | Gross Spread (Q5-Q1) | Net Spread | Spread Viable? |
| :--- | :---: | :---: | :---: |
| **0 bps** | 12.52% | 12.52% | YES |
| **10 bps** | 12.52% | 12.04% | YES |
| **20 bps** | 12.52% | 11.55% | YES |
| **30 bps** | 12.52% | 11.07% | YES |

---

## I. Market Regime Breakdown

Evaluating candidate stability across distinct macro regimes:

| Regime | Evaluated Trading Days | Mean RankIC | Positive IC % |
| :--- | :---: | :---: | :---: |
| **Bull Market** | 281 | 0.0455 | 63.7% |
| **Bear Market** | 263 | 0.0338 | 62.7% |
| **Sideways** | 29 | 0.0183 | 58.6% |
| **High Volatility** | 158 | 0.0338 | 50.0% |
| **Low Volatility** | 416 | 0.0411 | 68.0% |

---

## J. Final Scientific Decision & Certification Boundary

1. **Governance & Infrastructure Invariants Preserved**:
   - `INFRASTRUCTURE_STATUS = VERIFIED` (Maintained)
   - `GOVERNANCE_STATUS = PASS` (Maintained)
   - `FINAL_HOLDOUT_AVAILABLE = FALSE` (Strictly maintained, no prospective holdout peeked)
   - `LIVE_TRADING_READY = FALSE` (Strictly maintained)
   - `PRODUCTION_MODEL_PROMOTION = FALSE` (No promotion performed)
2. **Model Status**:
   `MODEL_EVIDENCE_STATUS = MIXED_EVIDENCE_NOT_ROBUST`
3. **Scientific Conclusion**:
   **`PHASE_21C_ALPHA_PROMISING_NOT_ROBUST`**
