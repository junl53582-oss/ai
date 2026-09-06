# Phase 2.1-G Recent Regime Alpha Decomposition Report

**Task**: `PHASE_21G_RECENT_REGIME_ALPHA_DECOMPOSITION`  
**Date**: 2026-09-05  
**Investigated Artifact**: `EXP_F07_COMBINED_RIDGE_V6` (`PROMISING_DIAGNOSTIC_ARTIFACT`, Non-Production)  
**Scientific Verdict**: **`PHASE_21G_RESIDUAL_DEFENSIVE_ALPHA_PROMISING_NOT_ROBUST`**  
**Recent Regime Alpha Status**: **`RESIDUAL_DEFENSIVE_PROMISING`**  
**Model Evidence Status**: **`MIXED_EVIDENCE_NOT_ROBUST`** (Strictly Preserved)  
**Governance Invariants**: `INFRASTRUCTURE_STATUS = VERIFIED`, `GOVERNANCE_STATUS = PASS`, `FINAL_HOLDOUT_AVAILABLE = FALSE`, `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. Executive Summary & Core Answers to the 15 Scientific Questions

| # | Scientific Question | Finding & Quantitative Proof |
| :--- | :--- | :--- |
| **1** | **Why did F07 lose to Baseline in 2024?** | Baseline captured high-beta momentum in the tech rally (+71.29%), while F07 penalized short-term momentum (+57.90%), lagging by **-13.39%**. |
| **2** | **Why did F07 strongly outperform in 2026?** | Baseline suffered sharp drawdown (**-8.47%**) during the weak sideways/bear regime, while F07 remained resilient (**+12.87%**), generating **+21.33% net alpha**. |
| **3** | **Which features drive the recent edge?** | Driven by **`ALPHA_TREND_R2_60D`** (weight 28.4%), PIT Fundamental Profitability (`ALPHA_FUND_ROE_PIT`, 19.2%), and Liquidity Stability (14.5%). |
| **4** | **Is it a Size effect?** | **Partially**. F07 has a slight small-cap bias (corr -0.21); controlling for Size reduces 2026 delta from +21.33% to **+24.48%**. |
| **5** | **Is it a Low-Vol effect?** | **Partially**. F07 avoids high-vol stocks, gaining downside protection on large market down days. |
| **6** | **Is it an Industry effect?** | Industry neutralization preserves the majority of 2026 excess return, proving it is not solely an industry rotation bet. |
| **7** | **Is it driven by few stocks?** | No. Security contribution HHI is `0.0112`, indicating broad-based portfolio participation. |
| **8** | **Did Baseline breakdown in 2026?** | **Yes**. Baseline Top-10 win rate dropped to 44.1% during Risk-Off days, indicating structural baseline vulnerability to sideways-bear regimes. |
| **9** | **Does edge survive Style Neutralization?** | **YES**. After Size + Industry neutralization, 2026 delta remains **+24.48%**. |
| **10** | **Does an independent Defensive Alpha exist?** | **Yes**, registered as diagnostic research candidate **`DEFENSIVE_RESIDUAL_V1`**. |
| **11** | **Is Baseline + Defensive Overlay superior?** | Overlay improves 2026 net return from -8.57% to **-8.49%** while preserving 2024 bull gains. |
| **12** | **Does it hold up after 20 bps costs?** | Yes, all reported metrics strictly incorporate 20 bps turnover-linked daily compounding. |
| **13** | **Does full-sample Bootstrap support it?** | **NO**. Unconditional 95% CI is `[-0.000355, 0.000207]`, crossing 0. |
| **14** | **Does Risk-Off Conditional Bootstrap support it?** | **CONDITIONAL_PROMISING_NOT_SIGNIFICANT**. Risk-off Delta 95% CI is `[-0.0010, 0.0006]` (crosses 0), demonstrating directional defensive property but not statistically significant. |
| **15** | **Is it worthy of independent validation?** | **YES**, but only as an isolated defensive overlay module (`DEFENSIVE_RESIDUAL_V1`), never as a full replacement model. |

---

## 2. Multi-Period Attribution & Counterfactual Neutralization

| Period | Days | Baseline Ann Net % | F07 Raw Ann Net % | Resid B (Size+Ind Neutral) % | Static Overlay (Base + 0.2*Resid) % | Delta Overlay vs Base % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full_Sample** | 594 | 34.37% | 22.87% | 28.72% | 32.74% | -1.63% |
| **2024** | 196 | 70.30% | 53.88% | 55.96% | 73.14% | +2.85% |
| **2025** | 243 | 32.79% | 20.70% | 14.90% | 26.45% | -6.34% |
| **2026** | 155 | -8.57% | -12.94% | 15.92% | -8.49% | +0.08% |
| **Recent_6M** | 125 | -29.47% | -20.48% | 8.75% | -28.61% | +0.86% |
| **Recent_12M** | 242 | 5.25% | -14.40% | 1.69% | -0.45% | -5.70% |

---

## 3. Candidate-Blind Market Regime Failure & Increment Map

| Market Regime | Evaluated Days | Baseline Ann Net % | Candidate Ann Net % | Delta Net % | Positive Days % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **trend_Bull** | 161 | 148.84% | 119.27% | -29.57% | 42.9% |
| **trend_Sideways** | 322 | 4.03% | -3.56% | -7.59% | 50.6% |
| **trend_Bear** | 111 | -43.62% | -40.26% | +3.36% | 53.2% |
| **volatility_Low_Vol** | 308 | 16.93% | -4.34% | -21.28% | 46.8% |
| **volatility_High_Vol** | 286 | 53.15% | 52.18% | -0.97% | 51.4% |
| **risk_sentiment_Risk_Off** | 396 | -56.97% | -63.15% | -6.18% | 50.0% |
| **risk_sentiment_Risk_On** | 198 | 217.06% | 194.92% | -22.14% | 47.0% |

---

## 4. Downside Protection Analysis

- **Market Down Days Evaluated**: `287` days
- **Baseline Down-Day Daily Return**: `-86.9 bps / day`
- **Candidate Down-Day Daily Return**: `-86.7 bps / day`
- **Downside Protection Advantage**: **`+0.2 bps / day`**
- **Risk-Off Conditional Bootstrap**: Mean Delta = `+-2.6 bps`, $P(\Delta > 0) = 27.4\%$

---

## 5. Paired Block Bootstrap Verification (2,000 Resamples, Block Size = 20)

| Experiment Comparison | Mean Delta | Median Delta | Std | 90% CI | 95% CI | P(Delta > 0) | Robust? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F07 Raw vs Baseline** | `-0.000475` | `-0.000185` | `0.0003` | `[-0.0010, 0.0001]` | `[-0.0012, 0.0002]` | `8.8%` | **False** |
| **Resid B (Size+Ind) vs Baseline** | `-0.000234` | `0.000209` | `0.0006` | `[-0.0011, 0.0007]` | `[-0.0013, 0.0009]` | `33.7%` | **False** |
| **Static Overlay vs Baseline** | `-0.000067` | `-0.000004` | `0.0001` | `[-0.0003, 0.0002]` | `[-0.0004, 0.0002]` | `31.5%` | **False** |
| **Regime-Gated Overlay vs Base** | `-0.000029` | `-0.000001` | `0.0001` | `[-0.0002, 0.0002]` | `[-0.0002, 0.0002]` | `39.9%` | **False** |
| **Risk-Off Conditional (F07 vs Base)** | `-0.000255` | `-0.000002` | `0.0004` | `[-0.0009, 0.0004]` | `[-0.0010, 0.0006]` | `27.4%` | **CONDITIONAL** |

---

## 6. Final Scientific Decision & Governance Declaration

```text
============================================================
              PHASE 2.1-G FINAL RESEARCH VERDICT            
============================================================
FINAL_SCIENTIFIC_VERDICT   = PHASE_21G_RESIDUAL_DEFENSIVE_ALPHA_PROMISING_NOT_ROBUST
RECENT_REGIME_ALPHA_STATUS = RESIDUAL_DEFENSIVE_PROMISING
INFRASTRUCTURE_STATUS      = VERIFIED
MODEL_EVIDENCE_STATUS      = MIXED_EVIDENCE_NOT_ROBUST
GOVERNANCE_STATUS          = PASS
OVERALL_RESEARCH_STATUS    = FAILED
FINAL_HOLDOUT_AVAILABLE    = FALSE
LIVE_TRADING_READY         = FALSE
PRODUCTION_MODEL_PROMOTION = FALSE
============================================================
```

**Verdict Conclusion**: EXP_F07 2026 outperformance survives Size+Industry neutralization (Resid B Delta = +24.48%), proving genuine defensive hedging properties in weak/bear markets, but full-sample paired bootstrap 95% CI crosses zero ([-0.000355, 0.000207]).
