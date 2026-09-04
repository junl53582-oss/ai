# Phase 2.1-E1 Scientific Accounting Reconciliation Report

**Task**: `PHASE_21E1_SCIENTIFIC_ACCOUNTING_RECONCILIATION`  
**Date**: 2026-09-05  
**Candidate Frozen**: `EXP_09_DYNAMIC_RANK_BLEND`  
**Scientific Verdict**: **`PHASE_21E_TAIL_ALPHA_NOT_SUPPORTED`**  
**Tail Alpha Evidence Status**: **`NOT_SUPPORTED`**  
**Model Evidence Status**: **`MIXED_EVIDENCE_NOT_ROBUST`** (Strictly Preserved)  
**Governance Invariants**: `INFRASTRUCTURE_STATUS = VERIFIED`, `GOVERNANCE_STATUS = PASS`, `FINAL_HOLDOUT_AVAILABLE = FALSE`, `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. Executive Summary & Verdict Stability

This phase conducts a root-cause forensic audit of all statistical, dimensional, and portfolio accounting discrepancies identified in Phase 2.1-E. 

**Core Scientific Finding**:
> **The rejection of `EXP_09_DYNAMIC_RANK_BLEND` remains robustly confirmed.** While correcting the 20-sleeve portfolio turnover eliminates the artificial 58.75x turnover penalty and restores the genuine bootstrap distribution for $\Delta 	ext{Q5-Q1}$, the candidate model's 95% bootstrap confidence intervals for both RankIC (`[-0.0065, 0.0093]`) and Q5-Q1 (`[-0.0002, 0.0001]`) **cross zero**, fold contributions are **`MIXED`**, and performance in 2026 suffers severe temporal decay.

---

## 2. Turnover Accounting Contract

| Dimension | Raw Single-Sleeve (Phase 2.1-E Flawed) | 20-Sleeve Portfolio (Phase 2.1-E1 Reconciled) | Unit |
| :--- | :---: | :---: | :---: |
| **Mean Daily One-Way Turnover** | `24.28%` (0.2428) | **`1.61%`** (`0.0161`) | decimal per day |
| **Median Daily One-Way Turnover** | - | **`1.50%`** (`0.0150`) | decimal per day |
| **P90 Daily One-Way Turnover** | - | **`2.17%`** (`0.0217`) | decimal per day |
| **P95 Daily One-Way Turnover** | - | **`2.50%`** (`0.0250`) | decimal per day |
| **Annualized One-Way Turnover** | `58.75x` (labeled as `58.8%`) | **`3.90 turns/yr`** (`390%`) | turns per year |

- **Dimensional Conflict Root Cause**: Phase 2.1-E computed turnover for an un-smoothed daily-rebalanced single basket ($24.28\% / 	ext{day}$) and multiplied by 242 ($58.75	ext{x}$), but mistakenly attached a `%` sign (`58.8%`), causing a 100x scale confusion.
- **Sleeve Smoothing Mechanics**: Under 20 staggered sleeves, only 1 sleeve ($5\%$ capital) rebalances each day, reducing daily portfolio turnover by a factor of ~6.3x.

---

## 3. Cost Attribution & Daily Compounding Recalculation

| Cost Rate | Gross Annual Compound | Annual Cost Drag | Net Annual Compound | Linear Approximation Net | Spread Viable? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0 bps** | `11.90%` | `0.00%` | `11.90%` | `12.42%` | **YES** |
| **10 bps** | `11.90%` | `0.87%` | `10.93%` | `11.55%` | **YES** |
| **20 bps** | `11.90%` | `1.74%` | `9.97%` | `10.68%` | **YES** |
| **30 bps** | `11.90%` | `2.61%` | `9.01%` | `9.81%` | **YES** |
| **50 bps** | `11.90%` | `4.35%` | `7.13%` | `8.06%` | **YES** |

*Note*: Official results use exact daily compounding $\prod (1 + R_t^{\text{gross}} - c_t) - 1$.

---

## 4. Delta Q5-Q1 = 0 Root Cause Audit

- **Anomaly in Phase 2.1-E**: $\Delta 	ext{Q5-Q1}$ had mean=0, std=0, CI=[0, 0].
- **Forensic Diagnosis**: In `scripts/phase21e_pipeline.py:659`, `compute_sleeve_overlapping_returns` hardcoded `sub["pred_score"]` without accepting a column parameter. `df_eval_base` was copied without renaming, evaluating candidate predictions twice.
- **Remediation**: Re-fitted independent baseline and candidate models, passed explicit prediction columns, verified `max_abs_diff = 1.000980 > 1e-4`, and re-calculated sleeve holdings.

---

## 5. Corrected Multi-Metric Paired Block Bootstrap (2,000 Resamples)

| Metric | Mean Delta | Median Delta | Std | 90% CI | 95% CI | P(Delta > 0) | Robust? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Delta RankIC** | `0.001432` | `0.003386` | `0.0040` | `[-0.0052, 0.0080]` | `[-0.0065, 0.0093]` | `65.3%` | **False** |
| **Delta Q5-Q1 Spread** | `-0.000060` | `-0.000058` | `0.0001` | `[-0.0002, 0.0001]` | `[-0.0002, 0.0001]` | `26.7%` | **False** |
| **Delta Top-5 Return** | `-0.000073` | `0.000193` | `0.0003` | `[-0.0006, 0.0005]` | `[-0.0007, 0.0006]` | `41.1%` | **False** |
| **Delta Top-10 Return** | `-0.000367` | `-0.000089` | `0.0003` | `[-0.0008, 0.0001]` | `[-0.0009, 0.0002]` | `10.1%` | **False** |
| **Delta Top-20 Return** | `-0.000267` | `-0.000118` | `0.0002` | `[-0.0006, 0.0001]` | `[-0.0006, 0.0001]` | `9.0%` | **False** |
| **Delta Net Alpha @ 20bps** | `-0.000060` | `-0.000058` | `0.0001` | `[-0.0002, 0.0001]` | `[-0.0002, 0.0001]` | `26.7%` | **False** |

---

## 6. Fold Attribution Reconciliation

- **Total Folds Evaluated**: `16`
- **Positive Delta Folds**: `9` (`56.2%`)
- **Negative Delta Folds**: `7`
- **Largest Positive Fold Delta**: `+0.0223`
- **Largest Positive Fold Share**: **`64.6%`**
- **Corrected Fold Classification**: **`MIXED`**
- *Reasoning*: Because negative folds exist (e.g. Fold 13 delta = -0.1969), positive fold delta exceeds 100% of the net sum. Code logic strictly categorizes this as `MIXED`.

---

## 7. Before vs Corrected Comparison

| Metric / Audit Item | Phase 2.1-E Reported Value | Corrected Phase 2.1-E1 Value | Root Cause & Scientific Impact |
| :--- | :---: | :---: | :--- |
| **Daily One-Way Turnover** | 24.28% | **1.61%** | Evaluated 20-sleeve aggregate holdings instead of raw single-sleeve. |
| **Annualized Turnover** | 58.8% | **3.90 turns/yr** | Corrected 100x scale ambiguity and sleeve smoothing. |
| **Delta Q5-Q1 95% CI** | [0.0000, 0.0000] | **[-0.0002, 0.0001]** | Fixed hardcoded pred_score bug in sleeve returns. |
| **Fold Concentration** | WELL_DISTRIBUTED | **MIXED** | Corrected classification rule when single fold share > 100%. |
| **Cost Net Spread @ 20bps** | -2.44% | **9.97%** | Daily compounding on true sleeve turnover. |

---

## 8. Final Scientific Decision & Governance Declaration

```text
============================================================
             PHASE 2.1-E1 SCIENTIFIC VERDICT               
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

**Verdict Summary**: Even after rigorous accounting reconciliation: Delta RankIC 95% CI [-0.0065, 0.0093] and Delta Q5-Q1 95% CI [-0.0002, 0.0001] both cross zero, fold concentration is MIXED, and recent 2026 spread is negative (-0.03%). Rejection verdict stands.
