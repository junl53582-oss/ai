# Phase 2.1-B r3.2 — Evidence Integrity & Scientific Closure

Canonical run: `research_8dbf062_20260831_155701` generated from CODE_FREEZE_SHA `8dbf06213b9af0a6614f377513dc997aa80266be`.

| Group | Status | Machine evidence |
|---|---|---|
| Infrastructure | INSUFFICIENT_EVIDENCE | STRICT_FUNDAMENTAL_PIT has no official-announcement evidence (`official_announcement_rows=0`). |
| Model evidence | MIXED_EVIDENCE_NOT_ROBUST | Fixed seeds have RankIC std `0.007994 > 0.0050`; regression 95% CI lower is `-0.03723`. |
| Governance | PASS | Historical OOS is explicitly not a prospective final holdout. |
| Overall | FAILED | Fail-closed result; it is not a model-tuning result. |

Dataset SHA256: `9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42`.
Calendar SHA256: `cf08829d987632359ac12537a2d8659354aa1acc71057670ad81af2921eb0c23`.
Artifact manifest SHA256: `d006cb4d40efc9f9c189ecce017480054fbb27fd18d0c8b7084895aacf373211`.
Gate matrix SHA256: `deeeb4f0858bb7a86ef87c2937cd94d5602ab4a3d7e33f58bfce475b5acfa8ff`.

`FINAL_HOLDOUT_AVAILABLE = FALSE`; `LIVE_TRADING_READY = FALSE`; `PRODUCTION_MODEL_PROMOTION = FALSE`.

The earlier `research_3ebbb9f_20260831_155027` run is retained as historical evidence and superseded by the strict PIT code fix.
