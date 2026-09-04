# POST MERGE INFRASTRUCTURE EVIDENCE CLOSEOUT

> **Task Name**: `POST_MERGE_INFRASTRUCTURE_EVIDENCE_CLOSURE`  
> **Repository**: [junl53582-oss/ai](https://github.com/junl53582-oss/ai)  
> **Evaluation Timestamp**: 2026-09-05T01:52:00+08:00  
> **Final Verdict**: `INFRASTRUCTURE_EVIDENCE_STILL_INSUFFICIENT`

---

## 1. Executive Summary & Canonical States

| Metric / Gate | Certified Research Value | Evaluation Source |
| :--- | :---: | :--- |
| **INFRASTRUCTURE_STATUS** | `INSUFFICIENT_EVIDENCE` | 9-Gate Research Certification Suite (`research_v2/governance/certification.py`) |
| **MODEL_EVIDENCE_STATUS** | `MIXED_EVIDENCE_NOT_ROBUST` | Bootstrap CI 95% Lower < 0 & Multi-seed variance > threshold |
| **GOVERNANCE_STATUS** | `PASS` | Holdout & Live Trading Governance verified |
| **OVERALL_RESEARCH_STATUS** | `FAILED` | Fail-Closed Policy (Infra Insufficient + Model Mixed) |
| **FINAL_HOLDOUT_AVAILABLE** | `FALSE` | Anti-overfitting prospective holdout unreleased |
| **LIVE_TRADING_READY** | `FALSE` | Strict fail-closed prohibition |
| **PRODUCTION_MODEL_PROMOTION** | `FALSE` | Frozen, promotion disabled |

---

## 2. Stage 1 — Resolve Canonical Main

- **Target Branch**: `main`
- **Upstream Sync**: Fast-forward pull from `origin/main` complete
- **POST_MERGE_MAIN_SHA**: `7183459e2ce3bf7151db79cce1b862be0924a38d`
- **Merge Provenance**:
  - PR #3 Merge Commit: `544f13b36ec116d11be1d7fa20617a96c1c87e76`
  - PR #4 Merge Commit: `7183459e2ce3bf7151db79cce1b862be0924a38d`
- **Working Tree**: Clean (`git status --short` verified)

---

## 3. Stage 2 — Full Repository Verification

Full test suite execution on commit `7183459e2ce3bf7151db79cce1b862be0924a38d`:

- **Collected**: 568 tests
- **Passed**: 566 tests
- **Failed**: 2 tests (detailed below)
- **Skipped**: 0 tests
- **Duration**: 287.40s (~4m47s)

### Test Failures Analysis:
1. `tests/test_factor_research.py::test_production_factor_manifest_hash_matches_physical_file`:
   - `AssertionError`: Physical SHA `81c75a65ff94...` != manifest `ad672d9cf585...`
2. `tests/test_production_hardening.py::TestManifestIntegrityVerification::test_factor_matrix_manifest_hash_matches_parquet`:
   - `AssertionError`: `ad672d9cf585...` != `81c75a65ff94...`

*Root Cause*: In GitHub Actions CI (clean checkout), large `.parquet` datasets (>100MB) are excluded by `.gitignore`, bypassing physical presence assertions. On the host workstation, the physical file `factor_matrix_300.parquet` (312MB) exists with SHA `81c75a65ff...`, which diverges from `factor_matrix_300.manifest.json` (`ad672d9cf5...`) and canonical run pointer (`9a882c4568...`).

### External Network Tests:
- `tests/test_corporate_actions.py::TestCrossSourceConsistency::test_cninfo_ex_dates_match_hfq_factor_events`:
  - Result: **PASSED** (External CNINFO endpoint was reachable; valid assertions verified).
  - Policy Reminder: In accordance with task directives, if an external endpoint is unreachable and SKIPPED, it is recorded as `EXTERNAL_EVIDENCE_UNAVAILABLE` and strictly prohibited from certifying a PASS.

---

## 4. Stage 3 — Provenance Recalculation

| Provenance Element | Observed Value / Hash | Verification Status | Verdict |
| :--- | :--- | :---: | :--- |
| **Canonical Main SHA** | `7183459e2ce3bf7151db79cce1b862be0924a38d` | Verified on disk and origin | Clean |
| **Run Pointer Frozen SHA** | `8dbf06213b9af0a6614f377513dc997aa80266be` | Mismatch against main HEAD | `STALE_EVIDENCE` |
| **Market Panel Parquet** | `47dc1429f20acfbd254c3c23f172c80f5d87852fa04bb4d6f60579af310033be` | Matches manifest `file_sha256` | Verified |
| **Factor Matrix Parquet** | Physical: `81c75a65ff...`<br>Manifest: `ad672d9cf5...`<br>Run Pointer: `9a882c4568...` | Triple Divergence | `INVALID_EVIDENCE` |
| **Dataset Lineage** | `market_daily_300.parquet` stores normalized `LOG_CIRC_MV` (mean=0/std=1) | Prohibits deterministic rebuild | `LINEAGE_POLLUTED` |
| **Model Registry** | 35 tracked models; active production: `m_20260903_194757_hybrid_bagging_ridge` | Verified clean schema | Verified |
| **Artifact Hash Chain** | 18 files in `research_8dbf062_20260831_155701` match `artifact_manifest.json` | Re-hashed 100% | Verified |
| **Working Tree Cleanliness** | No uncommitted files or modifications | Verified | Clean |

---

## 5. Stage 4 — Runtime Attestation

- **Distinction**: `CAPABILITY_REPORT.md` (static codebase capability proved by automated test fixtures) is strictly separated from `RUNTIME_ATTESTATION.md` (real execution facts).
- **Execution**: Evaluated via `tools/generate_audit_report.py` against active working tree:
  - Runtime Instance: `STALE_COMMIT_MISMATCH_run_b7c7f054`
  - Code Commit SHA: `c9e45e1586135fe28fe01876dbbecba031bd3667`
  - Current Git HEAD: `7183459e2ce3bf7151db79cce1b862be0924a38d`
  - External Trust Root: `UNPINNED_SELF_TRUST` / `DEV_UNTRUSTED_KEY`
  - Overall Rating: **`HIGH_RISK`** (Fail-Closed triggered by `runtime_commit_mismatch` and unpinned trust root).

---

## 6. Stage 5 — Infrastructure Gap Diagnosis

Detailed in [INFRASTRUCTURE_EVIDENCE_GAP_MATRIX.json](./INFRASTRUCTURE_EVIDENCE_GAP_MATRIX.json):

1. **`STRICT_FUNDAMENTAL_PIT` (`INSUFFICIENT_EVIDENCE`)**:
   - Required: Official PIT announcement chronology independently recorded (`official_announcement_rows > 0`, zero synthetic delay).
   - Observed: `official_announcement_rows == 0`. Raw quarterly earnings tables exist, but lack an independent, timestamped announcement chronology pipeline.
2. **`CANONICAL_CALENDAR_PROVENANCE` (`INSUFFICIENT_EVIDENCE`)**:
   - Required: Certified non-synthetic exchange calendar source, `calendar_artifact_sha256`, `run_mode == "certified"`.
   - Observed: Historical artifact lacks hardened schema fields; runtime relies on unauthenticated third-party Akshare/Sina calendar.
3. **`SOURCE_CODE_PROVENANCE` (`INSUFFICIENT_EVIDENCE`)**:
   - Required: Runtime git SHA equals artifact source code SHA equals canonical main SHA (`7183459e`).
   - Observed: Historical canonical run is bound to commit `8dbf0621`; runtime audit is bound to `c9e45e15`.
4. **`PRODUCTION_FACTOR_MATRIX_LINEAGE` (`FAIL` / `INSUFFICIENT_EVIDENCE`)**:
   - Required: Deterministic bit-level factor matrix reproduction; consistent manifest hash.
   - Observed: Physical file SHA (`81c75a65ff...`) diverges from manifest (`ad672d9cf5...`) and canonical pointer (`9a882c4568...`). Market data contains normalized values rather than raw capitalization.
5. **`WALKFORWARD_PURGE_GATE` & `QUANTILE_EVALUATION_INTEGRITY` (`INSUFFICIENT_EVIDENCE`)**:
   - Required: Hardened dual purge keys (`configured_purge_gap_trading_days`, `label_horizon_trading_days`) and runtime `daily_group_counts`.
   - Observed: Historical evidence artifact schema lacks these hardened verification fields.

---

## 7. Stage 6 — Fail-Closed Decision

Under strict governance rules:
- Any missing, stale, or unverified critical evidence mandates:
  $$	ext{INFRASTRUCTURE\_STATUS} = 	ext{INSUFFICIENT\_EVIDENCE}$$
- Forcible certification or overriding evidence gaps to achieve `PASS` is strictly prohibited.

---

## 8. Stage 7 — Final Run Pointer

- **Current File**: [reports/audit_hardening_v3/FINAL_RUN_POINTER.json](file:///C:/Users/lin/Documents/股票预测/reports/audit_hardening_v3/FINAL_RUN_POINTER.json)
- **Pointed Run**: `research_8dbf062_20260831_155701`
- **Validation Results**:
  - Validated against frozen commit `8dbf06213b9af0a6614f377513dc997aa80266be`:
    ```python
    VALID: True, ISSUE: ''
    ```
  - Validated against post-merge main commit `7183459e2ce3bf7151db79cce1b862be0924a38d`:
    ```python
    VALID: False, ISSUE: 'FINAL_RUN_POINTER code freeze mismatch'
    ```
- **New Run Pointer Creation**: Because no new canonical experiment was run (strictly prohibited by task constraints), no fabricated pointer was created. The pointer remains genuinely bound to the frozen run.

---

## 9. Final Determination & Minimum Viable Remediation

### Final Determination:
# **`INFRASTRUCTURE_EVIDENCE_STILL_INSUFFICIENT`**

### Next Minimum Viable Remediation Steps:
1. **Fix Market Dataset Lineage**: Update data ingestion pipeline to preserve raw, unstandardized circulating market capitalization (`circ_mv`), ensuring downstream factor neutralization can be deterministically reproduced.
2. **Deterministic Factor Matrix v2**: Rebuild `factor_matrix_300_v2` with locked configuration recipe and establish bit-level reproducible manifest hash.
3. **Official PIT Announcement Timeline**: Integrate exchange-certified announcement chronology for quarterly earnings (`yjbb`) so that `official_announcement_rows > 0`.
4. **Certified Research Run on Dedicated Code Freeze**: Execute the formal research runner on a dedicated code freeze commit with registered signing keys, populating all 32 hardened gates.
