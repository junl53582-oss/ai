"""Fail-closed research evidence certification."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

SEED_RANKIC_STD_MAX = 0.0050
FIXED_SEEDS = (42, 100, 2024)

@dataclass
class CertificationDecision:
    gate_id: str
    status: str
    passed: bool
    condition: str
    threshold: Any
    actual_value: Any
    reason: str
    evidence_artifacts: List[str] = field(default_factory=list)
    evidence_sha256: Dict[str, str] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def _artifact(root: Optional[Path], name: str):
    if root is None: return None, {}, "evidence_dir is required"
    path = Path(root) / name
    if not Path(root).is_dir(): return None, {}, "evidence_dir does not exist"
    if not path.is_file(): return None, {}, f"required artifact missing: {name}"
    return path, {name: _sha256(path)}, ""

def _json(root: Optional[Path], name: str):
    path, sha, issue = _artifact(root, name)
    if path is None: return None, sha, issue
    try: return json.loads(path.read_text(encoding="utf-8")), sha, ""
    except (OSError, json.JSONDecodeError) as exc: return None, sha, f"invalid JSON evidence {name}: {exc}"

def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))

def _gate(gate_id, passed, condition, threshold, actual, reason, artifact, sha, failed_status="FAIL"):
    return CertificationDecision(gate_id, "PASS" if passed else failed_status, passed, condition, threshold, actual, reason, [artifact] if sha else [], sha)

def validate_artifact_hash_chain(root: Path, expected: Dict[str, Any]) -> tuple[bool, str]:
    manifest, _, issue = _json(root, "artifact_manifest.json")
    if issue or not isinstance(manifest, dict) or not manifest:
        return False, issue or "empty artifact manifest"
    for rel, entry in manifest.items():
        required = ("sha256", "size_bytes", "generated_by", "source_code_sha", "dataset_sha256", "calendar_sha256", "config_hash", "run_id")
        if not isinstance(entry, dict) or not all(k in entry for k in required): return False, f"invalid manifest entry: {rel}"
        file_path = Path(root) / rel
        if not file_path.is_file(): return False, f"manifest file missing: {rel}"
        if _sha256(file_path) != entry["sha256"]: return False, f"manifest SHA mismatch: {rel}"
        if file_path.stat().st_size != entry["size_bytes"]: return False, f"manifest size mismatch: {rel}"
        for key in ("source_code_sha", "dataset_sha256", "calendar_sha256", "config_hash", "run_id"):
            if entry[key] != expected.get(key): return False, f"manifest provenance mismatch: {rel}:{key}"
    return True, ""

def validate_final_run_pointer(pointer_path: Path, expected_code_freeze_sha: str) -> tuple[bool, str]:
    if not pointer_path.is_file(): return False, "FINAL_RUN_POINTER missing"
    try: pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception as exc: return False, f"invalid FINAL_RUN_POINTER: {exc}"
    required = ("run_id", "code_freeze_sha", "dataset_sha256", "calendar_sha256", "config_hash", "artifact_manifest_sha256", "gate_matrix_sha256", "created_at", "run_status")
    if not all(k in pointer for k in required): return False, "FINAL_RUN_POINTER missing required field"
    if pointer["code_freeze_sha"] != expected_code_freeze_sha: return False, "FINAL_RUN_POINTER code freeze mismatch"
    run_dir = pointer_path.parent / "runs" / pointer["run_id"]
    manifest_path, matrix_path = run_dir / "artifact_manifest.json", run_dir / "audit_gate_matrix.json"
    if not manifest_path.is_file() or not matrix_path.is_file(): return False, "FINAL_RUN_POINTER target missing"
    if _sha256(manifest_path) != pointer["artifact_manifest_sha256"] or _sha256(matrix_path) != pointer["gate_matrix_sha256"]: return False, "FINAL_RUN_POINTER hash mismatch"
    try: manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc: return False, f"invalid target manifest: {exc}"
    if not manifest: return False, "empty target manifest"
    first = next(iter(manifest.values()))
    for key in ("dataset_sha256", "calendar_sha256", "config_hash", "run_id"):
        if pointer[key] != first.get(key): return False, f"FINAL_RUN_POINTER provenance mismatch: {key}"
    return True, ""

def evaluate_research_gates(prod_snap_before: Dict[str, Any], prod_snap_after: Dict[str, Any], prod_file_before_sha: str, prod_file_after_sha: str, fold_stability_df_records: List[Dict[str, Any]], bootstrap_results: Dict[str, Any], seed_results: Dict[str, Any], purge_audits: List[Dict[str, Any]], calendar_meta: Dict[str, Any], pit_meta: Dict[str, Any], feature_meta: Dict[str, Any], quantile_meta: Dict[str, Any], holdout_meta: Dict[str, Any], evidence_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Derive every gate from on-disk evidence. Metadata booleans never certify a gate."""
    gates = []
    trace, sha, issue = _json(evidence_dir, "fold_backtest_provenance.json")
    fields = {"fold_id","test_start","test_end","test_dates_count","candidate_oos_rows","baseline_oos_rows","candidate_prediction_sha256","baseline_prediction_sha256","candidate_backtest_run_id","baseline_backtest_run_id","candidate_equity_sha256","baseline_equity_sha256","candidate_orders_sha256","baseline_orders_sha256","candidate_trade_count","baseline_trade_count","candidate_strategy_return","baseline_strategy_return","candidate_benchmark_return","baseline_benchmark_return","candidate_excess_return","baseline_excess_return","candidate_minus_baseline","engine_config_hash","dataset_sha256","source_code_sha"}
    rows = trace.get("folds", []) if isinstance(trace, dict) else []
    trace_ok = not issue and trace.get("run_mode", "certified") == "certified" and bool(rows) and all(fields <= set(r) for r in rows)
    for gate_id in ("FORMAL_RESEARCH_RUNNER_EXECUTABLE", "REAL_FOLD_BACKTEST_PROVENANCE"):
        gates.append(_gate(gate_id, trace_ok, "complete persisted candidate and baseline fold trace", "all required provenance fields", {"folds":len(rows),"issue":issue}, "Fold trace verified." if trace_ok else issue or "incomplete fold trace", "fold_backtest_provenance.json", sha, "INSUFFICIENT_EVIDENCE"))
    prod, sha, issue = _json(evidence_dir, "production_isolation.json")
    prod_ok = not issue and isinstance(prod,dict) and prod.get("before") == prod.get("after") and prod.get("file_sha_before") == prod.get("file_sha_after") and prod_snap_before == prod_snap_after and prod_file_before_sha == prod_file_after_sha
    gates.append(_gate("PRODUCTION_MODEL_ISOLATION", prod_ok, "production snapshots are byte-identical", "before == after", prod or {"issue":issue}, "Production directory unchanged." if prod_ok else issue or "production snapshot mismatch", "production_isolation.json", sha))
    purge, sha, issue = _json(evidence_dir, "walk_forward_purge_audit.json")
    purge_rows = purge if isinstance(purge,list) else []
    purge_ok = not issue and bool(purge_rows)
    for r in purge_rows:
        keys = ("configured_purge_gap_trading_days", "label_horizon_trading_days", "actual_train_val_gap_trading_days","actual_val_test_gap_trading_days")
        if not all(k in r and _finite(r[k]) for k in keys): purge_ok, issue = False, "missing or non-finite trading-day purge evidence"; break
        required_gap = max(r["configured_purge_gap_trading_days"], r["label_horizon_trading_days"])
        if r["actual_train_val_gap_trading_days"] < required_gap or r["actual_val_test_gap_trading_days"] < required_gap: purge_ok, issue = False, "trading-day purge gap below horizon-required threshold"; break
    gates.append(_gate("WALKFORWARD_PURGE_GATE", purge_ok, "all trading-day gaps meet label horizon and configured purge", "actual gaps >= max(label_horizon, configured_purge)", {"folds":len(purge_rows),"issue":issue}, "Purge evidence verified." if purge_ok else issue, "walk_forward_purge_audit.json", sha, "INSUFFICIENT_EVIDENCE"))
    calendar, sha, issue = _json(evidence_dir, "calendar_metadata.json")
    dates = calendar.get("dates",[]) if isinstance(calendar,dict) else []
    cal_hash = hashlib.sha256("\n".join(dates).encode()).hexdigest() if dates else ""
    cal_ok = (not issue and calendar.get("run_mode") == "certified" and
              bool(calendar.get("calendar_source")) and calendar.get("calendar_source") not in {"DATASET_DERIVED", "SYNTHETIC_TEST_CALENDAR"} and
              bool(calendar.get("calendar_artifact_sha256")) and bool(dates) and dates == sorted(dates) and len(dates) == len(set(dates)) and
              _finite(calendar.get("dataset_overlap_count")) and calendar["dataset_overlap_count"] > 0 and
              calendar.get("calendar_sha256") == cal_hash and bool(calendar.get("source_code_sha")))
    gates.append(_gate("CANONICAL_CALENDAR_PROVENANCE", cal_ok, "unique ascending canonical dates overlap dataset", "valid source, SHA, source SHA and overlap > 0", {"dates":len(dates),"issue":issue}, "Calendar facts verified." if cal_ok else issue or "calendar validation failed", "calendar_metadata.json", sha, "INSUFFICIENT_EVIDENCE"))
    manifest, _, manifest_issue = _json(evidence_dir, "artifact_manifest.json")
    expected = next(iter(manifest.values())) if isinstance(manifest, dict) and manifest else {}
    chain_ok, chain_issue = validate_artifact_hash_chain(Path(evidence_dir), expected) if expected else (False, manifest_issue or "missing manifest")
    gates.append(_gate("ARTIFACT_HASH_CHAIN", chain_ok, "every manifest file and provenance field recomputes", "all entries hash, size and provenance match", {"issue": chain_issue}, "Artifact hash chain verified." if chain_ok else chain_issue, "artifact_manifest.json", _artifact(evidence_dir, "artifact_manifest.json")[1], "INSUFFICIENT_EVIDENCE"))
    try:
        root = Path(evidence_dir).resolve().parents[3]
        runtime_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        run_meta, _, meta_issue = _json(evidence_dir, "runtime_source_provenance.json")
        source_ok = (not meta_issue and isinstance(run_meta, dict) and
                     run_meta.get("worktree_clean_before_run") is True and
                     run_meta.get("runtime_git_sha") == expected.get("source_code_sha") and
                     runtime_sha == expected.get("source_code_sha"))
        source_issue = "" if source_ok else "source SHA mismatch or runtime was not clean before run"
    except Exception:
        source_ok, source_issue = False, "GIT_STATUS_UNAVAILABLE"
    gates.append(_gate("SOURCE_CODE_PROVENANCE", source_ok, "clean runtime git SHA equals artifact source SHA", "clean worktree and CODE_FREEZE_SHA match", {"issue": source_issue}, "Source provenance verified." if source_ok else source_issue, "artifact_manifest.json", _artifact(evidence_dir, "artifact_manifest.json")[1], "INSUFFICIENT_EVIDENCE"))
    pit, sha, issue = _json(evidence_dir, "fundamental_provenance_manifest.json")
    pit_ok = (not issue and isinstance(pit,dict) and pit.get("synthetic_delay_certified_count",1) == 0 and
              pit.get("invalid_chronology_count",1) == 0 and _finite(pit.get("official_announcement_rows")) and
              pit["official_announcement_rows"] > 0 and bool(pit.get("source_code_sha")))
    gates.append(_gate("STRICT_FUNDAMENTAL_PIT", pit_ok, "official PIT chronology is independently recorded", "zero synthetic-certified and chronology violations", pit or {"issue":issue}, "PIT facts verified." if pit_ok else issue or "PIT provenance incomplete", "fundamental_provenance_manifest.json", sha, "INSUFFICIENT_EVIDENCE"))
    quant, sha, issue = _json(evidence_dir, "quantile_evaluation_summary.json")
    expected_groups = quant.get("expected_n_groups") if isinstance(quant, dict) else None
    daily_counts = quant.get("daily_group_counts", {}) if isinstance(quant, dict) else {}
    complete_daily_groups = (isinstance(expected_groups, int) and expected_groups >= 2 and isinstance(daily_counts, dict) and bool(daily_counts) and
                             all(isinstance(counts, dict) and set(counts) == {f"Q{i}" for i in range(1, expected_groups + 1)} and
                                 all(isinstance(v, int) and v > 0 for v in counts.values()) for counts in daily_counts.values()))
    quant_ok = (not issue and isinstance(quant,dict) and quant.get("ranking_method") == "average" and
                quant.get("aggregation_method") == "mean_of_daily_quantile_returns" and
                _finite(quant.get("total_dates")) and _finite(quant.get("valid_quantile_dates")) and
                _finite(quant.get("invalid_quantile_dates")) and _finite(quant.get("invalid_tie_dates")) and
                _finite(quant.get("dates_missing_required_groups")) and complete_daily_groups and
                int(quant["valid_quantile_dates"]) == len(daily_counts) and
                int(quant["total_dates"]) == int(quant["valid_quantile_dates"]) + int(quant["invalid_quantile_dates"]))
    gates.append(_gate("QUANTILE_EVALUATION_INTEGRITY", quant_ok, "tie-preserving ranks and equal-date weighting verified", "average ranking + invariant checks", quant or {"issue":issue}, "Quantile facts verified." if quant_ok else issue or "quantile evidence incomplete", "quantile_evaluation_summary.json", sha, "INSUFFICIENT_EVIDENCE"))
    seed, sha, issue = _json(evidence_dir, "multi_seed_robustness.json")
    each, std = (seed.get("seed_rankic_each",{}), seed.get("seed_rankic_std")) if isinstance(seed,dict) else ({},None)
    seed_ok = not issue and set(map(int,each)) == set(FIXED_SEEDS) and _finite(std) and float(std) <= SEED_RANKIC_STD_MAX
    gates.append(_gate("MULTI_SEED_ROBUSTNESS", seed_ok, "fixed seeds 42, 100, 2024 meet bound", "std <= 0.0050", {"std":std,"seeds":list(each)}, "Multi-seed robustness verified." if seed_ok else issue or "fixed seed stability fails", "multi_seed_robustness.json", sha))
    boot, sha, issue = _json(evidence_dir, "bootstrap_comparison.json")
    b = boot[0] if isinstance(boot,list) and boot else {}
    lower, pair = b.get("bootstrap_ci_95_lower"), b.get("comparison_pair","")
    robust_ok = not issue and "_vs_" in pair and _finite(lower) and lower > 0
    gates.append(_gate("ROBUST_MODEL_IMPROVEMENT", robust_ok, "non-self comparison has positive 95% CI lower bound", "bootstrap_ci_95_lower > 0", {"comparison_pair":pair,"bootstrap_ci_95_lower":lower}, "Robust improvement verified." if robust_ok else issue or "robust improvement not established", "bootstrap_comparison.json", sha, "MIXED_EVIDENCE_NOT_ROBUST"))
    gov, sha, issue = _json(evidence_dir, "governance_manifest.json")
    hold_ok = not issue and isinstance(gov,dict) and gov.get("final_holdout_available") is False and gov.get("historical_oos_status") is True and bool(gov.get("source_code_sha"))
    live_ok = hold_ok and gov.get("live_trading_ready") is False and gov.get("production_model_promotion") is False
    gates.append(_gate("FINAL_HOLDOUT_GOVERNANCE", hold_ok, "historical OOS is not prospective holdout", "final_holdout_available == false", gov or {"issue":issue}, "Holdout governance accurate." if hold_ok else issue or "invalid holdout claim", "governance_manifest.json", sha))
    gates.append(_gate("LIVE_TRADING_GOVERNANCE", live_ok, "live and production promotion are disabled", "both flags == false", gov or {"issue":issue}, "No live/production claim." if live_ok else issue or "invalid live governance claim", "governance_manifest.json", sha))
    by = {g.gate_id:g for g in gates}
    infra_ids = ("FORMAL_RESEARCH_RUNNER_EXECUTABLE","REAL_FOLD_BACKTEST_PROVENANCE","PRODUCTION_MODEL_ISOLATION","WALKFORWARD_PURGE_GATE","CANONICAL_CALENDAR_PROVENANCE","STRICT_FUNDAMENTAL_PIT","QUANTILE_EVALUATION_INTEGRITY","ARTIFACT_HASH_CHAIN","SOURCE_CODE_PROVENANCE")
    infra = "VERIFIED" if all(by[x].passed for x in infra_ids) else "INSUFFICIENT_EVIDENCE"
    model = "ROBUST" if by["MULTI_SEED_ROBUSTNESS"].passed and by["ROBUST_MODEL_IMPROVEMENT"].passed else "MIXED_EVIDENCE_NOT_ROBUST"
    gov = "PASS" if by["FINAL_HOLDOUT_GOVERNANCE"].passed and by["LIVE_TRADING_GOVERNANCE"].passed else "FAIL"
    overall = "VERIFIED" if infra == "VERIFIED" and model == "ROBUST" and gov == "PASS" else ("INFRASTRUCTURE_VERIFIED_MODEL_EVIDENCE_MIXED" if infra == "VERIFIED" and gov == "PASS" else "FAILED")
    return {"INFRASTRUCTURE_STATUS":infra,"MODEL_EVIDENCE_STATUS":model,"GOVERNANCE_STATUS":gov,"OVERALL_RESEARCH_STATUS":overall,"OVERALL_STATUS":overall,"RESEARCH_INTEGRITY_VERIFIED":overall == "VERIFIED","FINAL_HOLDOUT_AVAILABLE":False,"LIVE_TRADING_READY":False,"PRODUCTION_MODEL_PROMOTION":False,"GATES":{g.gate_id:g.to_dict() for g in gates}}
