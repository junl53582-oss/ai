"""Fail-closed research evidence certification."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
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
        keys = ("purge_gap_days","actual_train_val_gap_trading_days","actual_val_test_gap_trading_days")
        if not all(k in r and _finite(r[k]) for k in keys): purge_ok, issue = False, "missing or non-finite trading-day purge evidence"; break
        if r["actual_train_val_gap_trading_days"] < r["purge_gap_days"] or r["actual_val_test_gap_trading_days"] < r["purge_gap_days"]: purge_ok, issue = False, "trading-day purge gap below threshold"; break
    gates.append(_gate("WALKFORWARD_PURGE_GATE", purge_ok, "all trading-day gaps meet purge", "actual gaps >= purge_gap_days", {"folds":len(purge_rows),"issue":issue}, "Purge evidence verified." if purge_ok else issue, "walk_forward_purge_audit.json", sha, "INSUFFICIENT_EVIDENCE"))
    calendar, sha, issue = _json(evidence_dir, "calendar_metadata.json")
    dates = calendar.get("dates",[]) if isinstance(calendar,dict) else []
    cal_hash = hashlib.sha256("\n".join(dates).encode()).hexdigest() if dates else ""
    cal_ok = not issue and bool(calendar.get("calendar_source")) and bool(dates) and dates == sorted(dates) and len(dates) == len(set(dates)) and _finite(calendar.get("dataset_overlap_count")) and calendar["dataset_overlap_count"] > 0 and calendar.get("calendar_sha256") == cal_hash and bool(calendar.get("source_code_sha"))
    gates.append(_gate("CANONICAL_CALENDAR_PROVENANCE", cal_ok, "unique ascending canonical dates overlap dataset", "valid source, SHA, source SHA and overlap > 0", {"dates":len(dates),"issue":issue}, "Calendar facts verified." if cal_ok else issue or "calendar validation failed", "calendar_metadata.json", sha, "INSUFFICIENT_EVIDENCE"))
    pit, sha, issue = _json(evidence_dir, "fundamental_provenance_manifest.json")
    pit_ok = (not issue and isinstance(pit,dict) and pit.get("synthetic_delay_certified_count",1) == 0 and
              pit.get("invalid_chronology_count",1) == 0 and _finite(pit.get("official_announcement_rows")) and
              pit["official_announcement_rows"] > 0 and bool(pit.get("source_code_sha")))
    gates.append(_gate("STRICT_FUNDAMENTAL_PIT", pit_ok, "official PIT chronology is independently recorded", "zero synthetic-certified and chronology violations", pit or {"issue":issue}, "PIT facts verified." if pit_ok else issue or "PIT provenance incomplete", "fundamental_provenance_manifest.json", sha, "INSUFFICIENT_EVIDENCE"))
    quant, sha, issue = _json(evidence_dir, "quantile_evaluation_summary.json")
    quant_ok = not issue and isinstance(quant,dict) and quant.get("ranking_method") == "average" and quant.get("daily_equal_weighted") is True and quant.get("all_equal_dates_invalid") is True and quant.get("row_shuffle_invariant") is True
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
    hold, sha, issue = _json(evidence_dir, "holdout_manifest.json")
    hold_ok = not issue and isinstance(hold,dict) and hold.get("FINAL_HOLDOUT_AVAILABLE") is False and hold.get("historical_oos") is True
    gates.append(_gate("FINAL_HOLDOUT_GOVERNANCE", hold_ok, "historical OOS is not prospective holdout", "FINAL_HOLDOUT_AVAILABLE == false", hold or {"issue":issue}, "Holdout governance accurate." if hold_ok else issue or "invalid holdout claim", "holdout_manifest.json", sha))
    gates.append(CertificationDecision("LIVE_TRADING_GOVERNANCE","PASS",True,"live promotion remains disabled","LIVE_TRADING_READY == false",{"LIVE_TRADING_READY":False,"PRODUCTION_MODEL_PROMOTION":False},"No live/production claim.",["holdout_manifest.json"],sha))
    by = {g.gate_id:g for g in gates}
    infra_ids = ("FORMAL_RESEARCH_RUNNER_EXECUTABLE","REAL_FOLD_BACKTEST_PROVENANCE","PRODUCTION_MODEL_ISOLATION","WALKFORWARD_PURGE_GATE","CANONICAL_CALENDAR_PROVENANCE","STRICT_FUNDAMENTAL_PIT","QUANTILE_EVALUATION_INTEGRITY")
    infra = "VERIFIED" if all(by[x].passed for x in infra_ids) else "INSUFFICIENT_EVIDENCE"
    model = "ROBUST" if by["MULTI_SEED_ROBUSTNESS"].passed and by["ROBUST_MODEL_IMPROVEMENT"].passed else "MIXED_EVIDENCE_NOT_ROBUST"
    gov = "PASS" if by["FINAL_HOLDOUT_GOVERNANCE"].passed and by["LIVE_TRADING_GOVERNANCE"].passed else "FAIL"
    overall = "VERIFIED" if infra == "VERIFIED" and model == "ROBUST" and gov == "PASS" else ("INFRASTRUCTURE_VERIFIED_MODEL_EVIDENCE_MIXED" if infra == "VERIFIED" and gov == "PASS" else "FAILED")
    return {"INFRASTRUCTURE_STATUS":infra,"MODEL_EVIDENCE_STATUS":model,"GOVERNANCE_STATUS":gov,"OVERALL_RESEARCH_STATUS":overall,"OVERALL_STATUS":overall,"RESEARCH_INTEGRITY_VERIFIED":overall == "VERIFIED","FINAL_HOLDOUT_AVAILABLE":False,"LIVE_TRADING_READY":False,"PRODUCTION_MODEL_PROMOTION":False,"GATES":{g.gate_id:g.to_dict() for g in gates}}
