"""
Tests for Legacy Baseline V1 Freeze (tests/test_phase2_1_legacy_freeze.py)
"""
import json
import hashlib
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from research_v2.registry.baseline_registry import (
    BaselineRegistry,
    BaselineIntegrityError,
    LEGACY_V1_SEAL_COMMIT,
    legacy_v1_hash_matches,
)


def test_legacy_v1_baseline_artifacts_exist():
    base_dir = Path("reports/baselines/legacy_v1")
    assert base_dir.exists()
    required_files = [
        "baseline_manifest.json",
        "legacy_model_semantics.json",
        "model_comparison.csv",
        "trading_fold_stability.csv",
        "seed_robustness.csv",
        "artifact_hashes.json",
        "freeze_evidence.json",
        "LEGACY_BASELINE_REPORT.md",
    ]
    for rf in required_files:
        p = base_dir / rf
        assert p.exists(), f"Missing legacy baseline artifact: {p}"


def test_3_tier_git_provenance_commits_exist():
    manifest_file = Path("reports/baselines/legacy_v1/baseline_manifest.json")
    data = json.loads(manifest_file.read_text(encoding="utf-8"))

    evidence_c = data["model_evidence_source_commit"]
    logic_c = data["certification_logic_source_commit"]
    art_c = data["certified_artifact_commit"]

    for c in [evidence_c, logic_c, art_c, LEGACY_V1_SEAL_COMMIT]:
        assert len(c) == 40
        res = subprocess.run(
            ["git", "cat-file", "-e", f"{c}^{{commit}}"],
            capture_output=True,
        )
        assert res.returncode == 0, (
            f"Commit {c} does not exist in local git history"
        )

    assert evidence_c != logic_c
    assert logic_c != art_c


def test_legacy_metrics_derived_from_source_artifacts():
    manifest_file = Path("reports/baselines/legacy_v1/baseline_manifest.json")
    data = json.loads(manifest_file.read_text(encoding="utf-8"))

    src_comp = pd.read_csv(
        "reports/model_research/model_comparison_certified.csv",
        keep_default_na=False,
    )
    src_folds = pd.read_csv(
        "reports/model_research/trading_fold_stability_verified.csv"
    )

    clf_src = src_comp[
        src_comp["model_id"] == "lightgbm_clf_baseline"
    ].iloc[0]
    ranker_src = src_comp[
        src_comp["model_id"] == "lightgbm_ranker"
    ].iloc[0]

    p = data["prediction_baseline"]
    assert float(p["mean_daily_rank_ic"]) == pytest.approx(
        float(clf_src["mean_daily_rank_ic"])
    )
    assert float(p["nw20_rank_icir"]) == pytest.approx(
        float(clf_src["rank_icir_nw_lag20"])
    )
    assert float(p["q5_minus_q1_spread"]) == pytest.approx(
        float(clf_src["q5_minus_q1_spread"])
    )
    assert int(p["common_ranking_rows"]) == int(
        clf_src["common_ranking_rows"]
    )
    assert int(p["common_oos_dates"]) == int(clf_src["common_oos_dates"])

    t = data["trading_candidate"]
    assert float(t["cost_adjusted_excess_return"]) == pytest.approx(
        float(ranker_src["cost_adjusted_excess_return"])
    )
    assert float(t["sharpe_ratio"]) == pytest.approx(
        float(ranker_src["sharpe_ratio"])
    )
    assert float(t["max_drawdown"]) == pytest.approx(
        float(ranker_src["max_drawdown"])
    )
    assert float(t["real_fold_win_ratio"]) == pytest.approx(
        float(src_folds["ranker_win"].mean()),
        abs=1e-4,
    )
    assert float(t["annualized_turnover_avg"]) == pytest.approx(
        float(src_folds["ranker_annualized_turnover"].mean()),
        abs=1e-2,
    )


def test_legacy_ranker_semantics_corrected():
    sem_file = Path(
        "reports/baselines/legacy_v1/legacy_model_semantics.json"
    )
    assert sem_file.exists()
    data = json.loads(sem_file.read_text(encoding="utf-8"))
    ranker_sem = data["models"]["trading_candidate"]

    assert ranker_sem["historical_artifact_model_id"] == "lightgbm_ranker"
    assert ranker_sem["legacy_model_id"] == "legacy_ordinal_ranker"
    assert ranker_sem["effective_estimator_class"] == "LGBMRanker"
    assert ranker_sem["effective_objective"] == "regression"
    assert ranker_sem["effective_metric"] == "rmse"
    assert ranker_sem["ranking_group_supplied"] is True
    assert ranker_sem["relevance_label"] == "daily_ordinal_0_to_4"
    assert ranker_sem["true_lambdarank_certified"] is False
    assert data["true_lambdarank_certified"] is False


def test_freeze_evidence_json_validity():
    ev_file = Path("reports/baselines/legacy_v1/freeze_evidence.json")
    assert ev_file.exists()
    ev_data = json.loads(ev_file.read_text(encoding="utf-8"))
    assert "source_artifacts" in ev_data
    assert "derived_metrics" in ev_data
    assert "prediction_baseline" in ev_data["derived_metrics"]
    assert "trading_candidate" in ev_data["derived_metrics"]


def test_legacy_git_seal_accepts_unmodified_copy(tmp_path):
    src_dir = Path("reports/baselines")
    dst_dir = tmp_path / "baselines"
    shutil.copytree(src_dir, dst_dir)

    reg = BaselineRegistry(
        baselines_dir=dst_dir,
        project_root=Path.cwd(),
    )
    assert reg.verify_integrity("LEGACY_BASELINE_V1") is True


def test_legacy_manifest_tamper_is_blocked_by_git_seal(tmp_path):
    """
    baseline_manifest.json 原本不在 artifact_hashes 的 artifacts 映射中。
    旧实现可被直接改写某些字段而不触发逐产物 Hash；Git seal 必须补上该根信任缺口。
    """
    src_dir = Path("reports/baselines")
    dst_dir = tmp_path / "baselines"
    shutil.copytree(src_dir, dst_dir)

    manifest_path = dst_dir / "legacy_v1" / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["live_trading_ready"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    reg = BaselineRegistry(
        baselines_dir=dst_dir,
        project_root=Path.cwd(),
    )
    with pytest.raises(BaselineIntegrityError, match="Git seal mismatch"):
        reg.verify_integrity("LEGACY_BASELINE_V1")


def test_legacy_hash_is_cross_platform_for_line_endings(tmp_path):
    path = tmp_path / "artifact.csv"
    path.write_bytes(b"\xef\xbb\xbfcol\n1\n2\n")

    frozen_windows_bytes = b"\xef\xbb\xbfcol\r\n1\r\n2\r\n"
    expected = hashlib.sha256(frozen_windows_bytes).hexdigest()

    assert legacy_v1_hash_matches(path, expected) is True
