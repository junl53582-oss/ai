"""
Legacy Baseline Validator (tools/validate_legacy_baseline.py)
只读验证不可篡改基准 LEGACY_BASELINE_V1 的完整性、密码学 Hash、固定 Git seal、指标派生自洽性与 Git 历史血缘。
"""
import sys
import json
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from research_v2.registry.baseline_registry import (
    BaselineRegistry,
    compute_file_sha256,
    legacy_v1_hash_matches,
    LEGACY_V1_SEAL_COMMIT,
)


def _fail(message: str) -> bool:
    print(f"[FAIL] {message}")
    return False


def _require_equal(name: str, actual, expected) -> bool:
    if actual != expected:
        print(f"[FAIL] {name}: expected {expected!r}, actual {actual!r}")
        return False
    return True


def validate_legacy_baseline() -> bool:
    baseline_dir = settings.REPORTS_DIR / "baselines" / "legacy_v1"
    manifest_file = baseline_dir / "baseline_manifest.json"
    hashes_file = baseline_dir / "artifact_hashes.json"

    print("==> [1/6] 检查 Manifest 与 Hash 文件存在性...")
    if not manifest_file.exists():
        return _fail(f"Missing manifest file: {manifest_file}")
    if not hashes_file.exists():
        return _fail(f"Missing hashes file: {hashes_file}")

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        json.loads(hashes_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _fail(f"Invalid baseline JSON: {exc}")

    print("==> [2/6] 验证 3 级 Git 血缘与 V1 seal commit 真实存在性...")
    commit_items = [
        ("model_evidence_source_commit", manifest.get("model_evidence_source_commit")),
        (
            "certification_logic_source_commit",
            manifest.get("certification_logic_source_commit"),
        ),
        ("certified_artifact_commit", manifest.get("certified_artifact_commit")),
        ("legacy_v1_seal_commit", LEGACY_V1_SEAL_COMMIT),
    ]
    for commit_key, sha in commit_items:
        if not sha:
            return _fail(f"Missing '{commit_key}'")
        res = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
        )
        if res.returncode != 0:
            return _fail(
                f"Git commit {sha} ({commit_key}) does not exist in local repository"
            )
        print(f"  * {commit_key}: {sha[:7]} [EXISTS]")

    print("==> [3/6] 验证 artifact_hash_manifest_sha256（跨平台 CRLF 兼容）...")
    expected_hashes_sha = manifest.get("artifact_hash_manifest_sha256")
    if not expected_hashes_sha:
        return _fail("Missing artifact_hash_manifest_sha256 in baseline manifest")

    if not legacy_v1_hash_matches(hashes_file, expected_hashes_sha):
        return _fail(
            "Hashes file SHA mismatch: "
            f"expected {expected_hashes_sha}, "
            f"actual raw {compute_file_sha256(hashes_file)}"
        )
    print(
        f"  * Hashes Manifest frozen SHA: "
        f"{expected_hashes_sha[:16]}... [MATCH]"
    )

    print("==> [4/6] 验证 Git seal + 全部冻结产物 Hash...")
    reg = BaselineRegistry(
        baselines_dir=settings.REPORTS_DIR / "baselines",
        project_root=PROJECT_ROOT,
    )
    try:
        reg.verify_integrity("LEGACY_BASELINE_V1")
        print(
            "  * Git seal and all frozen artifact hashes verified [PASS]"
        )
    except Exception as exc:
        return _fail(f"Integrity verification failed: {exc}")

    print("==> [5/6] 验证指标派生一致性与真实语义...")
    src_comp = (
        settings.REPORTS_DIR
        / "model_research"
        / "model_comparison_certified.csv"
    )
    if not src_comp.exists():
        return _fail(
            f"Source certified comparison CSV not found: {src_comp}"
        )

    df_src = pd.read_csv(src_comp, keep_default_na=False)
    clf_rows = df_src[df_src["model_id"] == "lightgbm_clf_baseline"]
    ranker_rows = df_src[df_src["model_id"] == "lightgbm_ranker"]
    if len(clf_rows) != 1:
        return _fail(
            f"Expected exactly one lightgbm_clf_baseline row, got {len(clf_rows)}"
        )
    if len(ranker_rows) != 1:
        return _fail(
            f"Expected exactly one lightgbm_ranker row, got {len(ranker_rows)}"
        )

    clf_src = clf_rows.iloc[0]
    ranker_src = ranker_rows.iloc[0]
    p_base = manifest["prediction_baseline"]
    t_base = manifest["trading_candidate"]

    checks = [
        _require_equal(
            "prediction.mean_daily_rank_ic",
            float(p_base["mean_daily_rank_ic"]),
            float(clf_src["mean_daily_rank_ic"]),
        ),
        _require_equal(
            "prediction.nw20_rank_icir",
            float(p_base["nw20_rank_icir"]),
            float(clf_src["rank_icir_nw_lag20"]),
        ),
        _require_equal(
            "trading.cost_adjusted_excess_return",
            float(t_base["cost_adjusted_excess_return"]),
            float(ranker_src["cost_adjusted_excess_return"]),
        ),
        _require_equal(
            "trading.sharpe_ratio",
            float(t_base["sharpe_ratio"]),
            float(ranker_src["sharpe_ratio"]),
        ),
        _require_equal(
            "trading.legacy_model_id",
            t_base["legacy_model_id"],
            "legacy_ordinal_ranker",
        ),
        _require_equal(
            "trading.effective_objective",
            t_base["effective_objective"],
            "regression",
        ),
        _require_equal(
            "trading_candidate_seed_robustness",
            manifest["trading_candidate_seed_robustness"],
            "NOT_CERTIFIED",
        ),
        _require_equal(
            "live_trading_ready",
            manifest["live_trading_ready"],
            False,
        ),
    ]
    if not all(checks):
        return False
    print(
        "  * Dynamic metrics, semantic corrections, and safety flags verified [PASS]"
    )

    print("==> [6/6] 验证 BaselineRecord 加载与 immutable guard...")
    try:
        base_rec = reg.get("LEGACY_BASELINE_V1", verify_integrity=True)
    except Exception as exc:
        return _fail(f"BaselineRecord load failed: {exc}")

    if base_rec.baseline_id != "LEGACY_BASELINE_V1":
        return _fail(
            f"Unexpected baseline_id: {base_rec.baseline_id}"
        )
    if base_rec.immutable is not True:
        return _fail("BaselineRecord immutable flag is not True")
    print("  * BaselineRecord load and immutability guard verified [PASS]")

    print(
        "\n===> [SUCCESS] LEGACY_BASELINE_V1 完整性、Git seal 与跨平台 Hash 门禁验证通过！"
    )
    return True


if __name__ == "__main__":
    ok = validate_legacy_baseline()
    sys.exit(0 if ok else 1)
