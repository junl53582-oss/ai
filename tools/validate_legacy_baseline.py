"""
Legacy Baseline Validator (tools/validate_legacy_baseline.py)
只读验证不可篡改基准 LEGACY_BASELINE_V1 的完整性、密码学防伪哈希、指标派生自洽性与 Git 历史血缘。
"""
import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from research_v2.registry.baseline_registry import BaselineRegistry, compute_file_sha256


def validate_legacy_baseline() -> bool:
    baseline_dir = settings.REPORTS_DIR / "baselines" / "legacy_v1"
    manifest_file = baseline_dir / "baseline_manifest.json"
    hashes_file = baseline_dir / "artifact_hashes.json"

    print("==> [1/6] 检查 Manifest 与 Hash 文件存在性...")
    if not manifest_file.exists():
        print(f"[FAIL] Missing manifest file: {manifest_file}")
        return False
    if not hashes_file.exists():
        print(f"[FAIL] Missing hashes file: {hashes_file}")
        return False

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    hashes = json.loads(hashes_file.read_text(encoding="utf-8"))

    print("==> [2/6] 验证 3 级 Git 血缘 Commit 真实存在性...")
    for commit_key in ["model_evidence_source_commit", "certification_logic_source_commit", "certified_artifact_commit"]:
        sha = manifest.get(commit_key)
        if not sha:
            print(f"[FAIL] Missing '{commit_key}' in baseline manifest")
            return False
        res = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=PROJECT_ROOT, capture_output=True)
        if res.returncode != 0:
            print(f"[FAIL] Git commit {sha} ({commit_key}) does not exist in local repository")
            return False
        print(f"  * {commit_key}: {sha[:7]} [EXISTS]")

    print("==> [3/6] 验证 artifact_hash_manifest_sha256 密码学防伪...")
    actual_hashes_sha = compute_file_sha256(hashes_file)
    expected_hashes_sha = manifest.get("artifact_hash_manifest_sha256")
    if actual_hashes_sha != expected_hashes_sha:
        print(f"[FAIL] Hashes file SHA mismatch: expected {expected_hashes_sha}, actual {actual_hashes_sha}")
        return False
    print(f"  * Hashes Manifest SHA: {actual_hashes_sha[:16]}... [MATCH]")

    print("==> [4/6] 验证全部冻结产物文件防伪 Hash 与磁盘一致性...")
    reg = BaselineRegistry(baselines_dir=settings.REPORTS_DIR / "baselines")
    try:
        reg.verify_integrity("LEGACY_BASELINE_V1")
        print("  * All 5 frozen artifacts verified against artifact_hashes.json [PASS]")
    except Exception as e:
        print(f"[FAIL] Integrity verification failed: {e}")
        return False

    print("==> [5/6] 验证指标派生一致性与真实语义...")
    src_comp = settings.REPORTS_DIR / "model_research" / "model_comparison_certified.csv"
    if not src_comp.exists():
        print(f"[FAIL] Source certified comparison CSV not found: {src_comp}")
        return False
    df_src = pd.read_csv(src_comp, keep_default_na=False)
    clf_src = df_src[df_src["model_id"] == "lightgbm_clf_baseline"].iloc[0]
    ranker_src = df_src[df_src["model_id"] == "lightgbm_ranker"].iloc[0]

    p_base = manifest["prediction_baseline"]
    t_base = manifest["trading_candidate"]

    assert float(p_base["mean_daily_rank_ic"]) == float(clf_src["mean_daily_rank_ic"])
    assert float(p_base["nw20_rank_icir"]) == float(clf_src["rank_icir_nw_lag20"])
    assert float(t_base["cost_adjusted_excess_return"]) == float(ranker_src["cost_adjusted_excess_return"])
    assert float(t_base["sharpe_ratio"]) == float(ranker_src["sharpe_ratio"])
    assert t_base["legacy_model_id"] == "legacy_ordinal_ranker"
    assert t_base["effective_objective"] == "regression"
    assert manifest["trading_candidate_seed_robustness"] == "NOT_CERTIFIED"
    assert manifest["live_trading_ready"] is False
    print("  * Dynamic metrics, semantic corrections, and safety flags verified [PASS]")

    print("==> [6/6] 验证 BaselineRecord 加载能力...")
    base_rec = reg.get("LEGACY_BASELINE_V1", verify_integrity=True)
    assert base_rec.baseline_id == "LEGACY_BASELINE_V1"
    assert base_rec.immutable is True
    print("  * BaselineRecord load and immutability guard verified [PASS]")

    print("\n===> [SUCCESS] LEGACY_BASELINE_V1 完整性与防伪门禁 100% 验证通过！")
    return True


if __name__ == "__main__":
    ok = validate_legacy_baseline()
    sys.exit(0 if ok else 1)
