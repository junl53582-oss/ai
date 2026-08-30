"""
V2 Baseline Registry (research_v2/registry/baseline_registry.py)
管理不可篡改的科学基准参照系 (如 LEGACY_BASELINE_V1)，具备文件 Hash 校验与固定 Git seal 防篡改验证。
"""
import json
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from .schemas import (
    BaselineRecord,
    ComparisonVsBaseline,
    BaselineIntegrityError,
    MissingExperimentMetricError,
    require_metric,
)

logger = logging.getLogger("baseline_registry")


# LEGACY_BASELINE_V1 首次完整 seal 的不可变 Git commit。
# 该 commit 已包含当前冻结 baseline_manifest / artifact_hashes / CSV / semantics / evidence / report。
# 后续若要变更基准，必须创建新的 baseline version，而不是修改 V1。
LEGACY_V1_SEAL_COMMIT = "d30dbe881463a6ac0e677cb17818d67d70ada628"
LEGACY_V1_SEALED_FILES = (
    "baseline_manifest.json",
    "artifact_hashes.json",
    "model_comparison.csv",
    "trading_fold_stability.csv",
    "seed_robustness.csv",
    "legacy_model_semantics.json",
    "freeze_evidence.json",
    "LEGACY_BASELINE_REPORT.md",
)


def compute_file_sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_crlf_bytes(data: bytes) -> bytes:
    """
    将 UTF-8 文本规范化为 Windows CRLF，同时保留是否存在 UTF-8 BOM。

    LEGACY_BASELINE_V1 的 Hash 在 Windows 上冻结，Git 存储会把文本行尾规范成 LF。
    因此 Linux CI 必须能复现“冻结时字节语义”，否则同一 Git 内容会被误判为篡改。
    """
    has_bom = data.startswith(b"\xef\xbb\xbf")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BaselineIntegrityError(
            "Legacy baseline text artifact is not valid UTF-8"
        ) from exc
    normalized_lf = text.replace("\r\n", "\n").replace("\r", "\n")
    crlf_text = normalized_lf.replace("\n", "\r\n")
    encoded = crlf_text.encode("utf-8")
    return (b"\xef\xbb\xbf" + encoded) if has_bom else encoded


def legacy_v1_hash_matches(path: Path, expected_sha256: str) -> bool:
    """
    LEGACY_BASELINE_V1 兼容 Hash 校验。

    先按工作区原始字节校验；若不匹配，再按冻结时 Windows CRLF 规范复算。
    只放宽“行尾表示”这一项，不放宽任何文本内容差异。
    """
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected_sha256:
        return True
    canonical = _canonical_crlf_bytes(data)
    return hashlib.sha256(canonical).hexdigest() == expected_sha256


def _normalized_text_sha256(data: bytes) -> str:
    """
    计算跨平台稳定文本 Hash：
    - 去除 UTF-8 BOM
    - CRLF/CR 统一为 LF

    仅用于 Git seal 的“语义等价文本”比较，避免 Windows checkout 的 autocrlf
    造成误报；artifact_hashes.json 的逐文件原始字节 Hash 仍保持不变。
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BaselineIntegrityError("Sealed baseline file is not valid UTF-8 text") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class BaselineRegistry:
    """不可篡改基准注册表（磁盘 Hash + Git seal 双层完整性验证）。"""

    def __init__(
        self,
        baselines_dir: Optional[Path] = None,
        project_root: Optional[Path] = None,
    ):
        if baselines_dir is None:
            from config.settings import settings

            baselines_dir = settings.REPORTS_DIR / "baselines"
        self.baselines_dir = Path(baselines_dir)
        self.project_root = (
            Path(project_root)
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self._cache: Dict[str, BaselineRecord] = {}
        self._load_builtins()

    def _load_builtins(self):
        legacy_dir = self.baselines_dir / "legacy_v1"
        manifest_file = legacy_dir / "baseline_manifest.json"
        hashes_file = legacy_dir / "artifact_hashes.json"

        if manifest_file.exists() and hashes_file.exists():
            try:
                manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                hashes_data = json.loads(hashes_file.read_text(encoding="utf-8"))
                artifacts_dict = hashes_data.get("artifacts", {})
                record = BaselineRecord(
                    baseline_id=manifest_data["baseline_id"],
                    baseline_status=manifest_data["baseline_status"],
                    created_at=manifest_data["created_at"],
                    model_evidence_source_commit=manifest_data[
                        "model_evidence_source_commit"
                    ],
                    certification_logic_source_commit=manifest_data.get(
                        "certification_logic_source_commit", ""
                    ),
                    certified_artifact_commit=manifest_data.get(
                        "certified_artifact_commit", ""
                    ),
                    dataset_sha256=manifest_data["dataset_sha256"],
                    feature_schema_hash=manifest_data["feature_schema_hash"],
                    prediction_baseline=manifest_data["prediction_baseline"],
                    trading_candidate=manifest_data["trading_candidate"],
                    legacy_label_semantics=manifest_data["legacy_label_semantics"],
                    artifact_hash_manifest_sha256=manifest_data.get(
                        "artifact_hash_manifest_sha256", ""
                    ),
                    artifact_hashes=artifacts_dict,
                    immutable=True,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise BaselineIntegrityError(
                    f"Malformed LEGACY_BASELINE_V1 metadata under {legacy_dir}"
                ) from exc
            self._cache["LEGACY_BASELINE_V1"] = record

    def _verify_legacy_git_seal(self, legacy_dir: Path) -> None:
        """
        将工作区中的 LEGACY_BASELINE_V1 与固定 seal commit 的 Git blob 比较。

        这一步补上原先“manifest + hashes 可被同时重写后重新自洽”的根信任缺口。
        只要 V1 任一 sealed 文件发生实质变化，即使攻击者同步重算内部 Hash，也会失败。
        """
        commit_check = subprocess.run(
            ["git", "cat-file", "-e", f"{LEGACY_V1_SEAL_COMMIT}^{{commit}}"],
            cwd=self.project_root,
            capture_output=True,
        )
        if commit_check.returncode != 0:
            raise BaselineIntegrityError(
                f"Legacy baseline seal commit is unavailable: {LEGACY_V1_SEAL_COMMIT}"
            )

        for filename in LEGACY_V1_SEALED_FILES:
            disk_path = legacy_dir / filename
            if not disk_path.exists():
                raise BaselineIntegrityError(
                    f"Sealed baseline file missing on disk: {disk_path}"
                )

            repo_path = f"reports/baselines/legacy_v1/{filename}"
            res = subprocess.run(
                ["git", "show", f"{LEGACY_V1_SEAL_COMMIT}:{repo_path}"],
                cwd=self.project_root,
                capture_output=True,
            )
            if res.returncode != 0:
                stderr = res.stderr.decode("utf-8", errors="replace").strip()
                raise BaselineIntegrityError(
                    f"Unable to read sealed Git blob {LEGACY_V1_SEAL_COMMIT}:{repo_path}: "
                    f"{stderr or 'git show failed'}"
                )

            disk_digest = _normalized_text_sha256(disk_path.read_bytes())
            sealed_digest = _normalized_text_sha256(res.stdout)
            if disk_digest != sealed_digest:
                raise BaselineIntegrityError(
                    f"Git seal mismatch for '{filename}'. "
                    f"LEGACY_BASELINE_V1 is immutable; create a new baseline version "
                    f"instead of modifying sealed V1."
                )

    def verify_integrity(self, baseline_id: str = "LEGACY_BASELINE_V1") -> bool:
        """
        全量检验基准文件防伪完整性：
        0. 对 LEGACY_BASELINE_V1 验证固定 Git seal（根信任）
        1. 验证 artifact_hashes.json SHA256 与 manifest 中记录一致
        2. 验证每个冻结产物文件在磁盘上的真实 SHA256 与 artifact_hashes.json 一致
        """
        legacy_dir = (
            self.baselines_dir / "legacy_v1"
            if baseline_id == "LEGACY_BASELINE_V1"
            else self.baselines_dir / baseline_id.lower()
        )
        manifest_file = legacy_dir / "baseline_manifest.json"
        hashes_file = legacy_dir / "artifact_hashes.json"

        if not manifest_file.exists():
            raise BaselineIntegrityError(
                f"Baseline manifest file not found: {manifest_file}"
            )
        if not hashes_file.exists():
            raise BaselineIntegrityError(
                f"Artifact hashes manifest not found: {hashes_file}"
            )

        if baseline_id == "LEGACY_BASELINE_V1":
            self._verify_legacy_git_seal(legacy_dir)

        try:
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            hashes_data = json.loads(hashes_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BaselineIntegrityError(
                f"Invalid JSON in baseline integrity metadata: {exc}"
            ) from exc

        if manifest_data.get("baseline_id") != baseline_id:
            raise BaselineIntegrityError(
                f"Baseline ID mismatch: requested {baseline_id}, "
                f"manifest contains {manifest_data.get('baseline_id')!r}"
            )
        if hashes_data.get("baseline_id") != baseline_id:
            raise BaselineIntegrityError(
                f"Artifact hash baseline ID mismatch: requested {baseline_id}, "
                f"hash manifest contains {hashes_data.get('baseline_id')!r}"
            )

        expected_manifest_hash = manifest_data.get("artifact_hash_manifest_sha256")
        if not expected_manifest_hash:
            raise BaselineIntegrityError(
                "baseline_manifest.json missing 'artifact_hash_manifest_sha256'"
            )

        actual_hashes_file_sha = compute_file_sha256(hashes_file)
        if baseline_id == "LEGACY_BASELINE_V1":
            hashes_match = legacy_v1_hash_matches(
                hashes_file, expected_manifest_hash
            )
        else:
            hashes_match = actual_hashes_file_sha == expected_manifest_hash

        if not hashes_match:
            raise BaselineIntegrityError(
                "artifact_hashes.json tamper detected! "
                f"Expected: {expected_manifest_hash}, "
                f"Actual raw SHA256: {actual_hashes_file_sha}"
            )

        artifacts = hashes_data.get("artifacts", {})
        if not isinstance(artifacts, dict) or not artifacts:
            raise BaselineIntegrityError(
                "artifact_hashes.json must contain a non-empty 'artifacts' mapping"
            )

        for filename, expected_sha in artifacts.items():
            if not isinstance(filename, str) or not filename:
                raise BaselineIntegrityError("Invalid artifact filename in hash manifest")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise BaselineIntegrityError(
                    f"Invalid SHA256 entry for frozen artifact '{filename}'"
                )
            f_path = legacy_dir / filename
            if not f_path.exists():
                raise BaselineIntegrityError(
                    f"Frozen artifact '{filename}' missing on disk: {f_path}"
                )
            actual_sha = compute_file_sha256(f_path)
            if baseline_id == "LEGACY_BASELINE_V1":
                artifact_match = legacy_v1_hash_matches(f_path, expected_sha)
            else:
                artifact_match = actual_sha == expected_sha

            if not artifact_match:
                raise BaselineIntegrityError(
                    f"Tamper detected in artifact '{filename}'! "
                    f"Expected SHA: {expected_sha}, "
                    f"Actual raw SHA256: {actual_sha}"
                )

        return True

    def register(self, record: BaselineRecord) -> None:
        if record.baseline_id in self._cache:
            existing = self._cache[record.baseline_id]
            if existing.immutable:
                raise ValueError(
                    f"Baseline '{record.baseline_id}' is immutable and cannot be overwritten."
                )
        self._cache[record.baseline_id] = record

    def get(self, baseline_id: str, verify_integrity: bool = True) -> BaselineRecord:
        if verify_integrity:
            self.verify_integrity(baseline_id)
        if baseline_id not in self._cache:
            self._load_builtins()
        if baseline_id not in self._cache:
            raise KeyError(f"Baseline '{baseline_id}' not found in registry.")
        return self._cache[baseline_id]

    def compare(
        self,
        candidate_pred: Dict[str, Any],
        candidate_trading: Dict[str, Any],
        baseline_id: str = "LEGACY_BASELINE_V1",
    ) -> ComparisonVsBaseline:
        """
        与指定基准进行因果差异比较 (Fail-Closed, Missing != 0):
        必要指标缺失时返回 NOT_COMPARABLE，禁止以 0 代替缺失值。
        """
        base = self.get(baseline_id, verify_integrity=True)
        b_pred = base.prediction_baseline
        b_trade = base.trading_candidate

        required_pred_keys = ["mean_daily_rank_ic", "nw20_rank_icir"]
        required_trade_keys = [
            "cost_adjusted_excess_return",
            "sharpe_ratio",
            "max_drawdown",
            "fold_win_ratio",
            "annualized_turnover",
        ]

        missing_keys: List[str] = []
        for k in required_pred_keys:
            if k not in candidate_pred or candidate_pred[k] is None:
                missing_keys.append(f"pred.{k}")
        for k in required_trade_keys:
            if k not in candidate_trading or candidate_trading[k] is None:
                missing_keys.append(f"trading.{k}")

        if missing_keys:
            return ComparisonVsBaseline(
                baseline_id=baseline_id,
                comparison_status="NOT_COMPARABLE",
                robust_improvement=None,
                missing_metrics=missing_keys,
            )

        try:
            cand_ic = require_metric(candidate_pred, "mean_daily_rank_ic")
            cand_ir = require_metric(candidate_pred, "nw20_rank_icir")
            cand_exc = require_metric(
                candidate_trading, "cost_adjusted_excess_return"
            )
            cand_sharpe = require_metric(candidate_trading, "sharpe_ratio")
            cand_mdd = require_metric(candidate_trading, "max_drawdown")
            cand_win = require_metric(candidate_trading, "fold_win_ratio")
            cand_to = require_metric(candidate_trading, "annualized_turnover")
        except MissingExperimentMetricError as exc:
            return ComparisonVsBaseline(
                baseline_id=baseline_id,
                comparison_status="NOT_COMPARABLE",
                robust_improvement=None,
                missing_metrics=[str(exc)],
            )

        base_ic = require_metric(b_pred, "mean_daily_rank_ic")
        base_ir = require_metric(b_pred, "nw20_rank_icir")
        base_exc = require_metric(b_trade, "cost_adjusted_excess_return")
        base_sharpe = require_metric(b_trade, "sharpe_ratio")
        base_mdd = require_metric(b_trade, "max_drawdown")
        base_win = (
            require_metric(b_trade, "real_fold_win_ratio")
            if "real_fold_win_ratio" in b_trade
            else require_metric(b_trade, "fold_win_ratio")
        )
        base_to = (
            require_metric(b_trade, "annualized_turnover_avg")
            if "annualized_turnover_avg" in b_trade
            else require_metric(b_trade, "annualized_turnover")
        )

        d_ic = cand_ic - base_ic
        d_ir = cand_ir - base_ir
        d_exc = cand_exc - base_exc
        d_sharpe = cand_sharpe - base_sharpe
        d_mdd = cand_mdd - base_mdd
        d_win = cand_win - base_win
        d_to = cand_to - base_to

        robust = bool(d_ic > 0 and d_exc > 0 and d_sharpe > 0)
        return ComparisonVsBaseline(
            baseline_id=baseline_id,
            comparison_status="COMPARABLE",
            delta_rank_ic=round(d_ic, 5),
            delta_nw20_rankicir=round(d_ir, 4),
            delta_excess_return=round(d_exc, 2),
            delta_sharpe=round(d_sharpe, 2),
            delta_max_drawdown=round(d_mdd, 2),
            delta_fold_win_ratio=round(d_win, 4),
            delta_turnover=round(d_to, 2),
            robust_improvement=robust,
            missing_metrics=[],
        )
