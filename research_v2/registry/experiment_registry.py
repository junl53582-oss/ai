"""
V2 Experiment Registry (research_v2/registry/experiment_registry.py)
严格执行“一次实验只改一个核心变量”的科学投研规范，管理全生命周期实验。
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import ExperimentRecord, ExperimentIntegrityError
from .baseline_registry import BaselineRegistry

logger = logging.getLogger("experiment_registry")


class ExperimentRegistry:
    """科学实验注册中枢（实验 Manifest 读取严格 fail-closed）。"""

    def __init__(
        self,
        experiments_dir: Optional[Path] = None,
        baseline_registry: Optional[BaselineRegistry] = None,
    ):
        if experiments_dir is None:
            from config.settings import settings

            experiments_dir = settings.REPORTS_DIR / "experiments"
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_registry = baseline_registry or BaselineRegistry()
        self._experiments: Dict[str, ExperimentRecord] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        """
        从磁盘加载实验注册记录。

        任何损坏 JSON、字段不完整、目录名与 experiment_id 不一致、或同 ID
        内容冲突都会直接阻断；禁止把坏 Manifest 静默 warning 后跳过。
        """
        for exp_file in sorted(
            self.experiments_dir.glob("*/experiment_manifest.json")
        ):
            try:
                data = json.loads(exp_file.read_text(encoding="utf-8"))
                rec = ExperimentRecord(**data)
                rec.validate()
            except Exception as exc:
                raise ExperimentIntegrityError(
                    f"Invalid experiment manifest: {exp_file}: {exc}"
                ) from exc

            expected_id = exp_file.parent.name
            if rec.experiment_id != expected_id:
                raise ExperimentIntegrityError(
                    f"Experiment path/id mismatch: directory '{expected_id}' "
                    f"contains experiment_id '{rec.experiment_id}'"
                )

            existing = self._experiments.get(rec.experiment_id)
            if existing is not None and existing.to_dict() != rec.to_dict():
                raise ExperimentIntegrityError(
                    f"Conflicting experiment records found for "
                    f"'{rec.experiment_id}'"
                )

            self._experiments[rec.experiment_id] = rec

    def register_experiment(
        self,
        record: ExperimentRecord,
        save_to_disk: bool = True,
    ) -> None:
        """注册新实验，严格校验唯一性、对照基准与单自变量设计。"""
        record.validate()
        if record.experiment_id in self._experiments:
            raise ValueError(
                f"Duplicate experiment_id '{record.experiment_id}' rejected."
            )

        # 检查对照基准有效性（包含 Baseline 完整性校验）
        try:
            self.baseline_registry.get(record.parent_baseline_id)
        except KeyError as exc:
            raise ValueError(
                f"Parent baseline '{record.parent_baseline_id}' "
                f"does not exist in BaselineRegistry."
            ) from exc

        if save_to_disk:
            exp_dir = self.experiments_dir / record.experiment_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = exp_dir / "experiment_manifest.json"

            # 如果磁盘上已有同名文件而内存未加载到，禁止覆盖。
            if manifest_path.exists():
                raise ExperimentIntegrityError(
                    f"Experiment manifest already exists on disk: {manifest_path}"
                )

            payload = json.dumps(
                record.to_dict(),
                indent=2,
                ensure_ascii=False,
            )
            tmp_path = exp_dir / ".experiment_manifest.json.tmp"
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(manifest_path)

        self._experiments[record.experiment_id] = record

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        if experiment_id not in self._experiments:
            self._load_from_disk()
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment '{experiment_id}' not found.")
        return self._experiments[experiment_id]

    def list_experiments(self) -> List[ExperimentRecord]:
        self._load_from_disk()
        return list(self._experiments.values())
