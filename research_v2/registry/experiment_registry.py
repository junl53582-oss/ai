"""
V2 Experiment Registry (research_v2/registry/experiment_registry.py)
严格执行“一次实验只改一个核心变量”的科学投研规范，管理全生命周期实验。
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from .schemas import ExperimentRecord
from .baseline_registry import BaselineRegistry

logger = logging.getLogger("experiment_registry")


class ExperimentRegistry:
    """科学实验注册中枢"""
    def __init__(self, experiments_dir: Optional[Path] = None, baseline_registry: Optional[BaselineRegistry] = None):
        if experiments_dir is None:
            from config.settings import settings
            experiments_dir = settings.REPORTS_DIR / "experiments"
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_registry = baseline_registry or BaselineRegistry()
        self._experiments: Dict[str, ExperimentRecord] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        for exp_file in self.experiments_dir.glob("*/experiment_manifest.json"):
            try:
                data = json.loads(exp_file.read_text(encoding="utf-8"))
                rec = ExperimentRecord(**data)
                self._experiments[rec.experiment_id] = rec
            except Exception as e:
                logger.warning(f"Failed to load experiment {exp_file}: {e}")

    def register_experiment(self, record: ExperimentRecord, save_to_disk: bool = True) -> None:
        """注册新实验，严格校验唯一性、对照基准与单自变量设计"""
        record.validate()
        if record.experiment_id in self._experiments:
            raise ValueError(f"Duplicate experiment_id '{record.experiment_id}' rejected.")

        # 检查对照基准有效性
        try:
            self.baseline_registry.get(record.parent_baseline_id)
        except KeyError:
            raise ValueError(f"Parent baseline '{record.parent_baseline_id}' does not exist in BaselineRegistry.")

        self._experiments[record.experiment_id] = record

        if save_to_disk:
            exp_dir = self.experiments_dir / record.experiment_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = exp_dir / "experiment_manifest.json"
            manifest_path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        if experiment_id not in self._experiments:
            self._load_from_disk()
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment '{experiment_id}' not found.")
        return self._experiments[experiment_id]

    def list_experiments(self) -> List[ExperimentRecord]:
        self._load_from_disk()
        return list(self._experiments.values())
