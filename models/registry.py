"""
模型注册表与生命周期管理 (models/registry.py)

背景 (Phase A / 2026-09-01 架构审计):
    审计判定 Model Registry 为 NOT_IMPLEMENTED: 研究运行每次覆盖 latest_lightgbm.pkl,
    无版本/无制品哈希/无血统, 且 PRODUCTION 概念不存在 —— 所谓"每日信号"实为
    "每日重跑 20 折研究" (research replay), 无法归因当天信号来自哪个模型版本。

本模块提供:
    1. 研究制品登记 (RESEARCH): 记录制品路径/模型类型/指标/数据集哈希/代码 commit/配置快照
    2. 生命周期状态机: RESEARCH → CANDIDATE → APPROVED → PRODUCTION → ARCHIVED
    3. 晋升门禁 (fail-closed): 每级晋升要求证据 + 审批人, 全流程留痕
        - RESEARCH→CANDIDATE : 需 OOS 指标 + 数据集哈希
        - CANDIDATE→APPROVED : 需 certification 证据引用
        - APPROVED→PRODUCTION: 需人工审批人 + prospective 验证 + paper trading 证据
    4. 单一生产模型查询 (get_production), 供推理路径使用

重要: 研究运行【永远】只能产生 RESEARCH 记录, 绝不能直接写 PRODUCTION。
    制品文件复制进注册表根目录 (settings.MODELS_DIR/registry) 是唯一授权的生产写入路径。

用法:
    registry = ModelRegistry()
    mid = registry.register_research_artifact(artifact_path, model_type="lightgbm",
                                              task_type="classification", metrics={...},
                                              dataset_sha256=..., config_snapshot={...})
    registry.promote(mid, ModelState.CANDIDATE, approver="researcher")
    registry.promote(mid, ModelState.PRODUCTION, approver="linjun", evidence={...})
"""
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class ModelState:
    RESEARCH = "RESEARCH"
    CANDIDATE = "CANDIDATE"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"

    ORDER = [RESEARCH, CANDIDATE, APPROVED, PRODUCTION]

    @classmethod
    def all_states(cls) -> List[str]:
        return [cls.RESEARCH, cls.CANDIDATE, cls.APPROVED, cls.PRODUCTION, cls.ARCHIVED]


# 允许的晋升路径与所需条件
TRANSITIONS: Dict[str, Dict[str, Any]] = {
    (ModelState.RESEARCH, ModelState.CANDIDATE): {
        "requires_metrics": True,
        "requires_dataset": True,
        "desc": "研究制品具备 OOS 指标与数据集血统后可成为候选",
    },
    (ModelState.CANDIDATE, ModelState.APPROVED): {
        "requires_certification": True,
        "desc": "候选模型需通过认证 (certification evidence) 才能获批",
    },
    (ModelState.APPROVED, ModelState.PRODUCTION): {
        "requires_approver": True,
        "requires_prospective": True,
        "requires_paper_trading": True,
        "desc": "生产上线需人工审批 + 前瞻验证 + 模拟盘证据",
    },
    (ModelState.PRODUCTION, ModelState.ARCHIVED): {
        "desc": "生产模型退役归档",
    },
}


@dataclass
class ModelRecord:
    """模型制品元数据记录 (血统 + 生命周期)"""
    model_id: str
    created_at: str
    model_type: str
    task_type: str
    state: str
    source_artifact: str            # 研究制品原始路径
    registry_artifact: Optional[str] = None   # 晋升后注册表内制品路径
    dataset_sha256: Optional[str] = None
    dataset_path: Optional[str] = None
    code_commit: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_count: Optional[int] = None
    notes: str = ""
    promotion_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PromotionError(Exception):
    """晋升校验失败 (fail-closed)"""
    pass


class ModelRegistry:
    """模型注册表: 研究制品登记 + 生命周期晋升 + 生产模型查询"""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else Path(settings.MODELS_DIR) / "registry"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        if self.index_path.exists():
            try:
                index = json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception as e:
                raise RuntimeError(f"模型注册表索引损坏: {self.index_path} ({e})")

        # 兼容全新 clone 仓库: 若索引中无任何生产模型，自动从标准生产目录 saved_models/production/*/metadata.json 发现并装载已上线模型
        has_production = any(r.get("state") == ModelState.PRODUCTION for r in index.values())
        if not has_production:
            prod_root = self.root.parent / "production"
            if prod_root.exists():
                for meta_file in prod_root.glob("*/metadata.json"):
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        mid = meta.get("model_id")
                        if mid and mid not in index:
                            index[mid] = meta
                    except Exception:
                        pass
        return index

    def _save_index(self, index: Dict[str, Dict[str, Any]]):
        self.index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _git_commit() -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return "UNKNOWN"

    # ------------------------------------------------------------ 登记
    def register_research_artifact(
        self,
        artifact_path: Path,
        model_type: str,
        task_type: str = "classification",
        metrics: Optional[Dict[str, Any]] = None,
        dataset_sha256: Optional[str] = None,
        dataset_path: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
        feature_count: Optional[int] = None,
        notes: str = "",
        state: Optional[str] = None,
    ) -> str:
        """
        登记研究制品。默认状态 RESEARCH —— 研究运行【禁止】登记为 PRODUCTION。
        """
        artifact = Path(artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(f"研究制品不存在: {artifact}")
        if state is not None and state != ModelState.RESEARCH:
            raise PromotionError("研究运行只能登记为 RESEARCH 状态, 禁止直接写入更高状态!")

        ts = datetime.now()
        base_id = f"m_{ts.strftime('%Y%m%d_%H%M%S')}_{model_type}"
        # 同一秒内多次注册必须唯一 (否则后一次覆盖前一次记录, 破坏生产唯一性与归档链)
        model_id = base_id
        index = self._load_index()
        dup = 1
        while model_id in index:
            dup += 1
            model_id = f"{base_id}_{dup}"
        record = ModelRecord(
            model_id=model_id,
            created_at=ts.isoformat(timespec="seconds"),
            model_type=model_type,
            task_type=task_type,
            state=ModelState.RESEARCH,
            source_artifact=str(artifact.resolve()),
            dataset_sha256=dataset_sha256,
            dataset_path=dataset_path,
            code_commit=self._git_commit(),
            config_snapshot=config_snapshot or {},
            metrics=dict(metrics or {}),
            feature_count=feature_count,
            notes=notes,
            promotion_history=[{
                "at": ts.isoformat(timespec="seconds"),
                "from": None,
                "to": ModelState.RESEARCH,
                "approver": "research_run",
                "note": "研究运行登记",
            }],
        )
        index[model_id] = record.to_dict()
        self._save_index(index)
        logger.info(f"研究制品已登记: {model_id} ({model_type}/{task_type}) <- {artifact}")
        return model_id

    # ------------------------------------------------------------ 晋升
    def promote(
        self,
        model_id: str,
        to_state: str,
        approver: str,
        evidence: Optional[Dict[str, Any]] = None,
        note: str = "",
    ) -> ModelRecord:
        """
        状态晋升 (fail-closed): 校验不通过直接抛 PromotionError, 绝不静默放行。
        """
        if to_state not in ModelState.all_states():
            raise PromotionError(f"非法目标状态: {to_state}")
        index = self._load_index()
        if model_id not in index:
            raise PromotionError(f"模型不存在于注册表: {model_id}")

        rec = ModelRecord(**index[model_id])
        from_state = rec.state
        key = (from_state, to_state)
        if key not in TRANSITIONS:
            raise PromotionError(f"非法状态跃迁: {from_state} -> {to_state} (仅允许 {list(TRANSITIONS)})")

        rules = TRANSITIONS[key]
        ev = evidence or {}

        if rules.get("requires_metrics") and not rec.metrics:
            raise PromotionError(f"{key} 需要 OOS 指标 (metrics), 当前为空")
        if rules.get("requires_dataset") and not rec.dataset_sha256:
            raise PromotionError(f"{key} 需要数据集哈希 (dataset_sha256), 当前缺失")
        if rules.get("requires_certification") and not ev.get("certification_ref"):
            raise PromotionError(f"{key} 需要认证证据引用 (evidence.certification_ref)")
        if rules.get("requires_approver") and not (approver and approver.strip()):
            raise PromotionError(f"{key} 需要人工审批人 (approver)")
        if rules.get("requires_prospective") and not ev.get("prospective_validation"):
            raise PromotionError(f"{key} 需要前瞻验证证据 (evidence.prospective_validation)")
        if rules.get("requires_paper_trading") and not ev.get("paper_trading"):
            raise PromotionError(f"{key} 需要模拟盘证据 (evidence.paper_trading)")

        # 晋升到 APPROVED/PRODUCTION: 制品复制进注册表 (唯一授权的生产写入路径)
        if to_state in (ModelState.APPROVED, ModelState.PRODUCTION):
            target_dir = self.root / model_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / "model.pkl"
            if not target_file.exists():
                shutil.copy2(rec.source_artifact, target_file)
            rec.registry_artifact = str(target_file)

            # 若晋升到 PRODUCTION，同步写入统一生产制品目录 (saved_models/production/<model_id>)
            if to_state == ModelState.PRODUCTION:
                try:
                    import hashlib
                    prod_dir = self.root.parent / "production" / model_id
                    prod_dir.mkdir(parents=True, exist_ok=True)
                    prod_file = prod_dir / "model.pkl"
                    if not prod_file.exists() or prod_file.stat().st_size != target_file.stat().st_size:
                        shutil.copy2(target_file, prod_file)

                    meta_path = prod_dir / "metadata.json"
                    meta_path.write_text(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

                    f_bytes = prod_file.read_bytes()
                    manifest_data = {
                        "model_id": model_id,
                        "model_type": rec.model_type,
                        "task_type": rec.task_type,
                        "file_name": "model.pkl",
                        "file_sha256": hashlib.sha256(f_bytes).hexdigest(),
                        "file_size_bytes": len(f_bytes),
                        "created_at": rec.created_at,
                        "state": ModelState.PRODUCTION,
                    }
                    (prod_dir / "manifest.json").write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as e:
                    logger.warning(f"同步写入生产制品目录失败 (非致命): {e}")

        # 生产唯一性: 新模型进 PRODUCTION 时, 旧 PRODUCTION 自动归档
        if to_state == ModelState.PRODUCTION:
            for mid, raw in index.items():
                if mid != model_id and raw.get("state") == ModelState.PRODUCTION:
                    old = ModelRecord(**raw)
                    old.state = ModelState.ARCHIVED
                    old.promotion_history.append({
                        "at": datetime.now().isoformat(timespec="seconds"),
                        "from": ModelState.PRODUCTION,
                        "to": ModelState.ARCHIVED,
                        "approver": approver,
                        "note": f"被 {model_id} 取代, 自动归档",
                    })
                    index[mid] = old.to_dict()
                    logger.info(f"原生产模型 {mid} 已自动归档")

        rec.state = to_state
        rec.promotion_history.append({
            "at": datetime.now().isoformat(timespec="seconds"),
            "from": from_state,
            "to": to_state,
            "approver": approver,
            "evidence": ev,
            "note": note or rules.get("desc", ""),
        })
        index[model_id] = rec.to_dict()
        self._save_index(index)
        logger.info(f"模型晋升: {model_id} {from_state} -> {to_state} (审批: {approver})")
        return rec

    # ------------------------------------------------------------ 查询
    def get(self, model_id: str) -> Optional[ModelRecord]:
        index = self._load_index()
        raw = index.get(model_id)
        return ModelRecord(**raw) if raw else None

    def get_production(self) -> Optional[ModelRecord]:
        """当前唯一 PRODUCTION 模型 (推理路径必须用此)"""
        index = self._load_index()
        for raw in index.values():
            if raw.get("state") == ModelState.PRODUCTION:
                return ModelRecord(**raw)
        return None

    def list_records(self, state: Optional[str] = None) -> List[ModelRecord]:
        index = self._load_index()
        recs = [ModelRecord(**raw) for raw in index.values()]
        if state:
            recs = [r for r in recs if r.state == state]
        return sorted(recs, key=lambda r: r.created_at, reverse=True)

    def resolve_artifact(self, model_id: str) -> Path:
        """返回可用于推理的制品路径 (优先标准生产制品目录，其次注册表内制品, 回退源制品)"""
        rec = self.get(model_id)
        if rec is None:
            raise PromotionError(f"模型不存在: {model_id}")
        # 1. 优先检查生产规范目录 saved_models/production/<model_id>/model.pkl
        prod_path = self.root.parent / "production" / model_id / "model.pkl"
        if prod_path.exists():
            return prod_path
        # 2. 检查注册表制品
        if rec.registry_artifact:
            p = Path(rec.registry_artifact)
            if not p.is_absolute():
                p = Path(settings.BASE_DIR) / p
            if p.exists():
                return p
        # 3. 回退源制品
        if rec.source_artifact:
            src = Path(rec.source_artifact)
            if not src.is_absolute():
                src = Path(settings.BASE_DIR) / src
            if src.exists():
                return src
        raise FileNotFoundError(f"模型制品缺失: {model_id}")
