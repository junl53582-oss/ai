"""
批量推理引擎 (models/inference.py)

背景 (Phase A / 2026-09-01 架构审计):
    审计判定推理层 NOT_IMPLEMENTED: 每日流程跑的是 run_walk_forward (重训 20 折),
    然后取 OOS 最后一日的预测当"今日信号" —— 这在架构上是 research replay,
    不是 inference: 模型每天都在变, 无法归因, 也无法回放/复现某一天的信号。

本模块提供真正的推理路径:
    1. 从 ModelRegistry 解析【唯一 PRODUCTION 模型】(或显式 model_id)
    2. 加载模型制品 (LightGBMQuantModel.load, 自带 fail-closed schema 校验)
    3. 对给定特征截面计算 pred_score 与截面排名 pred_rank
    4. 血统与漂移检查: 数据集哈希/代码 commit 与训练期不一致时显式告警

铁律: 本模块【禁止】训练、禁止写模型制品、禁止修改注册表状态。

用法:
    engine = BatchInference()                    # 默认取当前 PRODUCTION
    scored = engine.predict(features_df)         # -> 含 pred_score / pred_rank 的 DataFrame
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config.settings import settings
from models.registry import ModelRegistry, ModelState, PromotionError

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """推理前置条件不满足 (fail-closed)"""
    pass


class BatchInference:
    """批量截面推理引擎 (只推理, 不训练)"""

    def __init__(
        self,
        model_id: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
        require_production: bool = True,
    ):
        self.registry = registry or ModelRegistry()
        if model_id:
            self.record = self.registry.get(model_id)
            if self.record is None:
                raise InferenceError(f"注册表不存在模型: {model_id}")
            if require_production and self.record.state != ModelState.PRODUCTION:
                raise InferenceError(
                    f"模型 {model_id} 状态为 {self.record.state}, 非 PRODUCTION —— "
                    "推理禁止使用未上线模型 (若确需, 显式传 require_production=False)"
                )
        else:
            self.record = self.registry.get_production()
            if self.record is None:
                raise InferenceError(
                    "注册表无 PRODUCTION 模型: 必须先经研究→候选→获批→上线流程产出生产模型, "
                    "禁止以当日重训结果充当生产推理 (research replay)"
                )
        self.model = self._load_model()
        self.model_id = self.record.model_id

    # ---------------------------------------------------------------- 加载
    def _load_model(self):
        artifact = self.registry.resolve_artifact(self.record.model_id)
        from models.adapters import get_adapter
        try:
            adapter = get_adapter(self.record.model_type, self.record.task_type or "classification")
        except ValueError as e:
            raise InferenceError(f"暂不支持推理的模型类型: {self.record.model_type}") from e
        adapter.load(artifact)
        self.adapter = adapter
        return adapter

    # ---------------------------------------------------------------- 漂移检查
    def check_lineage(self, dataset_sha256: Optional[str] = None, strict: bool = False) -> Dict[str, Any]:
        """比较当前数据集/代码与训练期血统, 不一致时告警 (strict=True 则抛错)"""
        report = {
            "model_id": self.record.model_id,
            "trained_dataset_sha256": self.record.dataset_sha256,
            "current_dataset_sha256": dataset_sha256,
            "trained_code_commit": self.record.code_commit,
            "dataset_match": None,
        }
        if dataset_sha256 and self.record.dataset_sha256:
            report["dataset_match"] = (dataset_sha256 == self.record.dataset_sha256)
            if not report["dataset_match"]:
                msg = (f"数据集血统不一致: 训练 {self.record.dataset_sha256[:12]}... "
                       f"vs 当前 {dataset_sha256[:12]}... (特征口径可能漂移)")
                logger.warning(msg)
                if strict:
                    raise InferenceError(msg)
        return report

    # ---------------------------------------------------------------- 推理
    def predict(
        self,
        features_df: pd.DataFrame,
        date=None,
        dataset_sha256: Optional[str] = None,
        strict_lineage: bool = False,
    ) -> pd.DataFrame:
        """
        对截面特征打分。输出列: pred_score (+ pred_rank 若存在 in_universe 列)。

        Args:
            features_df: 含模型所需全部特征列的截面 (或面板) 数据
            date: 可选, 指定推理日 (传入面板时按该日过滤)
            dataset_sha256: 当前数据集哈希, 用于血统漂移检查
            strict_lineage: 血统不一致时是否直接失败
        """
        df = features_df.copy()
        if date is not None and "date" in df.columns:
            df = df[df["date"] == pd.to_datetime(date)].copy()
        if df.empty:
            raise InferenceError("推理输入为空 (缺少对应日期截面数据)")

        self.check_lineage(dataset_sha256=dataset_sha256, strict=strict_lineage)

        required = list(self.model.feature_names or [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise InferenceError(
                f"推理输入缺少 {len(missing)} 个模型特征: {missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''}"
            )

        df["pred_score"] = self.model.predict(df[required])
        if "in_universe" in df.columns:
            univ = df["in_universe"].fillna(False).astype(bool)
            df["pred_rank"] = df.loc[univ, "pred_score"].rank(ascending=False, pct=True)
        else:
            df["pred_rank"] = df["pred_score"].rank(ascending=False, pct=True)

        # 推理元数据 (可追溯每条信号来自哪个模型版本)
        df["model_id"] = self.record.model_id
        df["model_state"] = self.record.state
        df["inference_at"] = datetime.now().isoformat(timespec="seconds")

        logger.info(
            f"批量推理完成: model={self.record.model_id} ({self.record.state}) | "
            f"样本 {len(df)} 行 | 特征 {len(required)} 个"
        )
        return df
