"""
模型注册表与推理路径测试 (tests/test_model_registry.py)

锁定 Phase A 的 L13/L14 建设:
1. 研究运行只能登记 RESEARCH, 禁止直接写 PRODUCTION
2. 晋升门禁 fail-closed: 缺证据/缺审批人直接拒绝
3. PRODUCTION 唯一性: 新模型上线时旧生产模型自动归档
4. 推理路径: 无 PRODUCTION 时拒绝推理 (杜绝 research replay)
5. 血统漂移检查: 数据集哈希不一致时告警/严格模式抛错
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from models.registry import (ModelRegistry, ModelState, PromotionError,
                             ModelRecord)


@pytest.fixture()
def registry(tmp_path):
    return ModelRegistry(root=tmp_path / "registry")


@pytest.fixture()
def dummy_artifact(tmp_path) -> Path:
    """构造一个可被 LightGBMQuantModel.load 接受的最小制品"""
    from models.lightgbm_model import LightGBMQuantModel
    model = LightGBMQuantModel(task_type="classification")
    # 造极小训练样本以满足 fit 接口
    X = pd.DataFrame({"f1": np.random.rand(60), "f2": np.random.rand(60)})
    y = pd.Series(np.random.randint(0, 2, 60))
    model.fit(X, y)
    path = tmp_path / "research_model.pkl"
    model.save(filepath=path)
    return path


def _register(registry, artifact, **kw):
    return registry.register_research_artifact(
        artifact_path=artifact,
        model_type="lightgbm",
        task_type="classification",
        metrics=kw.pop("metrics", {"auc": 0.53, "rank_ic": 0.05}),
        dataset_sha256=kw.pop("dataset_sha256", "a" * 64),
        **kw,
    )


class TestRegistration:
    def test_research_only(self, registry, dummy_artifact):
        """研究运行只能登记 RESEARCH"""
        mid = _register(registry, dummy_artifact)
        assert registry.get(mid).state == ModelState.RESEARCH

    def test_cannot_register_directly_as_production(self, registry, dummy_artifact):
        """禁止研究运行直接登记为 PRODUCTION (研究禁写生产)"""
        with pytest.raises(PromotionError, match="RESEARCH"):
            registry.register_research_artifact(
                artifact_path=dummy_artifact, model_type="lightgbm", state=ModelState.PRODUCTION
            )

    def test_record_lineage(self, registry, dummy_artifact):
        mid = _register(registry, dummy_artifact, config_snapshot={"k": 1})
        rec = registry.get(mid)
        assert rec.dataset_sha256 == "a" * 64
        assert rec.code_commit is not None
        assert rec.config_snapshot == {"k": 1}
        assert rec.promotion_history and rec.promotion_history[0]["to"] == ModelState.RESEARCH


class TestPromotionGates:
    def test_research_to_candidate_needs_metrics(self, registry, dummy_artifact):
        mid = registry.register_research_artifact(
            artifact_path=dummy_artifact, model_type="lightgbm",
            metrics={}, dataset_sha256=None,
        )
        with pytest.raises(PromotionError, match="指标"):
            registry.promote(mid, ModelState.CANDIDATE, approver="researcher")

    def test_candidate_to_approved_needs_certification(self, registry, dummy_artifact):
        mid = _register(registry, dummy_artifact)
        registry.promote(mid, ModelState.CANDIDATE, approver="researcher")
        with pytest.raises(PromotionError, match="认证"):
            registry.promote(mid, ModelState.APPROVED, approver="linjun")
        # 提供认证证据后放行
        rec = registry.promote(mid, ModelState.APPROVED, approver="linjun",
                               evidence={"certification_ref": "reports/x/cert.json"})
        assert rec.state == ModelState.APPROVED
        assert rec.registry_artifact and Path(rec.registry_artifact).exists()

    def test_production_requires_approver_prospective_paper(self, registry, dummy_artifact):
        mid = _register(registry, dummy_artifact)
        registry.promote(mid, ModelState.CANDIDATE, approver="researcher")
        registry.promote(mid, ModelState.APPROVED, approver="linjun",
                         evidence={"certification_ref": "c.json"})
        # 缺审批人
        with pytest.raises(PromotionError, match="审批人"):
            registry.promote(mid, ModelState.PRODUCTION, approver="")
        # 缺前瞻验证 + 模拟盘
        with pytest.raises(PromotionError):
            registry.promote(mid, ModelState.PRODUCTION, approver="linjun")
        # 齐全后放行
        rec = registry.promote(mid, ModelState.PRODUCTION, approver="linjun", evidence={
            "certification_ref": "c.json",
            "prospective_validation": {"ref": "holdout_2026H1"},
            "paper_trading": {"ref": "paper_60d"},
        })
        assert rec.state == ModelState.PRODUCTION

    def test_illegal_transition_rejected(self, registry, dummy_artifact):
        mid = _register(registry, dummy_artifact)
        with pytest.raises(PromotionError, match="非法状态跃迁"):
            registry.promote(mid, ModelState.PRODUCTION, approver="linjun")


class TestProductionUniqueness:
    def test_old_production_auto_archived(self, registry, dummy_artifact):
        first = _register(registry, dummy_artifact)
        for mid in (first,):
            registry.promote(mid, ModelState.CANDIDATE, approver="researcher")
            registry.promote(mid, ModelState.APPROVED, approver="linjun",
                             evidence={"certification_ref": "c1.json"})
            registry.promote(mid, ModelState.PRODUCTION, approver="linjun", evidence={
                "certification_ref": "c1.json",
                "prospective_validation": {"ref": "p1"},
                "paper_trading": {"ref": "pt1"},
            })
        second = _register(registry, dummy_artifact, dataset_sha256="b" * 64)
        registry.promote(second, ModelState.CANDIDATE, approver="researcher")
        registry.promote(second, ModelState.APPROVED, approver="linjun",
                         evidence={"certification_ref": "c2.json"})
        registry.promote(second, ModelState.PRODUCTION, approver="linjun", evidence={
            "certification_ref": "c2.json",
            "prospective_validation": {"ref": "p2"},
            "paper_trading": {"ref": "pt2"},
        })
        assert registry.get_production().model_id == second
        assert registry.get(first).state == ModelState.ARCHIVED


class TestInference:
    def test_no_production_model_rejects_inference(self, registry):
        """无 PRODUCTION 时必须拒绝推理 (杜绝以当日重训充当生产)"""
        from models.inference import BatchInference, InferenceError
        with pytest.raises(InferenceError, match="PRODUCTION"):
            BatchInference(registry=registry)

    def test_production_inference_scores_and_ranks(self, registry, dummy_artifact):
        """生产模型推理: 输出 pred_score / pred_rank + 模型血统元数据"""
        from models.inference import BatchInference
        mid = _register(registry, dummy_artifact)
        registry.promote(mid, ModelState.CANDIDATE, approver="researcher")
        registry.promote(mid, ModelState.APPROVED, approver="linjun",
                         evidence={"certification_ref": "c.json"})
        registry.promote(mid, ModelState.PRODUCTION, approver="linjun", evidence={
            "certification_ref": "c.json",
            "prospective_validation": {"ref": "p"},
            "paper_trading": {"ref": "pt"},
        })
        engine = BatchInference(registry=registry)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05"] * 4),
            "symbol": ["A", "B", "C", "D"],
            "f1": [0.1, 0.4, 0.7, 0.9],
            "f2": [0.9, 0.6, 0.3, 0.1],
            "in_universe": [True, True, True, False],
        })
        out = engine.predict(df)
        assert {"pred_score", "pred_rank", "model_id", "model_state"} <= set(out.columns)
        assert out["model_state"].eq(ModelState.PRODUCTION).all()
        assert out["pred_rank"].notna().sum() == 3  # 非成分股不参与排名

    def test_missing_features_fail_closed(self, registry, dummy_artifact):
        from models.inference import BatchInference, InferenceError
        mid = _register(registry, dummy_artifact)
        registry.promote(mid, ModelState.CANDIDATE, approver="researcher")
        registry.promote(mid, ModelState.APPROVED, approver="linjun",
                         evidence={"certification_ref": "c"})
        registry.promote(mid, ModelState.PRODUCTION, approver="linjun", evidence={
            "certification_ref": "c", "prospective_validation": {"ref": "p"},
            "paper_trading": {"ref": "pt"}})
        engine = BatchInference(registry=registry)
        with pytest.raises(InferenceError, match="缺少"):
            engine.predict(pd.DataFrame({"date": ["2026-01-05"], "symbol": ["A"], "f1": [0.1]}))

    def test_lineage_drift_detected(self, registry, dummy_artifact):
        from models.inference import BatchInference, InferenceError
        mid = _register(registry, dummy_artifact)
        registry.promote(mid, ModelState.CANDIDATE, approver="researcher")
        registry.promote(mid, ModelState.APPROVED, approver="linjun",
                         evidence={"certification_ref": "c"})
        registry.promote(mid, ModelState.PRODUCTION, approver="linjun", evidence={
            "certification_ref": "c", "prospective_validation": {"ref": "p"},
            "paper_trading": {"ref": "pt"}})
        engine = BatchInference(registry=registry)
        report = engine.check_lineage(dataset_sha256="b" * 64)
        assert report["dataset_match"] is False
        with pytest.raises(InferenceError):
            engine.check_lineage(dataset_sha256="b" * 64, strict=True)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
