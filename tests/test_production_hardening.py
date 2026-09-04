"""
GEN4.5 生产加固与回归测试 (tests/test_production_hardening.py)

测试覆盖:
1. ModelRegistry 中所有标记为 PRODUCTION 的模型均可通过 BatchInference 加载并输出合规预测。
2. 调度器在生产环境下禁止 research 模式的防御机制 (fail-closed)。
3. 模型类型与实际模型对象的一致性验证 (防混淆)。
4. 统一制品目录 (saved_models/production/<model_id>) 与 manifest 哈希自洽性。
5. 每日盘后批处理脚本调用参数安全验证。
"""
import os
import json
import hashlib
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from config.settings import settings
from models.registry import ModelRegistry, ModelState
from models.inference import BatchInference, InferenceError
from models.adapters import (
    BaseModelAdapter,
    LightGBMAdapter,
    HybridBaggingRidgeAdapter,
    DRLAdapter,
    get_adapter,
    list_supported_adapters
)
from scheduler.daily_runner import run_daily_automation, is_production_environment


class TestProductionModelAdapterCompatibility:
    """任务1测试: 生产模型适配器与 BatchInference 推理兼容性"""

    def test_production_model_loads_via_batch_inference(self):
        """验证当前 ModelRegistry 中的唯一 PRODUCTION 模型能被 BatchInference 正常加载"""
        registry = ModelRegistry()
        prod_rec = registry.get_production()
        assert prod_rec is not None, "ModelRegistry 中必须存在唯一的 PRODUCTION 模型"
        assert prod_rec.model_type == "hybrid_bagging_ridge"

        engine = BatchInference(registry=registry)
        assert engine.model_id == prod_rec.model_id
        assert isinstance(engine.model, BaseModelAdapter)
        assert len(engine.model.feature_names) > 0

    def test_production_model_predicts_correctly(self):
        """验证生产模型对截面特征打分并生成合规的 pred_score 与 pred_rank"""
        engine = BatchInference()
        required_features = engine.model.feature_names
        assert len(required_features) >= 10

        # 构建合成截面数据
        n_rows = 15
        data = {f: np.random.uniform(0.0, 1.0, n_rows) for f in required_features}
        data["date"] = pd.to_datetime("2026-09-04")
        data["symbol"] = [f"{600000 + i:06d}.SH" for i in range(n_rows)]
        data["in_universe"] = [True] * 10 + [False] * 5
        df = pd.DataFrame(data)

        scored = engine.predict(df)
        assert "pred_score" in scored.columns
        assert "pred_rank" in scored.columns
        assert "model_id" in scored.columns
        assert "model_state" in scored.columns

        # 校验数值合法性
        scores = scored["pred_score"].dropna()
        assert len(scores) == n_rows
        assert np.isfinite(scores).all()

        # 校验 in_universe 过滤: 只有在股票池内的标的参与 pred_rank 排名
        ranks = scored.loc[scored["in_universe"], "pred_rank"]
        assert len(ranks.dropna()) == 10
        assert (ranks > 0.0).all() and (ranks <= 1.0).all()
        assert scored.loc[~scored["in_universe"], "pred_rank"].isna().all()

    def test_all_registered_adapters_interface(self):
        """验证所有已注册适配器均继承 BaseModelAdapter 并具备必要抽象接口"""
        for model_type in list_supported_adapters():
            adapter = get_adapter(model_type)
            assert isinstance(adapter, BaseModelAdapter)
            assert hasattr(adapter, "load")
            assert hasattr(adapter, "predict")
            assert hasattr(adapter, "feature_names")

    def test_unsupported_model_type_fails_closed(self, tmp_path):
        """验证不受支持的模型类型触发 fail-closed 抛错"""
        reg = ModelRegistry(root=tmp_path / "registry")
        dummy_file = tmp_path / "dummy.pkl"
        dummy_file.write_bytes(b"dummy")

        mid = reg.register_research_artifact(
            artifact_path=dummy_file,
            model_type="unknown_quantum_net",
            metrics={"auc": 0.6},
            dataset_sha256="c" * 64
        )
        reg.promote(mid, ModelState.CANDIDATE, approver="test")
        reg.promote(mid, ModelState.APPROVED, approver="test", evidence={"certification_ref": "c"})
        reg.promote(mid, ModelState.PRODUCTION, approver="test", evidence={
            "certification_ref": "c", "prospective_validation": True, "paper_trading": True
        })

        with pytest.raises(InferenceError, match="暂不支持推理的模型类型"):
            BatchInference(registry=reg)


class TestSchedulerProductionModeDefense:
    """任务2测试: 调度器生产运行模式与防御机制"""

    def test_scheduler_default_mode_is_inference(self):
        """验证调度器自动化流水线的默认模式为 inference"""
        import inspect
        sig = inspect.signature(run_daily_automation)
        assert sig.parameters["mode"].default == "inference"

    def test_production_runtime_blocks_research_mode(self):
        """验证显式传入 production_runtime=True 时，research 模式被严格拦截 (fail-closed)"""
        with pytest.raises(RuntimeError, match="fail-closed"):
            run_daily_automation(mode="research", production_runtime=True)

    def test_production_env_var_blocks_research_mode(self, monkeypatch):
        """验证当环境变量设置 PRODUCTION_RUNTIME=1 时，research 模式被严格拦截"""
        monkeypatch.setenv("PRODUCTION_RUNTIME", "1")
        assert is_production_environment() is True
        with pytest.raises(RuntimeError, match="fail-closed"):
            run_daily_automation(mode="research")

    def test_bat_script_contains_explicit_inference_mode(self):
        """验证 Windows 盘后批处理脚本显式包含 --mode inference 与 --production-runtime"""
        bat_path = Path(settings.BASE_DIR) / "scripts" / "run_daily_post_market.bat"
        assert bat_path.exists()
        content = bat_path.read_text(encoding="utf-8")
        assert "--mode inference" in content
        assert "--production-runtime" in content
        assert "[MODE=INFERENCE]" in content


class TestModelArtifactsAndAntiConfusingStructure:
    """任务3测试: 统一制品存储结构与防混淆规范"""

    def test_production_model_directory_structure(self):
        """验证 saved_models/production/<model_id>/ 目录结构与元数据完整性"""
        prod_dir = Path(settings.MODELS_DIR) / "production" / "m_20260903_194757_hybrid_bagging_ridge"
        assert prod_dir.exists()
        assert (prod_dir / "model.pkl").exists()
        assert (prod_dir / "metadata.json").exists()
        assert (prod_dir / "manifest.json").exists()

        manifest = json.loads((prod_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["model_id"] == "m_20260903_194757_hybrid_bagging_ridge"
        assert manifest["model_type"] == "hybrid_bagging_ridge"
        assert manifest["file_name"] == "model.pkl"

        # 验证 sha256 一致性
        actual_sha = hashlib.sha256((prod_dir / "model.pkl").read_bytes()).hexdigest()
        assert manifest["file_sha256"] == actual_sha

    def test_production_model_object_class_matches_type(self):
        """验证生产模型对象真实类为 MultiSeedBaggingModel，杜绝类型挂羊头卖狗肉"""
        import joblib
        from models.bagging_ensemble import MultiSeedBaggingModel
        prod_file = Path(settings.MODELS_DIR) / "production" / "m_20260903_194757_hybrid_bagging_ridge" / "model.pkl"
        obj = joblib.load(prod_file)
        assert isinstance(obj, MultiSeedBaggingModel)

    def test_legacy_artifacts_marked_and_preserved(self):
        """验证 saved_models/legacy 目录正确归档并提供了防混淆说明与清单"""
        legacy_dir = Path(settings.MODELS_DIR) / "legacy"
        assert legacy_dir.exists()
        assert (legacy_dir / "README.md").exists()
        assert (legacy_dir / "manifest.json").exists()

        manifest = json.loads((legacy_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "latest_lightgbm.pkl" in manifest
        readme = (legacy_dir / "README.md").read_text(encoding="utf-8")
        assert "DRLStrengthenedQuantModel" in readme


class TestManifestIntegrityVerification:
    """任务4测试: Manifest 防篡改哈希自洽性验证"""

    def test_factor_matrix_manifest_hash_matches_parquet(self):
        """验证 factor_matrix_300.parquet 与 manifest 的 SHA-256 绝对一致"""
        parquet_file = Path(settings.BASE_DIR) / "data_storage" / "research" / "factor_matrix_300.parquet"
        manifest_file = Path(settings.BASE_DIR) / "data_storage" / "research" / "factor_matrix_300.manifest.json"

        if parquet_file.exists() and manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            actual_sha = hashlib.sha256(parquet_file.read_bytes()).hexdigest()
            assert manifest["file_sha256"] == actual_sha
