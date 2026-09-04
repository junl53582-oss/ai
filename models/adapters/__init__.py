"""
统一量化模型适配层 (Model Adapter Layer)
提供工厂模式与动态注册机制，支持 LightGBM, Hybrid Bagging Ridge, DRL 等任意注册模型。
"""
from typing import Dict, Type, List
from models.adapters.base_adapter import BaseModelAdapter
from models.adapters.lightgbm_adapter import LightGBMAdapter
from models.adapters.hybrid_bagging_ridge_adapter import HybridBaggingRidgeAdapter
from models.adapters.drl_adapter import DRLAdapter


_ADAPTER_REGISTRY: Dict[str, Type[BaseModelAdapter]] = {
    # LightGBM 系列
    "lightgbm": LightGBMAdapter,
    "lightgbm_ranker": LightGBMAdapter,
    "lightgbm_reg": LightGBMAdapter,
    "regression": LightGBMAdapter,
    "ranking": LightGBMAdapter,
    "classification": LightGBMAdapter,
    # 袋装浅树与线性底仓混合模型
    "hybrid_bagging_ridge": HybridBaggingRidgeAdapter,
    "bagging_ensemble": HybridBaggingRidgeAdapter,
    # 深度强化学习智能体模型
    "drl": DRLAdapter,
    "drl_strengthened": DRLAdapter,
    "drl_agent": DRLAdapter,
}


def register_adapter(model_type: str, adapter_cls: Type[BaseModelAdapter]) -> None:
    """动态注册新的模型适配器"""
    _ADAPTER_REGISTRY[model_type.lower()] = adapter_cls


def get_adapter(model_type: str, task_type: str = "classification") -> BaseModelAdapter:
    """
    根据 model_type 返回对应的适配器实例。
    若未注册则抛出 ValueError，由调用方捕获并转为 fail-closed 异常。
    """
    cls = _ADAPTER_REGISTRY.get(model_type.lower())
    if cls is None:
        raise ValueError(
            f"暂不支持推理的模型类型: {model_type} (已注册适配器: {list(_ADAPTER_REGISTRY.keys())})"
        )
    return cls(task_type=task_type)


def list_supported_adapters() -> List[str]:
    """返回当前已注册的所有模型类型"""
    return sorted(list(_ADAPTER_REGISTRY.keys()))


__all__ = [
    "BaseModelAdapter",
    "LightGBMAdapter",
    "HybridBaggingRidgeAdapter",
    "DRLAdapter",
    "register_adapter",
    "get_adapter",
    "list_supported_adapters",
]
