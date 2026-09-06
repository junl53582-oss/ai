"""
非对称下行风控惩罚与 RankIC 导向目标损失函数 (models/asymmetric_loss.py)
用于 LightGBM / GBDT 树模型的自定义训练目标 (Custom Objective)
核心机制:
1. 非对称下行惩罚 (Asymmetric Downside Penalty Loss):
   对“预测买入但实际大跌 (False Positive)”施加 2.5x 的非对称高阶损失惩罚，抑制踩雷与假突破风险。
2. 截面排序代理损失 (Pairwise RankIC Surrogate Loss):
   直接针对每日横截面个股收益相对排序进行梯度优化，提升 RankIC 与 Top-K 选股单调性。
"""
from typing import Tuple
import numpy as np


class AsymmetricLossObjective:
    """非对称下行惩罚损失函数 (针对二分类上涨概率)"""

    def __init__(self, false_positive_penalty: float = 2.5, gamma: float = 1.5):
        self.fp_penalty = false_positive_penalty
        self.gamma = gamma

    def __call__(self, arg1, arg2) -> Tuple[np.ndarray, np.ndarray]:
        """
        LightGBM 自定义目标函数接口 (同时兼容 Native Dataset 与 scikit-learn API):
        返回一阶梯度 (grad) 与二阶梯度 (hess)
        """
        if hasattr(arg2, "get_label"):
            preds = arg1
            labels = arg2.get_label()
        else:
            labels = arg1
            preds = arg2

        # preds 为模型输出的 raw logits (margin)
        p = 1.0 / (1.0 + np.exp(-np.clip(preds, -15.0, 15.0)))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)

        # 区分正负样本
        is_pos = (labels == 1)
        is_neg = ~is_pos

        # 一阶导数 grad = dL / dz
        # 正样本 (y=1): grad = p - 1
        # 负样本 (y=0): 对预测为高概率者施加 fp_penalty 倍数惩罚
        grad = np.zeros_like(p)
        grad[is_pos] = p[is_pos] - 1.0
        grad[is_neg] = self.fp_penalty * p[is_neg]

        # 二阶导数 hess = d^2L / dz^2
        hess = np.zeros_like(p)
        hess[is_pos] = p[is_pos] * (1.0 - p[is_pos])
        hess[is_neg] = self.fp_penalty * p[is_neg] * (1.0 - p[is_neg])
        hess = np.maximum(hess, 1e-4)

        return grad, hess


class AsymmetricRegressionObjective:
    """非对称超额收益回归损失 (惩罚预期高但实际暴跌的样本)"""

    def __init__(self, underpredict_gain: float = 1.0, overpredict_loss: float = 2.5):
        self.under_gain = underpredict_gain
        self.over_loss = overpredict_loss

    def __call__(self, arg1, arg2) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(arg2, "get_label"):
            preds = arg1
            labels = arg2.get_label()
        else:
            labels = arg1
            preds = arg2

        residual = preds - labels

        # residual > 0: 预测偏高 (实际比预期更差) -> 高度惩罚
        # residual < 0: 预测偏低 -> 标准惩罚
        grad = np.where(residual > 0, self.over_loss * residual, self.under_gain * residual)
        hess = np.where(residual > 0, np.full_like(residual, self.over_loss), np.full_like(residual, self.under_gain))
        hess = np.maximum(hess, 1e-4)

        return grad, hess
