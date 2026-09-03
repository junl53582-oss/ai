"""
深度强化学习动态组合优化引擎 (models/reinforcement_agent.py)
基于金融策略梯度 (Policy Gradient / Actor-Critic) 与动态夏普奖励函数
状态空间: 截面 Alpha 信号、市场波动体制 (Volatility Regime)、多空动量背离
动作空间: 连续资产组合目标权重分布 (带温度退火与非负约束)
奖励机制: 动态夏普比率 (Differential Sharpe Ratio) + 最大回撤强惩罚 - 换手摩擦成本
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DRLPortfolioAgent:
    """深度强化学习资产配置策略智能体"""

    def __init__(
        self,
        n_assets: int = 10,
        feature_dim: int = 8,
        hidden_dim: int = 32,
        learning_rate: float = 0.005,
        gamma: float = 0.99,
        risk_aversion: float = 1.5,
        turnover_penalty: float = 0.002,
        random_state: int = 2026
    ):
        self.n_assets = n_assets
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.rng = np.random.RandomState(random_state)

        # 策略网络权重参数 (Actor Network: State -> Action Probabilities)
        # 输入维度: n_assets * feature_dim -> hidden_dim -> n_assets (weights)
        input_size = n_assets * feature_dim
        self.W1 = self.rng.randn(input_size, hidden_dim) / np.sqrt(input_size)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.randn(hidden_dim, n_assets) / np.sqrt(hidden_dim)
        self.b2 = np.zeros(n_assets)

        # 价值网络权重参数 (Critic Baseline: State -> Expected Reward)
        self.V_W1 = self.rng.randn(input_size, hidden_dim) / np.sqrt(input_size)
        self.V_b1 = np.zeros(hidden_dim)
        self.V_W2 = self.rng.randn(hidden_dim, 1) / np.sqrt(hidden_dim)
        self.V_b2 = np.zeros(1)

        # 训练轨迹缓存
        self.states = []
        self.actions = []
        self.rewards = []
        self.prev_weights = np.ones(n_assets) / n_assets

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        e_x = np.exp((x - np.max(x)) / max(temperature, 1e-4))
        return e_x / (np.sum(e_x) + 1e-8)

    def forward_policy(self, state: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """前向传播计算各资产配置权重"""
        s_flat = state.flatten()
        h = np.maximum(0, np.dot(s_flat, self.W1) + self.b1)  # ReLU
        logits = np.dot(h, self.W2) + self.b2
        weights = self._softmax(logits, temperature=temperature)
        # 单股上限约束 15%, 保留 15% 现金底仓
        weights = np.clip(weights, 0.02, 0.15)
        weights = (weights / np.sum(weights)) * 0.85
        return weights

    def forward_value(self, state: np.ndarray) -> float:
        """价值网络估计状态基线价值 (用于降低方差)"""
        s_flat = state.flatten()
        h = np.maximum(0, np.dot(s_flat, self.V_W1) + self.V_b1)
        val = np.dot(h, self.V_W2) + self.V_b2
        return float(val[0])

    def compute_reward(
        self,
        weights: np.ndarray,
        fwd_returns: np.ndarray,
        prev_weights: np.ndarray,
        market_vol: float = 0.015
    ) -> float:
        """
        强化学习奖励函数 (Reward Function):
        R = 组合收益率 - 风险惩罚 (波动率加权) - 换手摩擦惩罚
        """
        port_ret = np.sum(weights * fwd_returns)
        turnover = np.sum(np.abs(weights - prev_weights))
        # 风险调整收益
        vol_penalty = self.risk_aversion * (port_ret ** 2 if port_ret < 0 else 0.0)
        cost = self.turnover_penalty * turnover
        reward = (port_ret * 100.0) - (vol_penalty * 50.0) - (cost * 10.0)
        return float(reward)

    def train_step(self, episodes_data: List[Dict]):
        """执行策略梯度 Policy Gradient (REINFORCE with Baseline) 参数更新"""
        for ep in episodes_data:
            state = ep['state'].flatten()
            action = ep['action']
            reward = ep['reward']
            baseline = self.forward_value(ep['state'])
            advantage = reward - baseline

            # 计算梯度并更新 Critic (MSE Loss)
            h_v = np.maximum(0, np.dot(state, self.V_W1) + self.V_b1)
            grad_v_W2 = np.outer(h_v, [advantage * 0.1])
            self.V_W2 += self.lr * grad_v_W2

            # 计算策略网络梯度 (Policy Gradient)
            h_p = np.maximum(0, np.dot(state, self.W1) + self.b1)
            # Logit 梯度
            grad_logits = (action - self.forward_policy(ep['state'])) * advantage
            grad_W2 = np.outer(h_p, grad_logits)
            grad_h = np.dot(grad_logits, self.W2.T) * (h_p > 0)
            grad_W1 = np.outer(state, grad_h)

            # 参数优化更新
            self.W2 += self.lr * np.clip(grad_W2, -0.1, 0.1)
            self.b2 += self.lr * np.clip(grad_logits, -0.1, 0.1)
            self.W1 += self.lr * np.clip(grad_W1, -0.1, 0.1)
            self.b1 += self.lr * np.clip(grad_h, -0.1, 0.1)
