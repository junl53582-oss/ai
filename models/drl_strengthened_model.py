"""
第四代强化模型: Mega-Alpha 深度特征提取 + DRL 强化学习动态优化智能体 (models/drl_strengthened_model.py)
结合了 Qlib DoubleEnsemble、TabularMLP 流形表征 与 策略梯度强化学习动态风控自适应
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
import joblib

from models.mega_ensemble import MegaEnsembleQuantModel
from models.reinforcement_agent import DRLPortfolioAgent

logger = logging.getLogger(__name__)

class DRLStrengthenedQuantModel:
    """深度强化学习增强型量化预测系统 (Gen 4)"""

    def __init__(
        self,
        n_top_assets: int = 10,
        drl_lr: float = 0.005,
        random_state: int = 2026
    ):
        self.n_top_assets = n_top_assets
        # 1. 底层特征表征引擎 (第三代 MegaEnsemble)
        self.feature_model = MegaEnsembleQuantModel(
            task_type='classification',
            w_double_ensemble=0.55,
            w_mlp=0.25,
            w_ridge=0.20,
            n_de_submodels=3,
            random_state=random_state
        )
        # 2. 顶层强化学习智能体 (DRL Agent)
        # 特征维度: 4 维 (预测分, 波动率, 近端涨幅, 换手率)
        self.drl_agent = DRLPortfolioAgent(
            n_assets=n_top_assets,
            feature_dim=4,
            hidden_dim=32,
            learning_rate=drl_lr,
            random_state=random_state
        )
        self.feature_names = []

    def fit_base_model(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str], sample_weight: Optional[np.ndarray] = None):
        """训练底层特征表征基模"""
        self.feature_names = feature_names
        self.feature_model.fit(X, y, feature_names=feature_names, sample_weight=sample_weight)

    def train_drl_policy(self, cross_sections_data: List[Dict]):
        """
        在时序截面上强化训练 DRL 智能体
        cross_sections_data: 包含每个交易日 top 标的的 (state, fwd_returns, market_vol)
        """
        logger.info(f"[*] 启动 DRL 强化学习智能体策略梯度训练 (样本数: {len(cross_sections_data)} 个截面)...")
        episodes = []
        prev_w = np.ones(self.n_top_assets) / self.n_top_assets

        for item in cross_sections_data:
            state = item['state']  # shape: (n_assets, 4)
            fwd_ret = item['fwd_returns']
            mkt_vol = item.get('market_vol', 0.015)

            # 智能体决策
            action_w = self.drl_agent.forward_policy(state)
            # 计算奖励
            reward = self.drl_agent.compute_reward(action_w, fwd_ret, prev_w, market_vol=mkt_vol)

            episodes.append({
                'state': state,
                'action': action_w,
                'reward': reward
            })
            prev_w = action_w

        # 执行强化学习参数反向传播与策略更新
        self.drl_agent.train_step(episodes)
        logger.info("[+] DRL 强化学习策略网络参数优化更新完成！")

    def predict_alpha(self, X: pd.DataFrame) -> np.ndarray:
        """预测截面 Alpha 原始得分"""
        return self.feature_model.predict(X)

    def optimize_portfolio(self, top_candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        利用训练好的 DRL 智能体对候选标的执行动态组合权重强化优化
        输入需包含: pred_score, volatility, momentum, turnover
        """
        res_df = top_candidates_df.head(self.n_top_assets).copy()
        
        # 构建强化学习状态向量 (n_assets, 4)
        states = []
        for _, row in res_df.iterrows():
            sc = row.get('pred_score', 0.5)
            vol = row.get('volatility', 0.02)
            mom = row.get('pct_change', 0.0)
            to = row.get('turnover', 0.01)
            states.append([sc, vol, mom, to])
        
        state_mat = np.array(states)
        if len(state_mat) < self.n_top_assets:
            # 补齐
            pad = np.zeros((self.n_top_assets - len(state_mat), 4))
            state_mat = np.vstack([state_mat, pad])
        
        # DRL 智能体前向输出最优风险调整权重
        optimal_weights = self.drl_agent.forward_policy(state_mat)
        res_df['drl_target_weight'] = optimal_weights[:len(res_df)]
        # 归一化至总股票计划仓位 85%
        res_df['drl_target_weight'] = (res_df['drl_target_weight'] / res_df['drl_target_weight'].sum()) * 0.85
        return res_df
