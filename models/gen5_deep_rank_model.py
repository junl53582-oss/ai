"""
第五代深度跨期排序学习与多周期动量引擎 (Gen 5: DeepRank Multi-Horizon Quant Model)
架构设计:
1. 目标函数: 跨期波动率归一化多尺度 Alpha (5日战术反转 + 10日波段动量 + 20日趋势共振)
2. 特征预处理: 逐日截面百分位均匀流形编码 (Uniform Percentile Rank Encoding)，彻底消除极值漂移
3. 双塔门控交叉注意力架构 (Dual-Tower Gated Cross-Attention):
   - 塔 A: 微观资金流与流动性冲击塔 (Amihud, 换手异动, 影线非对称, 资金流累积)
   - 塔 B: 多尺度价格动能与波动率挤压塔 (Mom Acc, Yang-Zhang, ATR, Bollinger)
   - 门控机制: 市场体制自适应门控 (Market Regime Gating) 动态决定动能 vs 资金流的主导权
4. 非对称下行风险惩罚: 对大跌样本赋予 3.0x 梯度惩罚，杜绝追高接盘
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class GatedCrossAttention(nn.Module):
    """自适应市场机制门控交叉注意力单元"""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

    def forward(self, tower_a: torch.Tensor, tower_b: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([tower_a, tower_b], dim=-1)
        g = self.gate(combined)
        fused = self.proj(combined)
        return g * fused + (1.0 - g) * tower_a


class DeepRankAlphaNet(nn.Module):
    """第五代双塔门控深度排序预测网络"""
    def __init__(self, dim_flow: int, dim_mom: int, hidden_dim: int = 64):
        super().__init__()
        self.dim_flow = dim_flow
        self.dim_mom = dim_mom

        # 塔 A: 微观资金流与微结构特征网络
        self.tower_flow = nn.Sequential(
            nn.Linear(dim_flow, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )

        # 塔 B: 多尺度动量与波动率挤压网络
        self.tower_mom = nn.Sequential(
            nn.Linear(dim_mom, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )

        # 门控交叉注意力融合层
        self.fused_gate = GatedCrossAttention(hidden_dim // 2)

        # 输出头: 连续 Alpha 预测与置信度打分
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_flow: torch.Tensor, x_mom: torch.Tensor) -> torch.Tensor:
        h_flow = self.tower_flow(x_flow)
        h_mom = self.tower_mom(x_mom)
        fused = self.fused_gate(h_flow, h_mom)
        out = self.head(fused).squeeze(-1)
        return out


class Gen5DeepRankModel:
    """第五代量化生产模型包装器"""
    def __init__(
        self,
        flow_features: List[str],
        mom_features: List[str],
        hidden_dim: int = 64,
        random_state: int = 2026
    ):
        self.flow_features = flow_features
        self.mom_features = mom_features
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.net = DeepRankAlphaNet(
            dim_flow=len(flow_features),
            dim_mom=len(mom_features),
            hidden_dim=hidden_dim
        )
        self.is_fitted = False

    def _rank_transform(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """执行严格截面百分位均匀排序归一化，范围在 [-0.5, 0.5]"""
        res = []
        for col in cols:
            ranked = df.groupby('date')[col].rank(pct=True).fillna(0.5).values - 0.5
            res.append(ranked)
        return np.column_stack(res)

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        epochs: int = 6,
        batch_size: int = 512,
        lr: float = 0.002
    ):
        """模型训练 (带非对称下行风险惩罚损失函数)"""
        logger.info("[*] 正在执行第五代 DeepRank 双塔特征矩阵截面均匀排序编码...")
        X_flow = self._rank_transform(train_df, self.flow_features)
        X_mom = self._rank_transform(train_df, self.mom_features)
        y = train_df[target_col].fillna(0.0).values

        # 转为 PyTorch 张量
        t_flow = torch.tensor(X_flow, dtype=torch.float32)
        t_mom = torch.tensor(X_mom, dtype=torch.float32)
        t_y = torch.tensor(y, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(t_flow, t_mom, t_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.net.train()
        logger.info(f"[*] 启动 DeepRank 神经网络优化 (共 {len(train_df):,} 行训练数据, {epochs} Epochs)...")

        for ep in range(epochs):
            total_loss = 0.0
            for b_flow, b_mom, b_y in loader:
                optimizer.zero_grad()
                pred = self.net(b_flow, b_mom)

                # 非对称下行风险加权损失: 实际大跌但预测看涨的假突破施加 3.0x 重罚
                diff = pred - b_y
                weights = torch.where((b_y < 0) & (pred > 0), 3.0, 1.0)
                loss = torch.mean(weights * (diff ** 2))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(b_y)

            scheduler.step()
            avg_loss = total_loss / len(dataset)
            logger.info(f"  [Epoch {ep+1}/{epochs}] Asymmetric Rank Loss: {avg_loss:.6f}")

        self.is_fitted = True
        logger.info("[+] 第五代 DeepRank 双塔门控网络训练圆满完成！")

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """模型推理，输出强区分度的连续 Alpha 分值"""
        self.net.eval()
        X_flow = self._rank_transform(test_df, self.flow_features)
        X_mom = self._rank_transform(test_df, self.mom_features)

        t_flow = torch.tensor(X_flow, dtype=torch.float32)
        t_mom = torch.tensor(X_mom, dtype=torch.float32)

        with torch.no_grad():
            preds = self.net(t_flow, t_mom).cpu().numpy()
        return preds
