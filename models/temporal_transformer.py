"""
跨周期 Temporal Transformer 时序自注意力机制架构 (models/temporal_transformer.py)
基于 PyTorch 实现工业级多头时序自注意力、正弦位置编码与严格因果掩码 (Causal Mask)
专为 A 股时间序列微观动量突变、波动率聚集与跨期因果依赖设计
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """正弦/余弦时序位置编码"""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, D)
        L = x.size(1)
        return x + self.pe[:L, :]


class TemporalTransformerAlphaModel(nn.Module):
    """端到端 PyTorch 跨周期时序 Transformer 预测网络"""
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 20,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 1. 因子微观特征投影层
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len + 10)
        
        # 2. 时序 Transformer 编码器层 (因果时序掩码)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. 输出 Alpha 预测头
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成严格下三角因果掩码 (上三角为 -inf, 严禁未来穿越)"""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, input_dim)
        B, L, D = x.shape
        h = self.input_proj(x) # (B, L, d_model)
        h = self.pos_encoder(h)
        
        # 因果掩码: 只能看过去和现在，不能看未来
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h_trans = self.transformer(h, mask=causal_mask, is_causal=True) # (B, L, d_model)
        
        # 提取最新时序步 (L-1) 的表征
        latest_rep = h_trans[:, -1, :] # (B, d_model)
        out = self.head(latest_rep).squeeze(-1) # (B,)
        return out

    def fit_dataset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 5,
        batch_size: int = 256,
        lr: float = 0.001
    ):
        """端到端时序序列训练拟合"""
        self.train()
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for ep in range(epochs):
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(bx)
            avg_loss = total_loss / len(dataset)
            logger.info(f"Temporal Transformer Epoch {ep+1}/{epochs} | Loss: {avg_loss:.6f}")

    def predict_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """输入 (N, L, D) 预测输出一维 Alpha 评分"""
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_seq, dtype=torch.float32)
            preds = self(X_t).cpu().numpy()
        return preds
