"""
时序注意力深度波形编码专家模型 (models/temporal_waveform_expert.py)
基于 PyTorch 深度学习架构:
1. 提取个股过去 seq_len (15~20 天) 的 3D 时序序列张量 [Batch, seq_len, Features]
2. 双层因果多头自注意力编码器 (Causal Multi-Head Self-Attention) + 正弦位置编码
3. 精准刻画“长周期蓄势后放量起爆”真实波形，过滤“冲高回落诱多派发”假突破
4. 封装为可直接与 GBDT/Ridge 无缝集成的 Scikit-Learn 兼容模型接口
"""

import math
import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model: int, max_len: int = 60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:L, :]


class WaveformTransformerNet(nn.Module):
    """PyTorch 时序 Transformer 波形编码网络"""
    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, input_dim]
        B, L, D = x.shape
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        # 生成因果掩码
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h_trans = self.transformer(h, mask=causal_mask, is_causal=True)
        # 提取最新时间步特征
        rep = h_trans[:, -1, :]
        return self.head(rep).squeeze(-1)


class TemporalWaveformExpert:
    """时序注意力波形深度模型专家适配器"""

    def __init__(
        self,
        seq_len: int = 15,
        d_model: int = 32,
        n_heads: int = 4,
        epochs: int = 3,
        batch_size: int = 512,
        learning_rate: float = 0.002
    ):
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.model: Optional[WaveformTransformerNet] = None
        self.device = torch.device("cpu")
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def _prepare_3d_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        """
        按个股分组构建 3D 时序序列张量 [N, seq_len, D]
        """
        clean_df = df.copy().sort_values(by=["symbol", "date"]).reset_index(drop=True)
        feats = clean_df[feature_cols].fillna(0.0).values
        n_rows, n_feats = feats.shape

        # 标准化
        if self.feature_means is None:
            self.feature_means = np.mean(feats, axis=0)
            self.feature_stds = np.std(feats, axis=0)
            self.feature_stds = np.where(self.feature_stds < 1e-5, 1.0, self.feature_stds)

        feats_norm = (feats - self.feature_means) / self.feature_stds

        # 快速构建滑窗
        # 对于每个样本，取同股票过去 seq_len 步，若不足则用自身首步填补
        sequences = np.zeros((n_rows, self.seq_len, n_feats), dtype=np.float32)
        symbols = clean_df["symbol"].values

        # 基于符号边界快速切片
        unique_syms, split_indices = np.unique(symbols, return_index=True)
        split_indices = np.append(split_indices, n_rows)

        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1]
            sym_len = end - start
            sym_feats = feats_norm[start:end]

            for t in range(sym_len):
                global_idx = start + t
                history_start = max(0, t - self.seq_len + 1)
                hist = sym_feats[history_start : t + 1]
                pad_len = self.seq_len - len(hist)
                if pad_len > 0:
                    pad = np.repeat(hist[:1], pad_len, axis=0)
                    sequences[global_idx] = np.vstack([pad, hist])
                else:
                    sequences[global_idx] = hist

        return sequences

    def fit(self, df: pd.DataFrame, feature_cols: List[str], label_col: str):
        """端到端训练时序 Transformer 网络"""
        logger.info(f"正在构建 3D 时序注意力波形张量 (窗口: {self.seq_len} 天, 特征数: {len(feature_cols)})...")
        X_seq = self._prepare_3d_sequences(df, feature_cols)
        y = df[label_col].fillna(0.0).values.astype(np.float32)

        n_samples = len(X_seq)
        input_dim = len(feature_cols)
        self.model = WaveformTransformerNet(
            input_dim=input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=2
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        logger.info(f"启动 PyTorch 时序波形模型训练 (样本量: {n_samples}, 轮次: {self.epochs})...")
        self.model.train()

        # 批次训练
        for epoch in range(self.epochs):
            perm = np.random.permutation(n_samples)
            total_loss = 0.0
            batches = int(np.ceil(n_samples / self.batch_size))

            for b in range(batches):
                idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                bx = torch.tensor(X_seq[idx], dtype=torch.float32, device=self.device)
                by = torch.tensor(y[idx], dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(f"  Epoch [{epoch+1}/{self.epochs}] MSE Loss: {total_loss / max(batches, 1):.6f}")

        self.model.eval()
        logger.info("时序注意力波形编码专家拟合完成！")
        return self

    def predict(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """执行时序波形推理"""
        if self.model is None:
            raise RuntimeError("TemporalWaveformExpert has not been fitted.")
        self.model.eval()
        X_seq = self._prepare_3d_sequences(df, feature_cols)
        preds = []
        with torch.no_grad():
            batches = int(np.ceil(len(X_seq) / self.batch_size))
            for b in range(batches):
                bx = torch.tensor(X_seq[b * self.batch_size : (b + 1) * self.batch_size], dtype=torch.float32, device=self.device)
                out = self.model(bx)
                preds.append(out.cpu().numpy())

        return np.concatenate(preds)
