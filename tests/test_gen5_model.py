import pytest
import numpy as np
import pandas as pd
import torch
from models.gen5_deep_rank_model import GatedCrossAttention, DeepRankAlphaNet, Gen5DeepRankModel

def test_gated_cross_attention():
    cell = GatedCrossAttention(dim=16)
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)
    out = cell(a, b)
    assert out.shape == (4, 16)
    assert not torch.isnan(out).any()

def test_deep_rank_forward():
    net = DeepRankAlphaNet(dim_flow=5, dim_mom=6, hidden_dim=32)
    flow = torch.randn(8, 5)
    mom = torch.randn(8, 6)
    pred = net(flow, mom)
    assert pred.shape == (8,)
    assert not torch.isnan(pred).any()

def test_gen5_model_fit_predict():
    dates = pd.date_range('2026-01-01', periods=10, freq='D')
    symbols = [f'{i:06d}.SH' for i in range(20)]
    rows = []
    for d in dates:
        for s in symbols:
            rows.append({
                'date': d,
                'symbol': s,
                'f1': np.random.randn(),
                'f2': np.random.randn(),
                'm1': np.random.randn(),
                'm2': np.random.randn(),
                'target': np.random.randn()
            })
    df = pd.DataFrame(rows)
    model = Gen5DeepRankModel(flow_features=['f1', 'f2'], mom_features=['m1', 'm2'], hidden_dim=16)
    model.fit(df, target_col='target', epochs=2, batch_size=32)
    preds = model.predict(df.head(20))
    assert len(preds) == 20
    assert not np.isnan(preds).any()
    # 验证预测值具有真实方差，绝非死水常数
    assert np.std(preds) > 1e-4
