import pytest
torch = pytest.importorskip("torch", reason="PyTorch not installed, skipping Temporal Transformer tests")
import numpy as np
from models.temporal_transformer import PositionalEncoding, TemporalTransformerAlphaModel

def test_positional_encoding_torch():
    pe = PositionalEncoding(d_model=32, max_len=50)
    x = torch.zeros(2, 20, 32)
    out = pe(x)
    assert out.shape == (2, 20, 32)
    assert not torch.isnan(out).any()

def test_causal_mask_non_leakage_torch():
    """严苛验证时序 Transformer 在前向传播时无法获取未来数据"""
    model = TemporalTransformerAlphaModel(input_dim=8, seq_len=10, d_model=16, n_heads=2, d_ff=32)
    model.eval()
    
    # 构造两组序列，其中前 9 个时间步完全相同，仅第 10 个时间步不同
    seq1 = torch.randn(1, 10, 8)
    seq2 = seq1.clone()
    seq2[:, 9, :] = torch.randn(1, 8) # 改变最后一天的未来信息
    
    # 获取前 9 天的中间表征 (通过 hook 或比较)
    with torch.no_grad():
        h1 = model.input_proj(seq1)
        h1 = model.pos_encoder(h1)
        mask1 = torch.nn.Transformer.generate_square_subsequent_mask(10)
        out1 = model.transformer(h1, mask=mask1, is_causal=True)

        h2 = model.input_proj(seq2)
        h2 = model.pos_encoder(h2)
        out2 = model.transformer(h2, mask=mask1, is_causal=True)

    # 验证在第 0 到第 8 个时间步，表征必须完全一致！第 10 天的变化绝对不会影响前 9 天！
    diff_past = torch.abs(out1[:, :9, :] - out2[:, :9, :]).max().item()
    assert diff_past < 1e-6, f"严重因果泄露: 未来数据扰动了过去表征, diff={diff_past}"

def test_transformer_training_and_predict():
    model = TemporalTransformerAlphaModel(input_dim=6, seq_len=10, d_model=16, n_heads=2, d_ff=32)
    X_train = np.random.randn(64, 10, 6)
    y_train = np.random.randn(64)
    model.fit_dataset(X_train, y_train, epochs=2, batch_size=32)
    
    preds = model.predict_sequences(X_train[:10])
    assert len(preds) == 10
    assert not np.isnan(preds).any()
