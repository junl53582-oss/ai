import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from config.settings import settings

def test_drl_model_inference():
    model_path = settings.MODELS_DIR / "candidate_gen4_drl_model.pkl"
    assert model_path.exists(), "第四代强化模型未落盘"

    model = joblib.load(model_path)
    assert hasattr(model, "feature_model")
    assert hasattr(model, "drl_agent")

    # 构造假数据测试前向
    dummy_df = pd.DataFrame(np.random.randn(10, len(model.feature_names)), columns=model.feature_names)
    preds = model.predict_alpha(dummy_df)
    assert len(preds) == 10
    assert not np.isnan(preds).any()

    # 测试 DRL 权重优化
    cands = pd.DataFrame({
        'symbol': [f'STOCK_{i}' for i in range(10)],
        'pred_score': preds,
        'volatility': [0.02] * 10,
        'pct_change': [0.01] * 10,
        'turnover': [0.02] * 10
    })
    opt_df = model.optimize_portfolio(cands)
    assert 'drl_target_weight' in opt_df.columns
    assert len(opt_df) == 10
    assert np.isclose(opt_df['drl_target_weight'].sum(), 0.85, atol=1e-3)
