import pytest
import pandas as pd
import numpy as np
from models.bagging_ensemble import MultiSeedBaggingModel

def test_bagging_ensemble_fit_and_predict():
    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = 6
    
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=[f'f_{i}' for i in range(n_features)])
    # 模拟真实信号: f_0 与 f_1 正相关
    y = ((X['f_0'] * 0.5 + X['f_1'] * 0.3 + rng.normal(scale=0.1, size=n_samples)) > 0).astype(int)

    model = MultiSeedBaggingModel(
        seeds=[42, 100, 2024],
        task_type="classification",
        n_estimators=30,
        num_leaves=7
    )
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == n_samples
    assert np.all((preds >= 0.0) & (preds <= 1.0))

    fi = model.get_feature_importance(top_n=3)
    assert len(fi) == 3
    assert 'f_0' in fi['feature'].values or 'f_1' in fi['feature'].values
