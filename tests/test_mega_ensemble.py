import pytest
import numpy as np
import pandas as pd
from models.mega_ensemble import MegaEnsembleQuantModel

def test_mega_ensemble_fit_and_predict():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'f1': np.random.normal(0, 1, n),
        'f2': np.random.normal(0, 1, n),
        'f3': np.random.normal(0, 1, n),
    })
    y = pd.Series((df['f1'] * 0.5 + df['f2'] * 0.3 + np.random.normal(0, 0.5, n) > 0).astype(int))

    model = MegaEnsembleQuantModel(task_type='classification', n_de_submodels=2)
    model.fit(df, y, feature_names=['f1', 'f2', 'f3'])
    
    preds = model.predict(df)
    assert len(preds) == n
    assert not np.isnan(preds).any()
    assert (preds >= 0.0).all() and (preds <= 1.0).all()
