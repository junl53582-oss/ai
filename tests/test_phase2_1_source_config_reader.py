"""
Tests for Source Config Reader (tests/test_phase2_1_source_config_reader.py)
"""
from pathlib import Path
from research_v2.provenance.source_config_reader import (
    read_source_file_from_git,
    parse_settings_from_source,
    resolve_historical_effective_configs,
    compute_legacy_effective_model_config_hash
)


def test_ast_parsing_of_historical_settings():
    code = read_source_file_from_git("e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48", "config/settings.py")
    settings_dict = parse_settings_from_source(code)
    
    assert "LGBM_PARAMS_CLF" in settings_dict
    assert "LGBM_PARAMS" in settings_dict
    assert settings_dict["LGBM_PARAMS_CLF"]["objective"] == "binary"
    assert settings_dict["LGBM_PARAMS"]["objective"] == "regression"
    assert settings_dict["TRAIN_WINDOW_YEARS"] == 1.5
    assert settings_dict["VAL_WINDOW_MONTHS"] == 3
    assert settings_dict["TEST_WINDOW_MONTHS"] == 2
    assert settings_dict["LABEL_HORIZON"] == 20


def test_effective_ranker_config_resolution():
    eff = resolve_historical_effective_configs("e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48")
    models = eff["models"]
    
    ranker = models["lightgbm_ranker"]
    assert ranker["effective_estimator_class"] == "LGBMRanker"
    assert ranker["effective_objective"] == "regression"
    assert ranker["effective_metric"] == "rmse"
    assert ranker["true_lambdarank_certified"] is False

    clf = models["lightgbm_clf_baseline"]
    assert clf["effective_objective"] == "binary"


def test_legacy_effective_config_hash_invariance_to_current_code():
    h1 = compute_legacy_effective_model_config_hash()
    assert len(h1) == 64
    
    # 再次读取，哈希完全一致
    h2 = compute_legacy_effective_model_config_hash()
    assert h1 == h2


def test_altered_historical_source_fixture_changes_hash():
    base_eff = resolve_historical_effective_configs("e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48")
    base_hash = compute_legacy_effective_model_config_hash(base_eff)
    
    import copy
    mod_eff = copy.deepcopy(base_eff)
    mod_eff["models"]["lightgbm_ranker"]["effective_params"]["learning_rate"] = 0.099
    
    mod_hash = compute_legacy_effective_model_config_hash(mod_eff)
    assert mod_hash != base_hash
