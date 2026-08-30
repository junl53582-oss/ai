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
    assert eff["config_resolution_status"] == "FULLY_RESOLVED"
    assert eff["resolved_field_count"] > 0
    assert eff["unresolved_field_count"] == 0

    models = eff["models"]
    ranker = models["lightgbm_ranker"]
    assert ranker["effective_estimator_class"]["value"] == "LGBMRanker"
    assert ranker["effective_objective"]["value"] == "regression"
    assert ranker["effective_metric"]["value"] == "rmse"
    assert ranker["true_lambdarank_certified"]["value"] is False

    clf = models["lightgbm_clf_baseline"]
    assert clf["effective_objective"]["value"] == "binary"


def test_ast_parser_fixture_sensitivity():
    code_fixture_1 = """
LGBM_PARAMS = {"objective": "regression", "learning_rate": 0.03}
LGBM_PARAMS_CLF = {"objective": "binary", "learning_rate": 0.02}
"""
    code_fixture_2 = """
LGBM_PARAMS = {"objective": "regression", "learning_rate": 0.05}
LGBM_PARAMS_CLF = {"objective": "binary", "learning_rate": 0.02}
"""
    p1 = parse_settings_from_source(code_fixture_1)
    p2 = parse_settings_from_source(code_fixture_2)
    assert p1["LGBM_PARAMS"]["learning_rate"] == 0.03
    assert p2["LGBM_PARAMS"]["learning_rate"] == 0.05
    assert p1 != p2
