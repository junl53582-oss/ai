"""
Tests for Source Config Reader (tests/test_phase2_1_source_config_reader.py)
"""
import json
from pathlib import Path

import pytest

from research_v2.provenance.source_config_reader import (
    HistoricalSourceEvidenceError,
    read_source_file_from_git,
    parse_settings_from_source,
    verify_historical_model_source_semantics,
    resolve_historical_effective_configs,
    compute_legacy_effective_model_config_hash,
)


SOURCE_COMMIT = "e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48"


def test_ast_parsing_of_historical_settings():
    code = read_source_file_from_git(SOURCE_COMMIT, "config/settings.py")
    settings_dict = parse_settings_from_source(code)

    assert "LGBM_PARAMS_CLF" in settings_dict
    assert "LGBM_PARAMS" in settings_dict
    assert settings_dict["LGBM_PARAMS_CLF"]["objective"] == "binary"
    assert settings_dict["LGBM_PARAMS"]["objective"] == "regression"
    assert settings_dict["TRAIN_WINDOW_YEARS"] == 1.5
    assert settings_dict["VAL_WINDOW_MONTHS"] == 3
    assert settings_dict["TEST_WINDOW_MONTHS"] == 2
    assert settings_dict["LABEL_HORIZON"] == 20


def test_historical_model_semantics_are_proven_from_source():
    lgb_code = read_source_file_from_git(
        SOURCE_COMMIT, "models/lightgbm_model.py"
    )
    wf_code = read_source_file_from_git(
        SOURCE_COMMIT, "models/walk_forward.py"
    )
    semantics = verify_historical_model_source_semantics(lgb_code, wf_code)

    assert semantics["ranker_task_type"] == "ranking"
    assert semantics["ranker_estimator_class"] == "LGBMRanker"
    assert semantics["ranking_group_supplied"] is True
    assert semantics["relevance_label"] == "daily_ordinal_0_to_4"
    assert semantics["true_lambdarank_certified"] is False


def test_historical_source_semantics_fail_closed_if_guard_changes():
    lgb_code = read_source_file_from_git(
        SOURCE_COMMIT, "models/lightgbm_model.py"
    )
    wf_code = read_source_file_from_git(
        SOURCE_COMMIT, "models/walk_forward.py"
    )

    mutated = lgb_code.replace(
        'if "objective" not in fit_params:',
        'if False:',
        1,
    )
    assert mutated != lgb_code

    with pytest.raises(
        HistoricalSourceEvidenceError,
        match="lambdarank_guard",
    ):
        verify_historical_model_source_semantics(mutated, wf_code)


def test_effective_ranker_config_resolution():
    eff = resolve_historical_effective_configs(SOURCE_COMMIT)
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


def test_effective_config_hash_matches_frozen_baseline():
    eff = resolve_historical_effective_configs(SOURCE_COMMIT)
    actual = compute_legacy_effective_model_config_hash(eff)
    hashes = json.loads(
        Path(
            "reports/baselines/legacy_v1/artifact_hashes.json"
        ).read_text(encoding="utf-8")
    )
    expected = hashes["source_hashes"]["legacy_effective_model_config_hash"]
    assert actual == expected


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
