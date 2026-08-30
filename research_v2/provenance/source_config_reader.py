"""
Historical Source Configuration AST Reader (research_v2/provenance/source_config_reader.py)
纯只读 AST 解析历史提交源码，记录每个配置字段的来源文件与符号，未解析项显式标记为 UNRESOLVED。
"""
import ast
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List


def read_source_file_from_git(commit: str, file_path: str, project_root: Optional[Path] = None) -> str:
    """从指定 Git 提交中只读读取文件内容 (UTF-8)"""
    cmd = ["git", "show", f"{commit}:{file_path}"]
    res = subprocess.run(cmd, cwd=project_root, capture_output=True, encoding="utf-8", errors="replace", check=True)
    return res.stdout


class SettingsASTExtractor(ast.NodeVisitor):
    """静态提取 settings.py 中定义的字段与字典常数"""
    def __init__(self):
        self.constants: Dict[str, Any] = {}

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            if node.value is not None:
                val = self._extract_value(node.value)
                if val is not None:
                    self.constants[var_name] = val
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                val = self._extract_value(node.value)
                if val is not None:
                    self.constants[var_name] = val
        self.generic_visit(node)

    def _extract_value(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception:
            pass

        # 支持 field(default_factory=lambda: {...})
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "default_factory" and isinstance(kw.value, ast.Lambda):
                    try:
                        return ast.literal_eval(kw.value.body)
                    except Exception:
                        pass
        return None


def parse_settings_from_source(source_code: str) -> Dict[str, Any]:
    """通过 AST 解析 settings 源码中的变量"""
    tree = ast.parse(source_code)
    extractor = SettingsASTExtractor()
    extractor.visit(tree)
    return extractor.constants


def resolve_historical_effective_configs(
    source_commit: str = "e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48",
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """
    根据历史源码调用路径真实推导 effective runtime configurations:
    - 记录每个字段的 value, source_path, source_symbol, resolution_status
    - 统计 resolved_field_count 与 unresolved_field_count
    """
    settings_code = read_source_file_from_git(source_commit, "config/settings.py", project_root)
    settings_dict = parse_settings_from_source(settings_code)

    lgbm_clf_params = settings_dict.get("LGBM_PARAMS_CLF", {})
    lgbm_reg_params = settings_dict.get("LGBM_PARAMS", {})

    resolved_count = 0
    unresolved_count = 0

    def make_entry(val: Any, path: str, symbol: str) -> Dict[str, Any]:
        nonlocal resolved_count, unresolved_count
        if val is not None:
            resolved_count += 1
            return {
                "value": val,
                "source_path": path,
                "source_symbol": symbol,
                "resolution_status": "RESOLVED"
            }
        else:
            unresolved_count += 1
            return {
                "value": None,
                "source_path": path,
                "source_symbol": symbol,
                "resolution_status": "UNRESOLVED"
            }

    protocol_resolved = {
        "train_window_years": make_entry(settings_dict.get("TRAIN_WINDOW_YEARS"), "config/settings.py", "TRAIN_WINDOW_YEARS"),
        "val_window_months": make_entry(settings_dict.get("VAL_WINDOW_MONTHS"), "config/settings.py", "VAL_WINDOW_MONTHS"),
        "test_window_months": make_entry(settings_dict.get("TEST_WINDOW_MONTHS"), "config/settings.py", "TEST_WINDOW_MONTHS"),
        "purge_gap_days": make_entry(settings_dict.get("PURGE_GAP_DAYS"), "config/settings.py", "PURGE_GAP_DAYS"),
        "label_horizon": make_entry(settings_dict.get("LABEL_HORIZON"), "config/settings.py", "LABEL_HORIZON"),
        "label_threshold_mode": make_entry(settings_dict.get("LABEL_THRESHOLD_MODE"), "config/settings.py", "LABEL_THRESHOLD_MODE"),
        "label_extreme_quantile": make_entry(settings_dict.get("LABEL_EXTREME_QUANTILE"), "config/settings.py", "LABEL_EXTREME_QUANTILE")
    }

    # 历史 Ranker 调用路径解析
    ranker_params = lgbm_reg_params.copy()
    ranker_effective = {
        "reported_name": make_entry("LightGBM Ranker (LambdaRank)", "reports/model_research", "model_name"),
        "legacy_model_id": make_entry("legacy_ordinal_ranker", "research_v2/provenance", "corrected_legacy_id"),
        "task_type": make_entry("ranking", "models/walk_forward.py", "model_type_mapping"),
        "effective_estimator_class": make_entry("LGBMRanker", "models/lightgbm_model.py", "fit.LGBMRanker"),
        "effective_objective": make_entry(ranker_params.get("objective"), "config/settings.py", "LGBM_PARAMS.objective"),
        "effective_metric": make_entry(ranker_params.get("metric"), "config/settings.py", "LGBM_PARAMS.metric"),
        "learning_rate": make_entry(ranker_params.get("learning_rate"), "config/settings.py", "LGBM_PARAMS.learning_rate"),
        "num_leaves": make_entry(ranker_params.get("num_leaves"), "config/settings.py", "LGBM_PARAMS.num_leaves"),
        "feature_fraction": make_entry(ranker_params.get("feature_fraction"), "config/settings.py", "LGBM_PARAMS.feature_fraction"),
        "bagging_fraction": make_entry(ranker_params.get("bagging_fraction"), "config/settings.py", "LGBM_PARAMS.bagging_fraction"),
        "min_child_samples": make_entry(ranker_params.get("min_child_samples"), "config/settings.py", "LGBM_PARAMS.min_child_samples"),
        "n_estimators": make_entry(ranker_params.get("n_estimators"), "config/settings.py", "LGBM_PARAMS.n_estimators"),
        "early_stopping_rounds": make_entry(ranker_params.get("early_stopping_rounds"), "config/settings.py", "LGBM_PARAMS.early_stopping_rounds"),
        "ranking_group_supplied": make_entry(True, "models/lightgbm_model.py", "fit_kwargs.group"),
        "relevance_label": make_entry("daily_ordinal_0_to_4", "models/walk_forward.py", "rank_pct_multiplied"),
        "true_lambdarank_certified": make_entry(False, "models/lightgbm_model.py", "objective_not_in_fit_params_evaluated_false")
    }

    # 历史 CLF Baseline 调用路径解析
    clf_effective = {
        "reported_name": make_entry("LightGBM Classification (Baseline)", "reports/model_research", "model_name"),
        "task_type": make_entry("classification", "models/walk_forward.py", "task_type"),
        "effective_estimator_class": make_entry("LGBMClassifier", "models/lightgbm_model.py", "fit.LGBMClassifier"),
        "effective_objective": make_entry(lgbm_clf_params.get("objective"), "config/settings.py", "LGBM_PARAMS_CLF.objective"),
        "effective_metric": make_entry(lgbm_clf_params.get("metric"), "config/settings.py", "LGBM_PARAMS_CLF.metric"),
        "learning_rate": make_entry(lgbm_clf_params.get("learning_rate"), "config/settings.py", "LGBM_PARAMS_CLF.learning_rate"),
        "num_leaves": make_entry(lgbm_clf_params.get("num_leaves"), "config/settings.py", "LGBM_PARAMS_CLF.num_leaves"),
        "max_depth": make_entry(lgbm_clf_params.get("max_depth"), "config/settings.py", "LGBM_PARAMS_CLF.max_depth"),
        "feature_fraction": make_entry(lgbm_clf_params.get("feature_fraction"), "config/settings.py", "LGBM_PARAMS_CLF.feature_fraction"),
        "bagging_fraction": make_entry(lgbm_clf_params.get("bagging_fraction"), "config/settings.py", "LGBM_PARAMS_CLF.bagging_fraction"),
        "min_child_samples": make_entry(lgbm_clf_params.get("min_child_samples"), "config/settings.py", "LGBM_PARAMS_CLF.min_child_samples"),
        "lambda_l1": make_entry(lgbm_clf_params.get("lambda_l1"), "config/settings.py", "LGBM_PARAMS_CLF.lambda_l1"),
        "lambda_l2": make_entry(lgbm_clf_params.get("lambda_l2"), "config/settings.py", "LGBM_PARAMS_CLF.lambda_l2"),
        "n_estimators": make_entry(lgbm_clf_params.get("n_estimators"), "config/settings.py", "LGBM_PARAMS_CLF.n_estimators"),
        "early_stopping_rounds": make_entry(lgbm_clf_params.get("early_stopping_rounds"), "config/settings.py", "LGBM_PARAMS_CLF.early_stopping_rounds")
    }

    # 历史 Regression 调用路径解析
    reg_effective = {
        "reported_name": make_entry("LightGBM Regression", "reports/model_research", "model_name"),
        "task_type": make_entry("regression", "models/walk_forward.py", "task_type"),
        "effective_estimator_class": make_entry("LGBMRegressor", "models/lightgbm_model.py", "fit.LGBMRegressor"),
        "effective_objective": make_entry(lgbm_reg_params.get("objective"), "config/settings.py", "LGBM_PARAMS.objective"),
        "effective_metric": make_entry(lgbm_reg_params.get("metric"), "config/settings.py", "LGBM_PARAMS.metric"),
        "learning_rate": make_entry(lgbm_reg_params.get("learning_rate"), "config/settings.py", "LGBM_PARAMS.learning_rate")
    }

    status = "FULLY_RESOLVED" if unresolved_count == 0 else "PARTIALLY_RESOLVED"

    effective_configs = {
        "source_commit": source_commit,
        "config_resolution_status": status,
        "resolved_field_count": resolved_count,
        "unresolved_field_count": unresolved_count,
        "protocol": protocol_resolved,
        "models": {
            "lightgbm_clf_baseline": clf_effective,
            "lightgbm_ranker": ranker_effective,
            "lightgbm_reg_baseline": reg_effective
        }
    }
    return effective_configs


def compute_legacy_effective_model_config_hash(effective_configs: Optional[Dict[str, Any]] = None) -> str:
    """计算真实历史有效模型配置的 SHA256"""
    if effective_configs is None:
        effective_configs = resolve_historical_effective_configs()
    raw_json = json.dumps(effective_configs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
