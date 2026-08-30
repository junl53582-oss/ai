"""
Historical Source Configuration AST Reader (research_v2/provenance/source_config_reader.py)
纯只读 AST 解析历史提交的 Python 配置与模型构造函数，绝不执行历史代码、不 import 历史文件。
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
    - lightgbm_clf_baseline -> LGBM_PARAMS_CLF (objective=binary)
    - lightgbm_ranker -> LGBM_PARAMS (objective=regression, effective estimator LGBMRanker with regression objective)
    - lightgbm_reg_baseline -> LGBM_PARAMS (objective=regression)
    - double_ensemble -> submodels=5, subspace=20
    """
    settings_code = read_source_file_from_git(source_commit, "config/settings.py", project_root)
    settings_dict = parse_settings_from_source(settings_code)

    lgbm_clf_params = settings_dict.get("LGBM_PARAMS_CLF", {})
    lgbm_reg_params = settings_dict.get("LGBM_PARAMS", {})

    # 历史 Ranker 调用路径推导:
    # 在 e6da4a2 的 models/lightgbm_model.py 中:
    # if task_type == 'classification': self.params = settings.LGBM_PARAMS_CLF.copy()
    # else: self.params = settings.LGBM_PARAMS.copy()
    # 且 fit() 中: if 'objective' not in fit_params: fit_params['objective'] = 'lambdarank'
    # 因为 fit_params 继承了 LGBM_PARAMS 中的 objective='regression'，因此未被覆盖！
    ranker_effective_params = lgbm_reg_params.copy()
    ranker_effective_params["top_k_features"] = 20
    ranker_effective_params["feature_selection_method"] = "rank_ic_pruned"
    ranker_effective_params["weighting_mode"] = "recency_magnitude"
    ranker_effective_params["effective_estimator_class"] = "LGBMRanker"
    ranker_effective_params["ranking_group_supplied"] = True
    ranker_effective_params["relevance_label"] = "daily_ordinal_0_to_4"
    ranker_effective_params["true_lambdarank_certified"] = False

    double_ensemble_params = {
        "n_submodels": 5,
        "subspace_features": 20,
        "sample_decay": 0.95,
        "reweight_factor": 2.0,
        "base_model": "lightgbm_clf"
    }

    protocol_config = {
        "train_window_years": float(settings_dict.get("TRAIN_WINDOW_YEARS", 1.5)),
        "val_window_months": int(settings_dict.get("VAL_WINDOW_MONTHS", 3)),
        "test_window_months": int(settings_dict.get("TEST_WINDOW_MONTHS", 2)),
        "purge_gap_days": int(settings_dict.get("PURGE_GAP_DAYS", 25)),
        "label_horizon": int(settings_dict.get("LABEL_HORIZON", 20)),
        "label_threshold_mode": str(settings_dict.get("LABEL_THRESHOLD_MODE", "cross_sectional_extreme")),
        "label_extreme_quantile": float(settings_dict.get("LABEL_EXTREME_QUANTILE", 0.30))
    }

    effective_configs = {
        "source_commit": source_commit,
        "protocol": protocol_config,
        "models": {
            "lightgbm_clf_baseline": {
                "reported_name": "LightGBM Classification (Baseline)",
                "task_type": "classification",
                "effective_estimator_class": "LGBMClassifier",
                "effective_objective": lgbm_clf_params.get("objective", "binary"),
                "effective_metric": lgbm_clf_params.get("metric", ["binary_logloss", "auc"]),
                "effective_params": lgbm_clf_params,
                "feature_selection": "all",
                "weighting_mode": "none"
            },
            "lightgbm_ranker": {
                "reported_name": "LightGBM Ranker (LambdaRank)",
                "legacy_model_id": "legacy_ordinal_ranker",
                "task_type": "ranking",
                "effective_estimator_class": "LGBMRanker",
                "effective_objective": ranker_effective_params.get("objective", "regression"),
                "effective_metric": ranker_effective_params.get("metric", "rmse"),
                "effective_params": ranker_effective_params,
                "feature_selection": "rank_ic_pruned",
                "weighting_mode": "recency_magnitude",
                "true_lambdarank_certified": False
            },
            "lightgbm_reg_baseline": {
                "reported_name": "LightGBM Regression",
                "task_type": "regression",
                "effective_estimator_class": "LGBMRegressor",
                "effective_objective": lgbm_reg_params.get("objective", "regression"),
                "effective_metric": lgbm_reg_params.get("metric", "rmse"),
                "effective_params": lgbm_reg_params,
                "feature_selection": "all",
                "weighting_mode": "recency_magnitude"
            },
            "double_ensemble": {
                "reported_name": "DoubleEnsemble (Sample Reweight + Subspacing)",
                "task_type": "classification",
                "effective_estimator_class": "DoubleEnsembleQuantModel",
                "effective_objective": "binary",
                "effective_params": double_ensemble_params,
                "feature_selection": "top_20",
                "weighting_mode": "recency_magnitude"
            }
        }
    }
    return effective_configs


def compute_legacy_effective_model_config_hash(effective_configs: Optional[Dict[str, Any]] = None) -> str:
    """计算真实历史有效模型配置的 SHA256"""
    if effective_configs is None:
        effective_configs = resolve_historical_effective_configs()
    raw_json = json.dumps(effective_configs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
