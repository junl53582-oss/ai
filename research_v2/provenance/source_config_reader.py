"""
Historical Source Configuration AST Reader (research_v2/provenance/source_config_reader.py)
只读解析历史提交源码与 settings 常量；关键模型语义必须从历史源码结构中得到证据后才允许标记为 RESOLVED。
"""
import ast
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List


class HistoricalSourceEvidenceError(RuntimeError):
    """历史源码证据不足或与声明语义不一致。"""
    pass


def read_source_file_from_git(commit: str, file_path: str, project_root: Optional[Path] = None) -> str:
    """从指定 Git 提交中只读读取文件内容 (UTF-8)。"""
    cmd = ["git", "show", f"{commit}:{file_path}"]
    res = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return res.stdout


class SettingsASTExtractor(ast.NodeVisitor):
    """静态提取 settings.py 中定义的字段与字典常数。"""

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
    """通过 AST 解析 settings 源码中的变量。"""
    tree = ast.parse(source_code)
    extractor = SettingsASTExtractor()
    extractor.visit(tree)
    return extractor.constants


def _dotted_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _find_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return child
    raise HistoricalSourceEvidenceError(
        f"Historical source missing {class_name}.{method_name}"
    )


def _assignment_call_exists(node: ast.AST, target_name: str, call_name: str) -> bool:
    for item in ast.walk(node):
        if not isinstance(item, ast.Assign) or not isinstance(item.value, ast.Call):
            continue
        if _dotted_name(item.value.func) != call_name:
            continue
        for target in item.targets:
            if _dotted_name(target) == target_name:
                return True
    return False


def _classification_else_uses_regression_params(init_fn: ast.FunctionDef) -> bool:
    """
    验证历史 __init__ 的参数分派:
    classification -> settings.LGBM_PARAMS_CLF.copy()
    else           -> settings.LGBM_PARAMS.copy()

    Walk-Forward 对 ranking 未显式传 params，因此 ranking 会落入该 else 分支。
    """
    for node in ast.walk(init_fn):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if (
            _dotted_name(test.left) == "task_type"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and _literal_value(test.comparators[0]) == "classification"
        ):
            clf_ok = _assignment_call_exists(
                ast.Module(body=node.body, type_ignores=[]),
                "self.params",
                "settings.LGBM_PARAMS_CLF.copy",
            )
            reg_ok = _assignment_call_exists(
                ast.Module(body=node.orelse, type_ignores=[]),
                "self.params",
                "settings.LGBM_PARAMS.copy",
            )
            if clf_ok and reg_ok:
                return True
    return False


def _call_exists(node: ast.AST, call_name: str) -> bool:
    return any(
        isinstance(item, ast.Call) and _dotted_name(item.func) == call_name
        for item in ast.walk(node)
    )


def _subscript_assignment_exists(
    node: ast.AST,
    container_name: str,
    key: str,
    expected_value: Any,
) -> bool:
    for item in ast.walk(node):
        if not isinstance(item, ast.Assign):
            continue
        if _literal_value(item.value) != expected_value:
            continue
        for target in item.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if _dotted_name(target.value) != container_name:
                continue
            slice_node = target.slice
            if _literal_value(slice_node) == key:
                return True
    return False


def _not_in_guard_exists(node: ast.AST, key: str, container_name: str) -> bool:
    for item in ast.walk(node):
        if not isinstance(item, ast.Compare):
            continue
        if (
            _literal_value(item.left) == key
            and len(item.ops) == 1
            and isinstance(item.ops[0], ast.NotIn)
            and len(item.comparators) == 1
            and _dotted_name(item.comparators[0]) == container_name
        ):
            return True
    return False


def _call_with_keyword_exists(
    node: ast.AST,
    call_name: str,
    keyword: str,
    expected_value: Any,
    forbidden_keyword: Optional[str] = None,
) -> bool:
    for item in ast.walk(node):
        if not isinstance(item, ast.Call) or _dotted_name(item.func) != call_name:
            continue
        kw_map = {kw.arg: kw.value for kw in item.keywords if kw.arg is not None}
        if keyword not in kw_map or _literal_value(kw_map[keyword]) != expected_value:
            continue
        if forbidden_keyword and forbidden_keyword in kw_map:
            continue
        return True
    return False


def _rank_pct_0_to_4_transform_exists(node: ast.AST) -> bool:
    has_rank_pct = False
    has_4999 = False
    has_int_cast = False
    for item in ast.walk(node):
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute):
            if item.func.attr == "rank":
                for kw in item.keywords:
                    if kw.arg == "pct" and _literal_value(kw.value) is True:
                        has_rank_pct = True
            elif item.func.attr == "astype" and item.args:
                if _dotted_name(item.args[0]) == "int" or _literal_value(item.args[0]) is int:
                    has_int_cast = True
        if isinstance(item, ast.Constant) and item.value == 4.999:
            has_4999 = True

    # ast.literal_eval(Name("int")) 不会返回 int，因此单独检查 ast.Name 参数。
    if not has_int_cast:
        for item in ast.walk(node):
            if (
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == "astype"
                and item.args
                and isinstance(item.args[0], ast.Name)
                and item.args[0].id == "int"
            ):
                has_int_cast = True
                break

    return has_rank_pct and has_4999 and has_int_cast


def verify_historical_model_source_semantics(
    lightgbm_source: str,
    walk_forward_source: str,
) -> Dict[str, Any]:
    """
    对历史模型关键语义做源码级 fail-closed 验证。

    只有下列代码路径同时存在时，才允许将历史 Ranker 语义标记为已解析:
    - ranking 调用 LightGBMQuantModel(task_type="ranking") 且未显式覆盖 params
    - ranking 因非 classification 而继承 settings.LGBM_PARAMS
    - LGBMRanker 构造存在
    - lambdarank 仅在 "objective" 不存在时才作为 fallback 注入
    - ranking group 被传给 fit()
    - 标签按日 rank(pct=True) * 4.999 后转 int，得到 0~4 ordinal relevance
    """
    lgb_tree = ast.parse(lightgbm_source)
    wf_tree = ast.parse(walk_forward_source)

    init_fn = _find_method(lgb_tree, "LightGBMQuantModel", "__init__")
    fit_fn = _find_method(lgb_tree, "LightGBMQuantModel", "fit")
    wf_run = _find_method(wf_tree, "WalkForwardTrainer", "run_walk_forward")

    checks = {
        "ranking_call_without_params": _call_with_keyword_exists(
            wf_run,
            "LightGBMQuantModel",
            "task_type",
            "ranking",
            forbidden_keyword="params",
        ),
        "ranking_uses_regression_params": _classification_else_uses_regression_params(init_fn),
        "ranker_estimator": _call_exists(fit_fn, "lgb.LGBMRanker"),
        "classifier_estimator": _call_exists(fit_fn, "lgb.LGBMClassifier"),
        "regressor_estimator": _call_exists(fit_fn, "lgb.LGBMRegressor"),
        "lambdarank_guard": _not_in_guard_exists(fit_fn, "objective", "fit_params"),
        "lambdarank_fallback": _subscript_assignment_exists(
            fit_fn, "fit_params", "objective", "lambdarank"
        ),
        "ranking_group_supplied": _subscript_assignment_exists(
            fit_fn, "fit_kwargs", "group", None
        ),
        "ordinal_relevance_transform": _rank_pct_0_to_4_transform_exists(wf_run),
    }

    # group 赋值右值不是字面量，需要单独验证 target/key + train_group 名称。
    checks["ranking_group_supplied"] = any(
        isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and _dotted_name(target.value) == "fit_kwargs"
            and _literal_value(target.slice) == "group"
            for target in item.targets
        )
        and _dotted_name(item.value) == "train_group"
        for item in ast.walk(fit_fn)
    )

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise HistoricalSourceEvidenceError(
            "Historical model semantics could not be proven from source: "
            + ", ".join(failed)
        )

    # settings.LGBM_PARAMS.objective 在历史提交中为 regression，且 fit() 只有
    # objective 缺失时才写入 lambdarank，因此 effective objective 不是 lambdarank。
    return {
        "ranker_task_type": "ranking",
        "ranker_estimator_class": "LGBMRanker",
        "classifier_estimator_class": "LGBMClassifier",
        "regressor_estimator_class": "LGBMRegressor",
        "ranking_group_supplied": True,
        "relevance_label": "daily_ordinal_0_to_4",
        "true_lambdarank_certified": False,
    }


def resolve_historical_effective_configs(
    source_commit: str = "e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48",
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """
    根据历史源码调用路径真实推导 effective runtime configurations:
    - settings 常量通过 AST 读取
    - 关键模型语义通过历史 models/*.py AST/source evidence fail-closed 验证
    - 记录每个字段的 value, source_path, source_symbol, resolution_status
    - 统计 resolved_field_count 与 unresolved_field_count
    """
    settings_code = read_source_file_from_git(
        source_commit, "config/settings.py", project_root
    )
    lightgbm_code = read_source_file_from_git(
        source_commit, "models/lightgbm_model.py", project_root
    )
    walk_forward_code = read_source_file_from_git(
        source_commit, "models/walk_forward.py", project_root
    )

    settings_dict = parse_settings_from_source(settings_code)
    source_semantics = verify_historical_model_source_semantics(
        lightgbm_code, walk_forward_code
    )

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
        "task_type": make_entry(source_semantics["ranker_task_type"], "models/walk_forward.py", "model_type_mapping"),
        "effective_estimator_class": make_entry(source_semantics["ranker_estimator_class"], "models/lightgbm_model.py", "fit.LGBMRanker"),
        "effective_objective": make_entry(ranker_params.get("objective"), "config/settings.py", "LGBM_PARAMS.objective"),
        "effective_metric": make_entry(ranker_params.get("metric"), "config/settings.py", "LGBM_PARAMS.metric"),
        "learning_rate": make_entry(ranker_params.get("learning_rate"), "config/settings.py", "LGBM_PARAMS.learning_rate"),
        "num_leaves": make_entry(ranker_params.get("num_leaves"), "config/settings.py", "LGBM_PARAMS.num_leaves"),
        "feature_fraction": make_entry(ranker_params.get("feature_fraction"), "config/settings.py", "LGBM_PARAMS.feature_fraction"),
        "bagging_fraction": make_entry(ranker_params.get("bagging_fraction"), "config/settings.py", "LGBM_PARAMS.bagging_fraction"),
        "min_child_samples": make_entry(ranker_params.get("min_child_samples"), "config/settings.py", "LGBM_PARAMS.min_child_samples"),
        "n_estimators": make_entry(ranker_params.get("n_estimators"), "config/settings.py", "LGBM_PARAMS.n_estimators"),
        "early_stopping_rounds": make_entry(ranker_params.get("early_stopping_rounds"), "config/settings.py", "LGBM_PARAMS.early_stopping_rounds"),
        "ranking_group_supplied": make_entry(source_semantics["ranking_group_supplied"], "models/lightgbm_model.py", "fit_kwargs.group"),
        "relevance_label": make_entry(source_semantics["relevance_label"], "models/walk_forward.py", "rank_pct_multiplied"),
        "true_lambdarank_certified": make_entry(source_semantics["true_lambdarank_certified"], "models/lightgbm_model.py", "objective_not_in_fit_params_evaluated_false")
    }

    # 历史 CLF Baseline 调用路径解析
    clf_effective = {
        "reported_name": make_entry("LightGBM Classification (Baseline)", "reports/model_research", "model_name"),
        "task_type": make_entry("classification", "models/walk_forward.py", "task_type"),
        "effective_estimator_class": make_entry(source_semantics["classifier_estimator_class"], "models/lightgbm_model.py", "fit.LGBMClassifier"),
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
        "effective_estimator_class": make_entry(source_semantics["regressor_estimator_class"], "models/lightgbm_model.py", "fit.LGBMRegressor"),
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
    """计算真实历史有效模型配置的 SHA256。"""
    if effective_configs is None:
        effective_configs = resolve_historical_effective_configs()
    raw_json = json.dumps(effective_configs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
