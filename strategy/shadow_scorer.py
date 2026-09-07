"""影子模型打分器 (Shadow Scorer) — P1 模型可信化核心组件

目的: 让新旧模型同台竞技。旧生产模型 (m_20260903, 训练于已证伪的坏数据集) 的
pred_score 仍在官方清单中展示, 但其科学性已被今日研究推翻。本模块用【已定架构基线】
(lightgbm_ranker + Train-Only 折内滚动筛选) 对同一批标的独立打分, 作为"影子分"
与旧分并排展示, 由影子账本的前瞻表现决定谁退役。

科学纪律 (与 2026-09-07 滚动筛选实验完全一致):
- 训练切片 = 数据起点 → (矩阵最新交易日 - PURGE_GAP_DAYS 个交易日), 只用过去
- 因子选择 = FoldFeatureSelector 仅在训练切片上执行 (Train-Only, 无跨折泄漏)
- 标签 = label_up_down_20d (LabelRegistry v2), 与官方体系一致
- Fail-Closed: 任何一步失败即返回空影子分与原因, 绝不编造分数
"""
import io
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.settings import settings
from research_v2.labels.label_registry import LabelRegistry
from models.fold_feature_selector import FoldFeatureSelector
from models.lightgbm_model import LightGBMQuantModel

logger = logging.getLogger(__name__)

LABEL_COL = 'label_up_down_20d'
MATRIX_PATH = PROJECT_ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet'
POOL_FILE = PROJECT_ROOT / 'reports' / 'model_research' / 'candidate_pool_top40.txt'
CACHE_FILE = PROJECT_ROOT / 'data_storage' / 'cache' / 'shadow_scores_latest.json'
NON_FEATURE_COLS = {
    'date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount',
    'outstanding_share', 'pct_change', 'adj_open', 'adj_high', 'adj_low',
    'adj_close', 'adj_pct_change', 'data_source', 'adjustment_mode', 'name',
    'industry', 'board', 'current_is_st', 'st_status_known',
    'historical_st_rule_applied', 'is_st', 'is_st_unknown',
    'excluded_from_training', 'is_nontradable', 'is_subnew', 'is_suspended',
    'pre_close', 'limit_up_ratio', 'limit_down_ratio', 'limit_up_price',
    'limit_down_price', 'price_limit_rule_id', 'is_limit_up', 'is_limit_down',
    'is_limit_up_locked', 'is_limit_down_locked', 'circ_mv', 'circ_mv_raw',
    'benchmark_open', 'benchmark_close', 'benchmark_pct_change',
    'in_universe', LABEL_COL,
}

# 进程级缓存: 同一数据日只训练一次 (键 = 矩阵末行日期)
_CACHE: Dict[str, Any] = {}


def _load_pool(matrix: pd.DataFrame) -> list:
    if POOL_FILE.exists():
        pool = [x.strip() for x in POOL_FILE.read_text(encoding='utf-8').splitlines() if x.strip()]
        pool = [f for f in pool if f in matrix.columns]
        if pool:
            return pool
    return [c for c in matrix.columns
            if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(matrix[c])]


def _train_shadow_model(matrix: pd.DataFrame, top_k: int) -> Dict[str, Any]:
    raw = LabelRegistry.compute_label_v2(matrix, horizon=settings.LABEL_HORIZON)
    matrix = matrix.copy()
    matrix[LABEL_COL] = (raw > 0).astype(float).mask(raw.isna(), np.nan)

    all_dates = pd.Series(matrix['date'].unique()).sort_values().reset_index(drop=True)
    cutoff = all_dates.iloc[max(0, len(all_dates) - 1 - int(settings.PURGE_GAP_DAYS))]
    train_df = matrix[(matrix['date'] < cutoff) & matrix.get('in_universe', True)]
    train_df = train_df[train_df[LABEL_COL].notna()]
    if len(train_df) < 10000:
        raise RuntimeError(f'影子训练样本不足: {len(train_df)}')

    pool = _load_pool(matrix)
    selector = FoldFeatureSelector(top_n=top_k)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # 静音常数列 spearman 警告 (选择器已自动跳过无效 IC)
        fold_feats, _ = selector.select_features(
            train_df=train_df, candidate_features=pool, label_col=LABEL_COL,
            method='rank_ic_pruned', strict_selection=False
        )
    if not fold_feats:
        raise RuntimeError('影子因子选择无结果')

    y_tr = (train_df.groupby('date')[LABEL_COL].rank(pct=True) * 4.999).astype(int)
    model = LightGBMQuantModel(task_type='ranking', strict_mode=False)
    X_tr = train_df[fold_feats].copy()
    X_tr['date'] = train_df['date']  # ranker 的 group 推导依赖 date 列 (模型特征仍由 feature_names 限定)
    model.fit(
        X_train=X_tr,
        y_train=y_tr,
        feature_names=fold_feats,
    )

    latest_date = all_dates.iloc[-1]
    latest_xs = matrix[matrix['date'] == latest_date]
    scored = latest_xs.copy()
    scored['shadow_raw'] = model.predict(scored[fold_feats])
    scored['shadow_score'] = scored.groupby('date')['shadow_raw'].rank(pct=True)
    shadow_map = dict(zip(scored['symbol'], scored['shadow_score']))

    return {
        'shadow_map': shadow_map,
        'factors': fold_feats,
        'train_rows': int(len(train_df)),
        'train_end': str(cutoff)[:10],
        'as_of': str(latest_date)[:10],
    }


def _load_disk_cache(cache_path: Path, as_of: str) -> Optional[Dict[str, Any]]:
    try:
        if cache_path.exists():
            d = json.loads(cache_path.read_text(encoding='utf-8'))
            if d.get('as_of') == as_of and d.get('scores'):
                return d
    except Exception as e:
        logger.warning(f'[ShadowScorer] 读取影子缓存失败: {e}')
    return None


def _save_disk_cache(cache_path: Path, info: Dict[str, Any]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'as_of': info['as_of'],
            'train_end': info['train_end'],
            'factors': info['factors'],
            'train_rows': info['train_rows'],
            'scores': {k: float(v) for k, v in info['shadow_map'].items()},
            'generated_at': datetime.now().isoformat(timespec='seconds'),
        }
        tmp = cache_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
        tmp.replace(cache_path)
        logger.info(f'[ShadowScorer] 当日影子分已落盘缓存: {cache_path}')
    except Exception as e:
        logger.warning(f'[ShadowScorer] 写影子缓存失败 (不影响本次结果): {e}')


def compute_shadow_scores(
    top_df: pd.DataFrame,
    matrix_path: Optional[Path] = None,
    top_k: int = 10,
    force_refresh: bool = False,
    cache_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """对官方清单标的计算影子模型分 (0-1 截面百分位)。

    缓存策略: 同一数据日只训练一次 — 进程内缓存 + 磁盘缓存 (data_storage/cache/)。
    每天首次打开约 1-3 分钟, 之后全天秒读 (含进程重启)。
    Fail-Closed: 任何异常不抛出, 返回原表 + meta['error'], 影子列留空。
    """
    meta: Dict[str, Any] = {'factors': [], 'as_of': None, 'from_cache': None}
    out = top_df.copy()
    out['shadow_score'] = np.nan

    path = Path(matrix_path) if matrix_path else MATRIX_PATH
    cpath = Path(cache_path) if cache_path else CACHE_FILE
    if not path.exists():
        meta['error'] = f'因子矩阵不存在: {path}'
        logger.warning(f'[ShadowScorer] {meta["error"]}')
        return out, meta

    try:
        matrix = pd.read_parquet(path)
        matrix['date'] = pd.to_datetime(matrix['date'])
        cache_key = str(matrix['date'].max())[:10]
        info = None
        if not force_refresh:
            if cache_key in _CACHE:
                info = _CACHE[cache_key]
                meta['from_cache'] = 'memory'
            else:
                disk = _load_disk_cache(cpath, cache_key)
                if disk:
                    info = {
                        'shadow_map': {k: float(v) for k, v in disk['scores'].items()},
                        'factors': disk.get('factors', []),
                        'train_rows': disk.get('train_rows'),
                        'train_end': disk.get('train_end'),
                        'as_of': disk.get('as_of'),
                    }
                    _CACHE[cache_key] = info
                    meta['from_cache'] = 'disk'
        if info is None:
            info = _train_shadow_model(matrix, top_k)
            info['from_cache'] = False
            meta['from_cache'] = False
            _CACHE.clear()
            _CACHE[cache_key] = info
            _save_disk_cache(cpath, info)

        out['shadow_score'] = out['symbol'].map(info['shadow_map'])
        meta.update({'factors': info['factors'], 'train_rows': info.get('train_rows'),
                     'train_end': info['train_end'], 'as_of': info['as_of']})
        logger.info(f"[ShadowScorer] 影子打分完成 ({'缓存' if meta['from_cache'] else '新训练'}): "
                    f"{int(out['shadow_score'].notna().sum())}/{len(out)} 标的 | 因子={info['factors']}")
    except Exception as e:
        meta['error'] = f'{type(e).__name__}: {e}'
        logger.warning(f'[ShadowScorer] 影子打分 Fail-Closed: {meta["error"]}')

    return out, meta
