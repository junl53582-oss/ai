"""G2 期中考自动裁决器 (tools/g2_gate_verdict.py)

把"20 个交易日后自动裁决"从承诺变成代码:
  台账逐日积累 → 满 20 个交易日 → 自动出裁决书 (A/B/C 三赛道)
  指标: 平均 IC / ICIR / 胜率(IC>0 占比) / 最大回撤(价差累计曲线)
  判定: 晋级候选 / 观察 / 淘汰  → 裁决书落盘 JSON + Markdown, 面板读取展示

门槛 (预设启发式, 可在 G2_GATE_CRITERIA 调整):
  晋级候选: 平均IC >= 0.02 且 ICIR >= 2.0 且 胜率 >= 0.55
  观察:     平均IC > 0 (未同时达标)
  淘汰:     平均IC <= 0
每晚由 daily_reconciliation 自动调用; 幂等 (每次重算并覆盖最新裁决书)。
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# NOTE (2026-09-14): 禁止在 import 期替换 sys.stdout 对象 (GC 会关掉共享底层 buffer,
# 打红 pytest 全量套件 "I/O operation on closed file")。reconfigure 原地改编码, 安全。
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

LEDGER = ROOT / 'data_storage' / 'research' / 'shadow_reconciliation_ledger.csv'
VERDICT_DIR = ROOT / 'reports' / 'model_research'
MIN_DAYS = 20                       # 观察期交易日数 (G2 期中考门槛)
G2_GATE_CRITERIA = {'min_mean_ic': 0.02, 'min_icir': 2.0, 'min_win_rate': 0.55}
TRACK_LABELS = {'A': 'A赛道 (20日视野·纯量价)', 'B': 'B赛道 (40日视野·正交化基本面)',
                'C': 'C赛道 (40日视野·正交化基本面·+业绩预告)'}

logger = logging.getLogger(__name__)


def _max_drawdown(cum: pd.Series) -> float:
    """累计曲线最大回撤 (价差单位)"""
    if cum.empty:
        return 0.0
    running_max = cum.cummax()
    return float((cum - running_max).min())


def evaluate_g2_gate(ledger: pd.DataFrame, min_days: int = MIN_DAYS,
                     criteria: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """纯函数: 依据台账裁决三赛道 (可单测)。不足 min_days 返回 ready=False。"""
    crit = dict(G2_GATE_CRITERIA)
    if criteria:
        crit.update(criteria)
    n = len(ledger) if ledger is not None else 0
    out: Dict[str, Any] = {'ready': False, 'days': int(n), 'min_days': int(min_days),
                           'criteria': crit, 'tracks': {}}
    if ledger is None or ledger.empty or n < min_days:
        return out
    out['ready'] = True

    for tag, label in TRACK_LABELS.items():
        ic_col = f'ic_{tag}'
        sp_col = f'hi_lo_spread_{tag}'
        if ic_col not in ledger.columns:
            continue
        s = pd.to_numeric(ledger[ic_col], errors='coerce').dropna()
        if s.empty or len(s) < min_days:
            continue
        mean_ic = float(s.mean())
        std_ic = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        icir = float(mean_ic / (std_ic + 1e-12) * np.sqrt(252))
        win_rate = float((s > 0).mean())
        mdd = 0.0
        if sp_col in ledger.columns:
            cum = pd.to_numeric(ledger[sp_col], errors='coerce').dropna().cumsum()
            mdd = _max_drawdown(cum)
        if mean_ic >= crit['min_mean_ic'] and icir >= crit['min_icir'] and win_rate >= crit['min_win_rate']:
            verdict = '晋级候选'
        elif mean_ic > 0:
            verdict = '观察'
        else:
            verdict = '淘汰'
        out['tracks'][tag] = {'label': label, 'n_days': int(len(s)),
                              'mean_ic': round(mean_ic, 5),
                              'icir_annualized': round(icir, 3),
                              'win_rate': round(win_rate, 3),
                              'max_drawdown_spread': round(mdd, 4),
                              'verdict': verdict}

    # 相对排序 (按平均 IC) 与推荐
    ranked = sorted(out['tracks'].items(), key=lambda kv: kv[1]['mean_ic'], reverse=True)
    if ranked:
        out['ranking'] = [k for k, _ in ranked]
        best_tag, best = ranked[0]
        out['recommended'] = {'track': best_tag, 'mean_ic': best['mean_ic'],
                              'reason': f"观察期内平均 IC 最高 ({best['mean_ic']:+.4f})"}
        candidates = [k for k, v in out['tracks'].items() if v['verdict'] == '晋级候选']
        out['promotion_candidates'] = candidates
        if candidates:
            out['decision'] = f"晋级候选: {', '.join(candidates)} (最终晋级需人工确认)"
        else:
            out['decision'] = '无赛道达到晋级门槛 — 维持现役模型, 继续观察'
    return out


def _render_markdown(v: Dict[str, Any]) -> str:
    lines = [f"# G2 期中考裁决书", '',
             f"- 生成时间: {v.get('generated_at', datetime.now().isoformat(timespec='seconds'))}",
             f"- 观察期: {v.get('days')} / {v.get('min_days')} 个交易日",
             f"- 门槛: 平均IC >= {v['criteria']['min_mean_ic']} | ICIR >= {v['criteria']['min_icir']} | "
             f"胜率 >= {v['criteria']['min_win_rate']}", '']
    if not v.get('ready'):
        lines.append(f"**未到期** — 还需 {v.get('min_days', MIN_DAYS) - v.get('days', 0)} 个交易日, 暂不裁决。")
        return '\n'.join(lines)
    lines += ['| 赛道 | 平均IC | ICIR | 胜率 | 价差最大回撤 | 裁决 |', '|---|---|---|---|---|---|']
    for tag, t in v['tracks'].items():
        lines.append(f"| {t['label']} | {t['mean_ic']:+.4f} | {t['icir_annualized']:.2f} | "
                     f"{t['win_rate']*100:.1f}% | {t['max_drawdown_spread']:.4f} | **{t['verdict']}** |")
    lines += ['', f"**结论**: {v.get('decision')}"]
    if v.get('recommended'):
        lines.append(f"**推荐**: {v['recommended']['track']} 赛道 — {v['recommended']['reason']}")
    lines += ['', '> 观察期样本有限 (20 个交易日), 裁决为参考结论; 晋级须经人工确认并保留原模型回滚路径。',
              '> 本裁决书由自动系统按台账生成, 不构成投资建议。']
    return '\n'.join(lines)


def run_verdict(write: bool = True) -> Dict[str, Any]:
    """读取台账 → 裁决 → 落盘 (JSON + Markdown)"""
    if not LEDGER.exists():
        return {'ready': False, 'days': 0, 'reason': 'ledger_missing'}
    ledger = pd.read_csv(LEDGER, dtype={'snap_date': str})
    v = evaluate_g2_gate(ledger)
    v['generated_at'] = datetime.now().isoformat(timespec='seconds')
    if write:
        VERDICT_DIR.mkdir(parents=True, exist_ok=True)
        (VERDICT_DIR / 'g2_gate_verdict.json').write_text(
            json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')
        (VERDICT_DIR / 'g2_gate_verdict.md').write_text(_render_markdown(v), encoding='utf-8')
        logger.info(f"[G2] 裁决书已更新: days={v.get('days')}/{v.get('min_days')} ready={v.get('ready')}")
    return v


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    v = run_verdict(write=True)
    print(f"[G2] days={v.get('days')}/{v.get('min_days')} | ready={v.get('ready')}")
    if v.get('ready'):
        for tag, t in v['tracks'].items():
            print(f"  {t['label']}: IC {t['mean_ic']:+.4f} | ICIR {t['icir_annualized']:.2f} | "
                  f"胜率 {t['win_rate']*100:.1f}% | {t['verdict']}")
        print(f"  结论: {v.get('decision')}")
    else:
        print(f"  未到期, 还需 {v.get('min_days', MIN_DAYS) - v.get('days', 0)} 个交易日")
    return 0


if __name__ == '__main__':
    sys.exit(main())
