"""每日资金流数据积累器 (增量模式): 拉取全池标的的资金流日K历史, 落盘 parquet

幂等+增量: 只抓当日文件中缺失的标的; 全部就绪则跳过。
产物: data_storage/fundflow/fundflow_YYYYMMDD.parquet
用途: 资金流因子研究的数据底座 (每日自动刷新链的第 5 步调用)
"""
import io
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import settings
from data.em_direct_api import fetch_em_fundflow_history
from data.sina_fundflow_api import fetch_sina_fundflow_history


def fetch_any_source(symbol: str):
    """资金流双源获取: 新浪为主 (家宽稳定可用), 东财兜底 (解封后自动恢复)"""
    hist = fetch_sina_fundflow_history(symbol, min_days=120)
    if hist:
        return hist
    return fetch_em_fundflow_history(symbol)


def main() -> int:
    today = datetime.now().strftime("%Y%m%d")
    out_path = settings.DATA_DIR / "fundflow" / f"fundflow_{today}.parquet"

    existing_df, existing_syms = None, set()
    if out_path.exists():
        try:
            existing_df = pd.read_parquet(out_path)
            existing_syms = set(existing_df["symbol"].unique())
            print(f"[增量] 已有 {len(existing_syms)} 标的, 补齐缺失...", flush=True)
        except Exception as e:
            print(f"[!] 读取已有文件失败, 将全量重抓: {e}", flush=True)

    matrix_path = settings.DATA_DIR / "research" / "factor_matrix_300.parquet"
    symbols = sorted(pd.read_parquet(matrix_path, columns=["symbol"])["symbol"].unique())
    todo = [s for s in symbols if s not in existing_syms]
    if not todo:
        print(f"[skip] 全部 {len(symbols)} 标的已就绪: {out_path.name}")
        return 0
    print(f">>> 资金流积累: 待抓 {len(todo)}/{len(symbols)} | 目标: {out_path.name}", flush=True)

    frames = [] if existing_df is None else [existing_df]
    ok_count = 0
    t0 = time.time()
    for i, sym in enumerate(todo, 1):
        hist = fetch_any_source(sym)
        if hist:
            d = pd.DataFrame(hist)
            d["symbol"] = sym
            frames.append(d)
            ok_count += 1
        if i % 50 == 0:
            print(f"  [{i}/{len(todo)}] 本轮成功 {ok_count} | {time.time()-t0:.0f}s", flush=True)
            # 检查点落盘 (防长任务中断丢失进度): 覆盖写当前进度
            try:
                _cur = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol", "date"])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                _cur.to_parquet(out_path, index=False)
            except Exception as e:
                print(f"  [!] 检查点落盘失败: {e}", flush=True)
        time.sleep(0.2)

    if not frames:
        print("[-] 全部失败, 不落盘", flush=True)
        return 1

    full = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol", "date"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    coverage = full["symbol"].nunique() / len(symbols) * 100
    print(f"[+] 完成: 本轮新增 {ok_count}/{len(todo)} | 总覆盖 {full['symbol'].nunique()}/{len(symbols)} "
          f"标的 ({coverage:.1f}%) | {len(full):,} 行 | {time.time()-t0:.0f}s | 落盘: {out_path}", flush=True)
    if coverage < 85:
        print("[!] 警告: 覆盖率不足 85%, 资金流因子研究需按 symbol 可用性过滤", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
