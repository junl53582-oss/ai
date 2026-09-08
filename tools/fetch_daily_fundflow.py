"""每日资金流数据积累器: 拉取全池标的的资金流日K历史 (~120 交易日), 落盘 parquet

幂等: 当日文件已存在则跳过。产物: data_storage/fundflow/fundflow_YYYYMMDD.parquet
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


def main() -> int:
    today = datetime.now().strftime("%Y%m%d")
    out_path = settings.DATA_DIR / "fundflow" / f"fundflow_{today}.parquet"
    if out_path.exists():
        print(f"[skip] 今日资金流文件已存在: {out_path}")
        return 0

    matrix_path = settings.DATA_DIR / "research" / "factor_matrix_300.parquet"
    symbols = sorted(pd.read_parquet(matrix_path, columns=["symbol"])["symbol"].unique())
    print(f">>> 资金流积累: {len(symbols)} 标的 | 目标: {out_path.name}", flush=True)

    frames, ok_count = [], 0
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        hist = fetch_em_fundflow_history(sym)
        if hist:
            df = pd.DataFrame(hist)
            df["symbol"] = sym
            frames.append(df)
            ok_count += 1
        if i % 50 == 0:
            print(f"  [{i}/{len(symbols)}] 成功 {ok_count} | {time.time()-t0:.0f}s", flush=True)
        time.sleep(0.15)

    if not frames:
        print("[-] 全部失败, 不落盘", flush=True)
        return 1

    full = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    coverage = ok_count / len(symbols) * 100
    print(f"[+] 完成: {ok_count}/{len(symbols)} 标的 ({coverage:.1f}%) | {len(full):,} 行 | "
          f"{time.time()-t0:.0f}s | 落盘: {out_path}", flush=True)
    # 覆盖率门禁: < 85% 记警告 (数据仍保留, 但下游因子研究需知晓)
    if coverage < 85:
        print(f"[!] 警告: 覆盖率不足 85%, 资金流因子研究需按 symbol 可用性过滤", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
