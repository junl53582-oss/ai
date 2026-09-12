"""全历史业绩预告拉取 (2018 起, 32 个报告期, 按期缓存可断点续跑)

东财 datacenter RPT_PUBLIC_OP_PREDICT: 公告日早于正式财报 1-2 个月 (PIT 增量源)。
页间限速 2.5s (东财封禁教训)。日志: artifacts/forecast_fetch.log
"""
import io
import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = ROOT / 'artifacts' / 'forecast_fetch.log'


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    from data.fundamentals_forecast import fetch_all_forecast_reports, forecast_cache_dir
    log(">>> 业绩预告全历史拉取开始 (2018+)")
    n = fetch_all_forecast_reports(start_year='2018')
    files = sorted(forecast_cache_dir().glob('forecast_*.parquet'))
    total_rows = 0
    import pandas as pd
    for f in files:
        try:
            total_rows += len(pd.read_parquet(f, columns=['symbol']))
        except Exception:
            pass
    log(f"<<< 完成: 新增 {n} 期 | 缓存 {len(files)} 期 | 累计 {total_rows:,} 条预告")
    return 0


if __name__ == '__main__':
    sys.exit(main())
