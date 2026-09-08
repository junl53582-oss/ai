"""全量基本面历史拉取 (2018 起): 走步训练期的财报基础数据

东财 datacenter 直连 + 限速 + 冷却 (适配器内建)。日志: artifacts/fundamentals_fetch.log
失败可无限重跑 (缓存按报告期落盘, 已抓期秒过)。
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

LOG_PATH = ROOT / 'artifacts' / 'fundamentals_fetch.log'


class _Tee:
    def __init__(self, path: Path):
        self._files = [sys.stdout, open(path, 'a', encoding='utf-8')]

    def write(self, text):
        for f in self._files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self._files:
            f.flush()


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(LOG_PATH)
    sys.stderr = sys.stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    from data.fundamentals import FundamentalsProvider
    fp = FundamentalsProvider(delay_days=110)
    print(f">>> 全量基本面拉取开始 | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    n = fp.fetch_all_reports(start_year='2018')
    print(f">>> 完成 | 报告期 {n} 个 | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return 0 if n > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
