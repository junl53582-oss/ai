"""每日数据刷新编排器 (headless): 行情同步 -> 研究矩阵重建 -> 生产推理 -> 快讯行情

用途: 让面板全部数据滚动到最新完整交易日。由计划任务或手动触发, 全程日志落盘。
步骤:
  1. DataManager.sync_and_build_dataset()  更新 market_daily.parquet 到最新收盘 (~40 分钟, 全量重拉)
  2. tools/rebuild_dataset_300_from_sync.py 重建研究矩阵 factor_matrix_300.parquet (~3 分钟)
  3. tools/predict_stocks.py               生产模型重新推理生成最新清单 (~12 秒)
  4. AutoSyncEngine.sync_picks_and_news    真实个股快讯 + 实时行情回填 (~1 分钟)
"""
import io
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


class _Tee:
    """同时写 stdout 与日志文件 (任务调度器环境下 stdout 不可见, 依赖文件日志)"""

    def __init__(self, path: Path):
        self._files = [sys.stdout, open(path, "a", encoding="utf-8")]

    def write(self, text):
        for f in self._files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self._files:
            f.flush()


def _setup_tee() -> Path:
    log_path = ROOT / "artifacts" / "refresh_daily_data.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(log_path)
    sys.stderr = sys.stdout
    return log_path


def run_step(name: str, args, timeout: int = 3600) -> bool:
    print(f"\n===== [STEP] {name} | {time.strftime('%H:%M:%S')} =====", flush=True)
    t0 = time.time()
    ok = False
    try:
        r = subprocess.run([PY, "-u", *args], cwd=str(ROOT), timeout=timeout)
        ok = (r.returncode == 0)
    except subprocess.TimeoutExpired:
        print(f"  [!] 步骤超时 ({timeout}s), 终止子进程", flush=True)
    print(f"===== [{'DONE' if ok else 'FAIL'}] {name} | {time.time()-t0:.0f}s =====", flush=True)
    return ok


def main() -> int:
    log_path = _setup_tee()
    t0 = time.time()
    print(f">>> 每日数据刷新开始 | {time.strftime('%Y-%m-%d %H:%M:%S')} | 日志: {log_path}", flush=True)

    # Step 1: 全市场行情同步 (内联执行, 无法子进程化——数据对象在内存中传递)
    print("\n===== [STEP] 行情同步 sync_and_build_dataset | 开始 =====", flush=True)
    sys.path.insert(0, str(ROOT))
    from config import settings
    from data.data_fetcher import DataFetcher
    from data.data_manager import DataManager
    from data.universe_provider import create_universe_provider

    fetcher = DataFetcher(allow_synthetic=False)
    manager = DataManager(universe_provider=create_universe_provider(settings), fetcher=fetcher)
    market_df = manager.sync_and_build_dataset()
    print(f"===== [DONE] 行情同步 | {len(market_df):,} 行 | 最新日期 {market_df['date'].max()} =====", flush=True)

    # Step 2-4: 子进程执行
    if not run_step("研究矩阵重建 rebuild_dataset_300_from_sync", ["tools/rebuild_dataset_300_from_sync.py"], timeout=1200):
        return 2
    if not run_step("生产推理 predict_stocks", ["tools/predict_stocks.py"], timeout=600):
        return 3

    print("\n===== [STEP] 快讯行情回填 sync_picks_and_news =====", flush=True)
    from data.live_market_and_news_api import AutoSyncEngine
    for f in ["latest_stock_picks.csv", "csi500_stock_picks.csv", "aggressive_stock_picks.csv"]:
        fp = settings.ARTIFACTS_DIR / f
        if fp.exists():
            AutoSyncEngine.sync_picks_and_news(fp)
            print(f"  synced: {f}", flush=True)

    # Step 5: 资金流数据每日积累 (东财 push2his HTTP 通道, ~120 日历史, 幂等)
    if not run_step("资金流积累 fetch_daily_fundflow", ["tools/fetch_daily_fundflow.py"], timeout=1800):
        print("[WARN] 资金流积累失败 (不影响主链路), 明日重试", flush=True)

    print(f"\n>>> 每日数据刷新全部完成 | 总耗时 {time.time()-t0:.0f}s | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
