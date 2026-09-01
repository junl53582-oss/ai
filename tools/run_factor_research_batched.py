"""
因子研究分批运行器 (tools/run_factor_research_batched.py)

背景 (2026-09-01): 全量 79 因子研究约 4-6 小时单核运行, 且本机环境多次在 1-2h
窗口外部终止长跑进程。分批把任务切成可独立重试的单元:

    batch1: factors [0:20)   -> reports/production_research_v2/batch1
    batch2: factors [20:40)  -> .../batch2
    batch3: factors [40:60)  -> .../batch3
    batch4: factors [60:79)  -> .../batch4

每批失败自动重试 (最多 max_retries 次), 全部完成后可合并各批报告做全因子对比。
注意: 全局 BH-FDR 在校正时只覆盖批内因子, 跨批对比时以 factor_ic/factor_summary 为准。

用法:
    python tools/run_factor_research_batched.py [--batch-size 20] [--max-retries 2] [--dataset ...]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"C:\Users\lin\AppData\Local\Programs\Python\Python311\python.exe"

BATCHES = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 79)]


def run_batch(start: int, end: int, dataset: str, out_root: Path, attempt: int) -> bool:
    out_dir = out_root / f"batch{start}_{end}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, "-u", str(ROOT / "tools" / "run_factor_research.py"),
        "--dataset", dataset,
        "--output-dir", str(out_dir),
        "--factor-start", str(start),
        "--factor-end", str(end),
    ]
    log_path = ROOT / "artifacts" / f"v2_research_batch{start}_{end}_a{attempt}.log"
    print(f"[{time.strftime('%H:%M:%S')}] 批次 {start}:{end} 第 {attempt} 次启动 -> {out_dir.name}", flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    n_files = len(list(out_dir.glob("*.csv"))) + len(list(out_dir.glob("*.json"))) + len(list(out_dir.glob("*.md")))
    ok = rc == 0 and n_files >= 5
    print(f"[{time.strftime('%H:%M:%S')}] 批次 {start}:{end} {'成功' if ok else '失败'} (rc={rc}, 产物={n_files})", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(ROOT / "data_storage" / "research" / "factor_matrix_300_v2.parquet"))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--batches", type=str, default=None, help="逗号分隔批次序号, 如 '1,2' 只跑前两批")
    args = parser.parse_args()

    out_root = ROOT / "reports" / "production_research_v2"
    batches = BATCHES if not args.batches else [BATCHES[int(i) - 1] for i in args.batches.split(",")]
    print(f"分批研究启动: {len(batches)} 批, 数据集 {args.dataset}")

    all_ok = True
    for start, end in batches:
        ok = False
        for attempt in range(1, args.max_retries + 1):
            ok = run_batch(start, end, args.dataset, out_root, attempt)
            if ok:
                break
            print(f"  批次 {start}:{end} 第 {attempt} 次失败, 重试...", flush=True)
            time.sleep(10)
        if not ok:
            all_ok = False
            print(f"FATAL: 批次 {start}:{end} 重试耗尽仍失败", flush=True)

    print("=" * 60)
    print(f"分批研究{'全部完成' if all_ok else '存在失败批次'} -> {out_root}")
    print("合并建议: 用各批 factor_ic.csv / factor_summary.csv / factor_selection.csv 做全因子对比")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
