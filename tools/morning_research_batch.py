"""早晨研究批处理: 资金流补齐 -> 真实资金流因子走步实验 (07:30 计划任务调用)"""
import io
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = ROOT / 'artifacts' / 'morning_research_batch.log'


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(name, args, timeout):
    log(f">>> {name} 开始")
    ok = False
    try:
        r = subprocess.run([PY, '-u', *args], cwd=str(ROOT), timeout=timeout)
        ok = (r.returncode == 0)
    except subprocess.TimeoutExpired:
        log(f"  [!] {name} 超时")
    log(f"<<< {name} {'DONE' if ok else 'FAIL'}")
    return ok


def main() -> int:
    log("=" * 60)
    log("早晨研究批处理启动 (资金流补齐 + 走步实验)")
    run("资金流补齐", ["tools/fetch_daily_fundflow.py"], 3600)
    # 实验即使补齐部分失败也照跑 (已有数据的标的仍可评估)
    run("真实资金流因子走步实验", ["tools/fundflow_walkforward_test.py"], 1500)
    log("早晨批处理结束")
    return 0


if __name__ == '__main__':
    sys.exit(main())
