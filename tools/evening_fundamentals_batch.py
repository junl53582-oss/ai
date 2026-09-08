"""晚间基本面批处理: 全量财报拉取 -> 基本面因子走步实验 (计划任务调用)"""
import io
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = ROOT / 'artifacts' / 'evening_fundamentals_batch.log'


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
    log("晚间基本面批处理启动 (财报全量拉取 + 因子走步实验)")
    run("基本面全量拉取(2018+)", ["tools/fetch_all_fundamentals.py"], 5400)
    # 实验照跑: 缓存按报告期落盘, 部分成功也可评估
    run("基本面因子走步实验", ["tools/fundamentals_walkforward_test.py"], 1800)
    log("晚间批处理结束")
    return 0


if __name__ == '__main__':
    sys.exit(main())
