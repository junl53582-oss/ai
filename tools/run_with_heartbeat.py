"""
研究/训练长任务心跳包装器 (tools/run_with_heartbeat.py)

用法:
    python tools/run_with_heartbeat.py <cmd...>
    # 例: python tools/run_with_heartbeat.py python -u tools/run_factor_research.py --dataset ... --output-dir ...

行为:
    1. 以子进程方式运行目标命令, 每 60s 记录心跳 (时间戳 + 子进程存活 + 已运行秒数)
    2. 输出落盘: <cwd>/artifacts/heartbeat_<name>.log
    3. 子进程退出后记录退出码与总时长 (0=成功; 非0/信号终止可见)
    4. 若子进程被外部终止 (无退出码), 明确标记 KILLED_EXTERNALLY
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    cmd = sys.argv[1:]
    cwd = Path.cwd()
    name = Path(cmd[-1]).stem if cmd else "task"
    log_path = cwd / "artifacts" / f"heartbeat_{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"HEARTBEAT START: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(cwd))

    exit_code = None
    while True:
        time.sleep(60)
        rc = proc.poll()
        elapsed = int(time.time() - t0)
        if rc is None:
            log(f"heartbeat: alive, running={elapsed}s, pid={proc.pid}")
        else:
            exit_code = rc
            break

    elapsed = int(time.time() - t0)
    if exit_code is None:
        log(f"HEARTBEAT END: KILLED_EXTERNALLY (no exit code, killed after {elapsed}s)")
        sys.exit(130)
    log(f"HEARTBEAT END: exit_code={exit_code}, duration={elapsed}s")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
