import os
import sys
import subprocess
from pathlib import Path

task_name = "QuantAutopilot_1505"
python_exe = sys.executable
script_path = str(Path("tools/daily_autopilot_runner.py").resolve())
work_dir = str(Path(".").resolve())

print("================================================================================")
print(">>> [注册 Windows 计划任务] 每日 15:05 全自动量化巡航调度")
print("================================================================================")
print(f"任务名称:      {task_name}")
print(f"Python 解释器: {python_exe}")
print(f"执行脚本路径:  {script_path}")
print(f"工作目录:      {work_dir}")

# 包装命令行
cmd_str = f'"{python_exe}" "{script_path}"'

# 执行 schtasks 创建任务 (周一至周五 15:05:00)
cmd = [
    "schtasks", "/Create",
    "/TN", task_name,
    "/TR", cmd_str,
    "/SC", "WEEKLY",
    "/D", "MON,TUE,WED,THU,FRI",
    "/ST", "15:05",
    "/F"
]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print(res.stderr)

print("\n[*] 正在查询 Windows Task Scheduler 注册状态:")
query_res = subprocess.run(["schtasks", "/Query", "/TN", task_name, "/FO", "LIST"], capture_output=True, text=True)
print(query_res.stdout)
if query_res.returncode == 0:
    print(f"\n[+] Windows 计划任务 [{task_name}] 注册成功并在系统中正式生效！")
