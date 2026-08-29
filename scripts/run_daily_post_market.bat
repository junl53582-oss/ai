@echo off
chcp 65001 > nul
echo ======================================================================
echo    A股量化系统 - 每日盘后全自动流水线 (15:05 Post-Market Runner)
echo ======================================================================

set PROJECT_DIR=C:\Users\lin\Documents\股票预测
cd /d %PROJECT_DIR%

REM 必须使用 Python 3.11 (pandas/lightgbm/akshare 均安装在该版本)。
REM 系统默认 python 指向 3.13 且无项目依赖，会导致定时任务直接失败。
set PY311=C:\Users\lin\AppData\Local\Programs\Python\Python311\python.exe

echo [%date% %time%] 启动盘后自动同步、走步预测、组合优化与消息推送...
"%PY311%" -u scheduler/daily_runner.py --optimizer risk_parity

if %ERRORLEVEL% equ 0 (
    echo [%date% %time%] ✅ 盘后量化决策任务执行成功！
) else (
    echo [%date% %time%] ❌ 任务执行出现异常，错误码: %ERRORLEVEL%
)

echo ======================================================================
