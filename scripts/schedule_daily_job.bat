@echo off
chcp 65001 > nul
echo ======================================================================
echo    A股量化系统 - 自动安装每日 15:05 Windows 定时计划任务
echo ======================================================================

set TASK_NAME=AShare_Quant_Daily_Post_Market
set SCRIPT_PATH=C:\Users\lin\Documents\股票预测\scripts\run_daily_post_market.bat

echo 正在创建 Windows 计划任务: %TASK_NAME% ...
echo 执行时间: 每个工作日 (周一至周五) 15:05:00
echo 目标脚本: %SCRIPT_PATH%

schtasks /create /tn "%TASK_NAME%" /tr "\"%SCRIPT_PATH%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 15:05 /f

if %ERRORLEVEL% equ 0 (
    echo.
    echo ✅ 计划任务安装成功！系统将在每个交易日 15:05 自动执行量化决策流水线并推送结果。
    echo 如需手动查看或删除任务，请在命令行运行: schtasks /query /tn "%TASK_NAME%"
) else (
    echo.
    echo ❌ 创建计划任务失败，请以管理员身份运行此批处理脚本。
)

echo ======================================================================
pause
