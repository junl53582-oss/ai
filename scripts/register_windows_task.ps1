 = "QuantAutopilot_1505"
 = (Get-Command python).Source
 = Join-Path C:\Users\lin\Documents\股票预测 "tools\daily_autopilot_runner.py"
 = C:\Users\lin\Documents\股票预测.Path

Write-Host "================================================================================"
Write-Host ">>> [注册 Windows 本地系统定时任务] 每日 15:05 自动执行全流程巡航"
Write-Host "================================================================================"
Write-Host "任务名称:   "
Write-Host "Python解释器: "
Write-Host "执行脚本:   "
Write-Host "工作目录:   "

# 使用 schtasks 创建任务 (周一至周五 15:05:00)
 = """ """
 = "schtasks /Create /TN "" /TR "" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:05 /F"

Write-Host "正在执行系统任务注册命令: "
Invoke-Expression 

Write-Host "
[*] 正在查询验证已注册的计划任务状态:"
schtasks /Query /TN "" /FO LIST
