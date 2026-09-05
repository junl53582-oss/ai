@echo off
chcp 65001 > nul
echo ======================================================================
echo    正在启动 A股多因子机器学习量化决策看板 (Streamlit Dashboard)
echo ======================================================================

set PROJECT_DIR=%~dp0..
cd /d "%PROJECT_DIR%"

set PY311=C:\Users\lin\AppData\Local\Programs\Python\Python311\python.exe

echo [%date% %time%] 启动 Streamlit 服务 (端口: 8501)...
"%PY311%" -m streamlit run dashboard/app.py

pause
