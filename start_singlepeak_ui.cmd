@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if not errorlevel 1 (
    python singlepeak_batch_ui.py
) else (
    py -3 singlepeak_batch_ui.py
)

if errorlevel 1 pause
endlocal
