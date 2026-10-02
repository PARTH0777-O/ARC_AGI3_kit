@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\kaggle.exe" (
    set "KAGGLE=.venv\Scripts\kaggle.exe"
) else (
    set "KAGGLE=kaggle"
)

echo Checking Kaggle kernel status...
"%KAGGLE%" kernels status your-username/arc-agi-3-agent
