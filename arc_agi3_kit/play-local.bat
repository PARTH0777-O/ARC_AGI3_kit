@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON=..\.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

if exist "scripts\play_local.py" (
    set "SCRIPT=scripts\play_local.py"
) else (
    set "SCRIPT=arc_agi3_kit\scripts\play_local.py"
)

"%PYTHON%" "%SCRIPT%" %*
