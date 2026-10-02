@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

if exist ".venv\Scripts\kaggle.exe" (
    set "KAGGLE=.venv\Scripts\kaggle.exe"
) else (
    set "KAGGLE=kaggle"
)

echo Building Kaggle submission notebook...
"%PYTHON%" scripts\build_notebook.py
if errorlevel 1 (
    echo Error building notebook.
    exit /b 1
)

echo Pushing notebook to Kaggle...
"%KAGGLE%" kernels push -p notebooks
