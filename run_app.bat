@echo off
REM MonsoonWatch - start the Streamlit app on Windows.
REM Double-click this file. First run installs packages (~3-5 min); later runs start in seconds.
REM Needs Python 3.12 or newer (python.org, tick "Add python.exe to PATH" when installing).
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || (echo Python 3.12+ not found. Install it from python.org and try again. & pause & exit /b 1)
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt || (echo Package install failed. & pause & exit /b 1)
) else (
    call .venv\Scripts\activate.bat
)
echo Opening MonsoonWatch at http://localhost:8501  (close this window to stop it)
python -m streamlit run app/app.py
pause
