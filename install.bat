@echo off
setlocal EnableDelayedExpansion

title Wildlife Camera Trap Analyzer - Windows Installer

echo ============================================================
echo  Wildlife Camera Trap Auto-Analyzer - Windows Setup
echo ============================================================
echo.

REM --- Check for Conda ---
conda --version >nul 2>&1
if not errorlevel 1 (
    echo Conda detected. Which environment manager do you prefer?
    echo   1^) venv  ^(standard Python, recommended^)
    echo   2^) Conda / Miniconda
    set /p CHOICE="Enter 1 or 2 [default: 1]: "
    if "!CHOICE!"=="2" goto :conda_setup
)
goto :venv_setup

REM ================================================================
:conda_setup
echo.
echo [INFO] Setting up Conda environment from environment.yml...
conda env create -f environment.yml --force
if errorlevel 1 (
    echo [ERROR] Conda environment creation failed.
    pause
    exit /b 1
)
echo [OK] Conda environment 'wildlife-analyzer' created.

echo @echo off > run.bat
echo call conda activate wildlife-analyzer >> run.bat
echo echo Starting Wildlife Camera Trap Auto-Analyzer... >> run.bat
echo echo Access the app at: http://localhost:8501 >> run.bat
echo python -m streamlit run app.py >> run.bat

echo.
echo ============================================================
echo  Conda setup complete!
echo  To download AI models (one-time, ~1.5 GB):
echo      conda activate wildlife-analyzer
echo      python force_download.py
echo  To start the app:
echo      run.bat
echo ============================================================
pause
exit /b 0

REM ================================================================
:venv_setup
REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11 from:
    echo         https://www.python.org/downloads/
    echo         Check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found.

if exist "venv\" (
    echo [INFO] Virtual environment already exists, skipping creation.
) else (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

call venv\Scripts\activate.bat
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [INFO] Installing dependencies (10-20 min on first run)...
pip install --no-cache-dir -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check internet connection and retry.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.

echo [INFO] Pre-downloading AI models (~1.5 GB). This only happens once...
python force_download.py
echo [OK] Models ready.

echo @echo off > run.bat
echo call venv\Scripts\activate.bat >> run.bat
echo echo Starting Wildlife Camera Trap Auto-Analyzer... >> run.bat
echo echo Access the app at: http://localhost:8501 >> run.bat
echo python -m streamlit run app.py >> run.bat
echo [OK] Created run.bat launcher.

echo.
echo ============================================================
echo  Installation complete!
echo  To start the app, double-click run.bat or run:
echo      run.bat
echo ============================================================
pause
