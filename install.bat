@echo off
setlocal EnableDelayedExpansion

title Wildlife Camera Trap Analyzer - Windows Installer

echo ============================================================
echo  Wildlife Camera Trap Auto-Analyzer - Windows Setup
echo ============================================================
echo.

REM --- Fresh install: check flag OR ask interactively if no args given ---
set FRESH=false
for %%A in (%*) do (
    if "%%A"=="--fresh" set FRESH=true
)

if "!FRESH!"=="false" (
    if exist "venv\" (
        echo An existing installation was found.
        echo.
        echo   1^) Update / repair  - keep the current environment, install missing packages
        echo   2^) Fresh install     - wipe everything and start clean
        echo.
        set /p INSTALL_TYPE="Enter 1 or 2 [default: 1]: "
        if "!INSTALL_TYPE!"=="2" set FRESH=true
    )
)

if "!FRESH!"=="true" (
    echo.
    echo [INFO] Fresh install: removing existing environment and download cache...
    if exist "venv\" (
        rd /s /q "venv"
        echo [OK] Removed venv.
    )
    if exist "temp_packages_download\" (
        rd /s /q "temp_packages_download"
        echo [OK] Removed temp_packages_download.
    )
    if exist "run.bat" del /f /q "run.bat"
    conda env remove -n wildlife-analyzer --yes >nul 2>&1
    echo [OK] Clean slate ready.
    echo.
)

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
set PYTHONHTTPSVERIFY=0

REM --- Create temporary SSL patch directory and sitecustomize.py ---
REM     Use absolute path so pip subprocesses (which run from a temp dir) can find it.
set SSL_PATCH_DIR=%~dp0temp_ssl_patch
mkdir "!SSL_PATCH_DIR!"
(echo import ssl) > "!SSL_PATCH_DIR!\sitecustomize.py"
(echo ssl._create_default_https_context = ssl._create_unverified_context) >> "!SSL_PATCH_DIR!\sitecustomize.py"
set PYTHONPATH=!SSL_PATCH_DIR!;!PYTHONPATH!

conda env create -f environment.yml --force
set CONDA_ERR=!errorlevel!

REM --- Clean up SSL patch ---
set PYTHONPATH=
rd /s /q "!SSL_PATCH_DIR!"
set SSL_PATCH_DIR=

if !CONDA_ERR! neq 0 (
    echo [ERROR] Conda environment creation failed.
    pause
    exit /b 1
)
echo [OK] Conda environment 'wildlife-analyzer' created.

echo [INFO] Installing patched ultralytics-yolov5 and megadetector...
call conda activate wildlife-analyzer
python install_dependencies.py --conda --trusted-host pypi.org --trusted-host files.pythonhosted.org
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [INFO] Pre-downloading AI models (~1.5 GB). This only happens once...
python force_download.py
echo [OK] Models ready.

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
python -m pip install --upgrade pip --quiet --trusted-host pypi.org --trusted-host files.pythonhosted.org

REM --- Fix SSL certificate issues on corporate/managed networks ---
REM     Some packages (e.g. ultralytics-yolov5) make HTTPS requests inside
REM     their setup.py during build. On machines with SSL inspection (corporate
REM     proxies, antivirus MITM) this fails with CERTIFICATE_VERIFY_FAILED.
REM     pip-system-certs patches pip to use the Windows certificate store.
echo [INFO] Installing SSL certificate helper for corporate networks...
python -m pip install pip-system-certs --quiet --trusted-host pypi.org --trusted-host files.pythonhosted.org

REM     Also set PYTHONHTTPSVERIFY=0 so that setup.py subprocesses (which use
REM     urllib directly, bypassing pip's cert store) also skip verification.
REM     This only applies for the duration of this installer session.
set PYTHONHTTPSVERIFY=0
set PYTHONWARNINGS=ignore::UserWarning

REM --- Create temporary SSL patch directory and sitecustomize.py ---
REM     Use absolute path so pip subprocesses (which run from a temp dir) can find it.
set SSL_PATCH_DIR=%~dp0temp_ssl_patch
mkdir "!SSL_PATCH_DIR!"
(echo import ssl) > "!SSL_PATCH_DIR!\sitecustomize.py"
(echo ssl._create_default_https_context = ssl._create_unverified_context) >> "!SSL_PATCH_DIR!\sitecustomize.py"
set PYTHONPATH=!SSL_PATCH_DIR!;!PYTHONPATH!

echo [INFO] Installing dependencies (10-20 min on first run)...
python install_dependencies.py --trusted-host pypi.org --trusted-host files.pythonhosted.org
set PIP_ERR=!errorlevel!

REM --- Clean up SSL patch ---
set PYTHONPATH=
rd /s /q "!SSL_PATCH_DIR!"
set SSL_PATCH_DIR=

if !PIP_ERR! neq 0 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo.
    echo  Common causes on Windows:
    echo   1. Corporate proxy / SSL inspection  ^(most common^)
    echo      Run this command manually, then retry install.bat:
    echo        set PYTHONHTTPSVERIFY=0
    echo   2. No internet access
    echo      Check your connection and proxy settings.
    echo   3. Python version mismatch
    echo      Use Python 3.11, 3.12, or 3.13 from python.org
    echo.
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
