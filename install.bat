@echo off
setlocal EnableDelayedExpansion

title Wildlife Camera Trap Analyzer - Windows Installer

echo ============================================================
echo  Wildlife Camera Trap Auto-Analyzer - Windows Installer
echo ============================================================
echo.

REM ============================================================
REM  STEP 0 — Fresh vs update choice
REM ============================================================
set FRESH=false
for %%A in (%*) do (
    if "%%A"=="--fresh" set FRESH=true
)

if "!FRESH!"=="false" (
    if exist "venv\" (
        echo An existing installation was found.
        echo.
        echo   1^) Update / repair  - keep existing environment, install missing packages
        echo   2^) Fresh install    - wipe everything and start completely clean
        echo.
        set /p INSTALL_TYPE="Enter 1 or 2 [default: 1]: "
        if "!INSTALL_TYPE!"=="2" set FRESH=true
    )
)

if "!FRESH!"=="true" (
    echo.
    echo [INFO] Removing existing environment and cache...
    if exist "venv\"                  rd /s /q "venv"
    if exist "temp_packages_download\" rd /s /q "temp_packages_download"
    if exist "run.bat"                del /f /q "run.bat"
    conda env remove -n wildlife-analyzer --yes >nul 2>&1
    echo [OK] Clean slate ready.
    echo.
)

REM ============================================================
REM  STEP 1 — Find a compatible Python (3.11 or 3.12 preferred)
REM    megadetector and yolov5 have no pre-built wheels for
REM    Python 3.14+. Using the Windows Python Launcher (py) to
REM    prefer 3.12 or 3.11 avoids source-build failures on MSVC.
REM ============================================================
set PYTHON=

REM Try specific versions via the Windows Python Launcher first
for %%V in (3.12 3.11 3.13) do (
    if "!PYTHON!"=="" (
        py -%%V --version >nul 2>&1
        if not errorlevel 1 (
            set PYTHON=py -%%V
        )
    )
)

REM Fall back to whatever 'python' resolves to
if "!PYTHON!"=="" (
    python --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON=python
    )
)

if "!PYTHON!"=="" (
    echo [ERROR] Python not found.
    echo         Install Python 3.12 from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('!PYTHON! --version 2^>^&1') do set PYVER=%%v
echo [OK] Python !PYVER! found  ^(!PYTHON!^).

REM Warn if Python 3.14+ is the only option — some wheels may be missing
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    if %%b GEQ 14 (
        echo.
        echo [WARN] Python !PYVER! is very new. Some packages ^(megadetector,
        echo        yolov5^) may not have pre-built wheels yet and could fail
        echo        to compile from source. If installation fails, install
        echo        Python 3.12 from https://www.python.org/downloads/ and
        echo        re-run this installer.
        echo.
    )
)

REM ============================================================
REM  STEP 2 — Create virtual environment
REM ============================================================
if exist "venv\" (
    echo [INFO] Virtual environment already exists.
) else (
    echo [INFO] Creating virtual environment with !PYTHON!...
    !PYTHON! -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM ============================================================
REM  STEP 3 — Activate the virtual environment
REM ============================================================
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment active.

REM ============================================================
REM  STEP 4 — Upgrade pip
REM ============================================================
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet ^
    --trusted-host pypi.org ^
    --trusted-host files.pythonhosted.org
echo [OK] pip is up to date.

REM ============================================================
REM  STEP 5 — SSL workaround for corporate/managed networks
REM    pip-system-certs makes pip trust the Windows certificate
REM    store, which already contains your organisation's CA.
REM    PYTHONHTTPSVERIFY=0 covers any build-step subprocesses
REM    that bypass pip and call urllib directly.
REM ============================================================
echo [INFO] Installing SSL certificate helper...
python -m pip install pip-system-certs --quiet ^
    --trusted-host pypi.org ^
    --trusted-host files.pythonhosted.org
set PYTHONHTTPSVERIFY=0
set PYTHONWARNINGS=ignore::UserWarning

REM ============================================================
REM  STEP 6 — Install all dependencies from requirements.txt
REM ============================================================
echo.
echo [INFO] Installing dependencies from requirements.txt
echo       This may take 10-20 minutes on the first run.
echo       Do not close this window.
echo.

python -m pip install -r requirements.txt ^
    --trusted-host pypi.org ^
    --trusted-host files.pythonhosted.org

if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo.
    echo  Common causes:
    echo   1. Corporate proxy / SSL inspection
    echo      Try running:  set PYTHONHTTPSVERIFY=0
    echo      Then re-run install.bat
    echo   2. No internet connection
    echo      Check your network and try again.
    echo   3. Python version too new
    echo      Some packages may not yet support Python 3.14+
    echo      Try Python 3.11 or 3.12 from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo.
echo [OK] All packages installed.

REM ============================================================
REM  STEP 7 — Download AI model weights  (~1.5 GB, one-time)
REM ============================================================
echo.
echo [INFO] Downloading AI model weights (~1.5 GB).
echo       This only happens once. Do not close this window.
echo.
python force_download.py
if errorlevel 1 (
    echo.
    echo [WARNING] Model download failed or was interrupted.
    echo           You can re-run it any time with:
    echo               venv\Scripts\activate.bat
    echo               python force_download.py
    echo           The app will try to download on first use otherwise.
    echo.
)
echo [OK] Models ready.

REM ============================================================
REM  STEP 8 — Create run.bat launcher
REM ============================================================
(
    echo @echo off
    echo call "%~dp0venv\Scripts\activate.bat"
    echo echo Starting Wildlife Camera Trap Auto-Analyzer...
    echo echo Access the app at: http://localhost:8501
    echo python -m streamlit run "%~dp0app.py"
    echo pause
) > run.bat
echo [OK] Created run.bat launcher.

REM ============================================================
REM  Done
REM ============================================================
echo.
echo ============================================================
echo  Installation complete!
echo.
echo  To start the app:
echo      Double-click run.bat
echo  or run from this folder:
echo      run.bat
echo ============================================================
echo.
pause
