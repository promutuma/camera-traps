#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Wildlife Camera Trap Auto-Analyzer - Mac/Linux Installer"
echo "============================================================"
echo ""

# --- Detect OS ---
OS="$(uname -s)"
echo "[INFO] Detected OS: $OS"

# --- System dependencies (Linux only) ---
if [[ "$OS" == "Linux" ]]; then
    if command -v apt-get &>/dev/null; then
        echo "[INFO] Installing system dependencies (requires sudo)..."
        sudo apt-get update -qq
        sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 curl python3-venv
        echo "[OK] System dependencies installed."
    elif command -v dnf &>/dev/null; then
        echo "[INFO] Installing system dependencies via dnf..."
        sudo dnf install -y mesa-libGL glib2 libSM libXext curl python3-virtualenv
    fi
fi

# --- Choose environment manager ---
USE_CONDA=false
if command -v conda &>/dev/null; then
    echo ""
    echo "Conda detected. Which environment manager do you prefer?"
    echo "  1) venv (standard Python, recommended)"
    echo "  2) Conda / Miniconda"
    read -rp "Enter 1 or 2 [default: 1]: " CHOICE
    if [[ "$CHOICE" == "2" ]]; then
        USE_CONDA=true
    fi
fi

if [[ "$USE_CONDA" == true ]]; then
    # ── Conda path ────────────────────────────────────────────────────────
    echo "[INFO] Setting up Conda environment from environment.yml..."
    conda env create -f environment.yml --force
    echo "[OK] Conda environment 'wildlife-analyzer' created."

    # Create run.sh for conda
    cat > run.sh << 'RUNEOF'
#!/usr/bin/env bash
set -e
eval "$(conda shell.bash hook)"
conda activate wildlife-analyzer
echo "Starting Wildlife Camera Trap Auto-Analyzer..."
echo "Access the app at: http://localhost:8501"
python -m streamlit run app.py
RUNEOF

    echo ""
    echo "============================================================"
    echo " Conda setup complete!"
    echo " To download AI models (one-time, ~1.5 GB):"
    echo "     conda activate wildlife-analyzer && python force_download.py"
    echo " To start the app:"
    echo "     ./run.sh"
    echo "============================================================"

else
    # ── venv path ─────────────────────────────────────────────────────────
    # Check Python
    if command -v python3.13 &>/dev/null; then
        PYTHON=python3.13
    elif command -v python3.12 &>/dev/null; then
        PYTHON=python3.12
    elif command -v python3.11 &>/dev/null; then
        PYTHON=python3.11
    elif command -v python3 &>/dev/null; then
        PYTHON=python3
    else
        echo "[ERROR] Python 3 not found. Install Python 3.12 from https://www.python.org/downloads/"
        exit 1
    fi

    PYVER=$($PYTHON --version 2>&1)
    echo "[OK] $PYVER found."

    if [[ -d "venv" ]]; then
        echo "[INFO] Virtual environment already exists, skipping creation."
    else
        echo "[INFO] Creating virtual environment..."
        $PYTHON -m venv venv
        echo "[OK] Virtual environment created."
    fi

    source venv/bin/activate
    echo "[INFO] Upgrading pip..."
    pip install --upgrade pip --quiet

    echo "[INFO] Installing dependencies (10–20 min on first run)..."
    pip install --no-cache-dir -r requirements.txt
    echo "[OK] Dependencies installed."

    echo "[INFO] Pre-downloading AI models (~1.5 GB). This only happens once..."
    python force_download.py
    echo "[OK] Models ready."

    # Create run.sh for venv
    cat > run.sh << 'RUNEOF'
#!/usr/bin/env bash
set -e
if [[ ! -d "venv" ]]; then
    echo "[ERROR] venv not found. Please run ./install.sh first."
    exit 1
fi
source venv/bin/activate
echo "Starting Wildlife Camera Trap Auto-Analyzer..."
echo "Access the app at: http://localhost:8501"
python -m streamlit run app.py
RUNEOF

    echo ""
    echo "============================================================"
    echo " Installation complete!"
    echo " To start the app:"
    echo "     ./run.sh"
    echo "============================================================"
fi

chmod +x run.sh
