#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Wildlife Camera Trap Auto-Analyzer - Mac/Linux Installer"
echo "============================================================"
echo ""

# --- Handle --fresh flag ---
FRESH=false
for arg in "$@"; do
    if [[ "$arg" == "--fresh" ]]; then
        FRESH=true
    fi
done

if [[ "$FRESH" == true ]]; then
    echo "[INFO] --fresh: removing existing environment and download cache..."
    rm -rf venv temp_packages_download run.sh
    if command -v conda &>/dev/null; then
        conda env remove -n wildlife-analyzer --yes 2>/dev/null || true
    fi
    echo "[OK] Clean slate ready."
    echo ""
fi

# --- Detect OS ---
OS="$(uname -s)"
echo "[INFO] Detected OS: $OS"

# --- System dependencies (Linux only) ---
if [[ "$OS" == "Linux" ]]; then
    if command -v apt-get &>/dev/null; then
        echo "[INFO] Installing system dependencies (requires sudo)..."
        sudo apt-get update -qq
        sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 curl python3-venv python3.12-venv 2>/dev/null || \
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
    
    # Create temporary SSL patch
    mkdir -p temp_ssl_patch
    cat > temp_ssl_patch/sitecustomize.py << 'EOF'
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
EOF
    export ORIGINAL_PYTHONPATH="$PYTHONPATH"
    export PYTHONPATH="temp_ssl_patch:$PYTHONPATH"

    conda env create -f environment.yml --force
    CONDA_ERR=$?

    # Clean up SSL patch
    export PYTHONPATH="$ORIGINAL_PYTHONPATH"
    rm -rf temp_ssl_patch

    if [[ $CONDA_ERR -ne 0 ]]; then
        echo "[ERROR] Conda environment creation failed."
        exit 1
    fi
    echo "[OK] Conda environment 'wildlife-analyzer' created."

    echo "[INFO] Installing patched ultralytics-yolov5 and megadetector..."
    eval "$(conda shell.bash hook)"
    conda activate wildlife-analyzer
    python3 install_dependencies.py --conda
    if [[ $? -ne 0 ]]; then
        echo "[ERROR] Failed to install dependencies."
        exit 1
    fi

    echo "[INFO] Pre-downloading AI models (~1.5 GB). This only happens once..."
    python3 force_download.py
    echo "[OK] Models ready."

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

    # Install CPU-only PyTorch when no NVIDIA GPU is present, to avoid
    # downloading ~3 GB of unused CUDA libraries.
    INSTALL_DEPS_ARGS=""
    if ! command -v nvidia-smi &>/dev/null; then
        echo "[INFO] No NVIDIA GPU detected. Installing CPU-only PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
        echo "[OK] CPU-only PyTorch installed."
        # Exclude torch/torchvision from the parallel download phase — they're
        # already installed above and pip download ignores the venv, so without
        # this filter it re-downloads the huge CUDA wheels and hits disk quota.
        grep -vE '^torch(vision)?' requirements.txt > /tmp/requirements_cpu_filtered.txt
        INSTALL_DEPS_ARGS="--requirements=/tmp/requirements_cpu_filtered.txt"
    fi

    echo "[INFO] Installing dependencies (10–20 min on first run)..."
    
    # Create temporary SSL patch
    mkdir -p temp_ssl_patch
    cat > temp_ssl_patch/sitecustomize.py << 'EOF'
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
EOF
    export ORIGINAL_PYTHONPATH="$PYTHONPATH"
    export PYTHONPATH="temp_ssl_patch:$PYTHONPATH"

    python3 install_dependencies.py $INSTALL_DEPS_ARGS
    PIP_ERR=$?

    # Clean up SSL patch
    export PYTHONPATH="$ORIGINAL_PYTHONPATH"
    rm -rf temp_ssl_patch

    if [[ $PIP_ERR -ne 0 ]]; then
        echo "[ERROR] Dependency installation failed."
        exit 1
    fi
    echo "[OK] Dependencies installed."

    echo "[INFO] Installing FastAPI backend dependencies..."
    pip install -r backend/requirements.txt --quiet
    echo "[OK] Backend dependencies installed."

    echo "[INFO] Installing frontend dependencies (npm install)..."
    if command -v npm &>/dev/null; then
        (cd frontend && npm install --silent)
        echo "[OK] Frontend dependencies installed."
    else
        echo "[WARNING] npm not found — skipping frontend install."
        echo "          Install Node.js from https://nodejs.org/ then run: cd frontend && npm install"
    fi

    echo "[INFO] Pre-downloading AI models (~1.5 GB). This only happens once..."
    python force_download.py
    echo "[OK] Models ready."

    # Create run.sh that starts the FastAPI + React stack via dev.sh
    cat > run.sh << 'RUNEOF'
#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/dev.sh" ]]; then
    echo "[ERROR] dev.sh not found. Please re-run ./install.sh."
    exit 1
fi
exec bash "$SCRIPT_DIR/dev.sh"
RUNEOF

    echo ""
    echo "============================================================"
    echo " Installation complete!"
    echo " To start the app:"
    echo "     ./run.sh        (or: bash dev.sh)"
    echo ""
    echo " Frontend  → http://localhost:5173"
    echo " API docs  → http://localhost:8000/docs"
    echo "============================================================"
fi

chmod +x run.sh
