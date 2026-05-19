#!/usr/bin/env bash
set -e
if [[ ! -d "venv" ]]; then
    echo "[ERROR] Virtual environment not found. Please run ./install.sh first."
    exit 1
fi
source venv/bin/activate
echo "Starting Wildlife Camera Trap Auto-Analyzer..."
echo "Access the app at: http://localhost:8501"
python -m streamlit run app.py
