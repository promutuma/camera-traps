@echo off
title Wildlife Camera Trap Analyzer

if not exist "venv\" (
    echo [ERROR] Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
echo Starting Wildlife Camera Trap Auto-Analyzer...
echo Access the app at: http://localhost:8501
python -m streamlit run app.py
