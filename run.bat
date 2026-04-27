@echo off
title Network Intrusion Detection System - Model Demo

echo ========================================
echo   Network Intrusion Detection System
echo   Model Demonstration Application
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

echo [INFO] Python detected: 
python --version
echo.

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install requirements
echo [INFO] Installing dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [INFO] Dependencies installed successfully
echo.

:: Launch Streamlit app
echo [INFO] Starting Network Intrusion Detection System...
echo [INFO] Opening browser automatically...
echo.

:: Wait a moment for the server to start, then open browser
start "" http://localhost:8501
streamlit run app.py --server.port=8501 --server.headless=true

:: Pause if there's an error
if %errorlevel% neq 0 (
    echo [ERROR] Application failed to start
    pause
)

pause