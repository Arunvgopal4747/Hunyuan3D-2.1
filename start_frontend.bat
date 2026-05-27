@echo off
title Hunyuan3D-2.1 Frontend
color 0A

echo.
echo  ============================================
echo   Hunyuan3D-2.1 -- HuggingFace Space Client
echo  ============================================
echo.

:: Change to the script's own directory (works from any location / shortcut)
cd /d "%~dp0"

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

:: Install / upgrade required packages silently if missing
echo  Checking dependencies...
python -m pip install --quiet --upgrade gradio gradio_client 2>nul
echo  Dependencies OK.
echo.

:: Launch the frontend
echo  Starting local server at http://127.0.0.1:7860
echo  Press Ctrl+C to stop.
echo.
python hf_frontend.py %*

:: If Python exits (Ctrl+C or crash), pause so the window stays visible
echo.
echo  Server stopped. Press any key to close.
pause >nul
