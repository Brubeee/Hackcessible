@echo off
echo ======================================================================
echo Launching Hackcessible India 2026 - Conversational Visual Layer
echo ======================================================================
echo.

cd /d "%~dp0\.."

REM Activate backend virtual environment
if not exist "backend\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at backend\.venv
    echo Please ensure dependencies are installed.
    pause
    exit /b 1
)

set PYTHONPATH=backend

echo Starting local FastAPI backend and web server on http://127.0.0.1:8000...
echo All audio processing and speech recognition will run locally.
echo Audio is processed in memory and is not written to disk by this application.
echo.

start http://127.0.0.1:8000

backend\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
