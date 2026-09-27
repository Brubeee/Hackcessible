@echo off
echo ======================================================================
echo Running Hackcessible Validation (Tests + Lint + Build + Scripted E2E + PCM Audio + Noise E2E)
echo ======================================================================
echo.

cd /d "%~dp0\.."

set PYTHONPATH=backend

echo [1/7] Running Backend Pytest Suite...
backend\.venv\Scripts\pytest.exe backend\tests -v
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Backend unit tests failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [2/7] Running Frontend Vitest Component Suite...
cd frontend
call npm.cmd test
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Frontend component tests failed!
    cd ..
    exit /b %ERRORLEVEL%
)
cd ..

echo.
echo [3/7] Running Frontend Lint...
cd frontend
call npm.cmd run lint
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Frontend lint failed!
    cd ..
    exit /b %ERRORLEVEL%
)
cd ..

echo.
echo [4/7] Building Production Frontend...
cd frontend
call npm.cmd run build
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Frontend production build failed!
    cd ..
    exit /b %ERRORLEVEL%
)
cd ..

echo.
echo [5/7] Running Scripted Demo WebSocket Verification...
backend\.venv\Scripts\python.exe scripts\verify_e2e.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] E2E verification failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [6/7] Replaying Sample PCM Audio Through WebSocket...
if exist "samples\test_video_60s.wav" (
    backend\.venv\Scripts\python.exe scripts\verify_websocket_audio.py
    if errorlevel 1 (
        echo [FAIL] PCM audio WebSocket verification failed!
        exit /b 1
    )
) else (
    echo [SKIP] Local test audio samples\test_video_60s.wav is not present.
)

echo.
echo [7/7] Stress Testing Captioning With Controlled Noise...
if exist "samples\test_video_60s.wav" (
    backend\.venv\Scripts\python.exe scripts\verify_noisy_audio.py
    if errorlevel 1 (
        echo [FAIL] Noisy audio WebSocket verification failed!
        exit /b 1
    )
) else (
    echo [SKIP] Local test audio is required for the controlled-noise replay.
)

echo.
echo ======================================================================
echo ALL TESTS PASSED SUCCESSFULLY!
echo ======================================================================
