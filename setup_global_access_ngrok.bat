@echo off
echo ========================================================================
echo           GLOBAL ACCESS SETUP - NGROK TUNNELING
echo ========================================================================
echo.
echo This will make your sentiment analysis app accessible from ANYWHERE
echo in the world using ngrok (free tunneling service).
echo.
echo ========================================================================

REM Check if ngrok exists
where ngrok >nul 2>&1
if %errorLevel% == 0 (
    echo ✓ ngrok is already installed
    goto :start_tunnel
) else (
    echo ❌ ngrok is not installed
    echo.
    echo INSTALLING NGROK...
    echo.
)

REM Download ngrok
echo [1/4] Downloading ngrok...
echo.
echo Please follow these steps:
echo.
echo 1. Go to: https://ngrok.com/download
echo 2. Download ngrok for Windows
echo 3. Extract ngrok.exe to this folder
echo 4. Sign up for free account at: https://ngrok.com/signup
echo 5. Get your authtoken from: https://dashboard.ngrok.com/auth
echo.
echo Press any key after you've downloaded ngrok.exe to this folder...
pause >nul

REM Check again if ngrok exists
if not exist ngrok.exe (
    echo.
    echo ❌ ngrok.exe not found in current folder!
    echo Please download and extract ngrok.exe here.
    pause
    exit /b 1
)

:setup_auth
echo.
echo [2/4] Setting up ngrok authentication...
echo.
echo Enter your ngrok authtoken (from https://dashboard.ngrok.com/auth):
set /p NGROK_TOKEN=Token: 

ngrok config add-authtoken %NGROK_TOKEN%

if %errorLevel% == 0 (
    echo ✓ Authentication configured successfully!
) else (
    echo ❌ Failed to set authtoken
    pause
    exit /b 1
)

:start_tunnel
echo.
echo [3/4] Starting your Flask app...
echo.

REM Start Flask app in background
start /min cmd /c "python app.py"

echo Waiting for Flask to start...
timeout /t 5 /nobreak >nul

echo.
echo [4/4] Creating global tunnel...
echo.
echo ========================================================================
echo              🌍 CREATING GLOBAL ACCESS TUNNEL 🌍
echo ========================================================================
echo.

REM Start ngrok tunnel
echo Starting ngrok tunnel on port 5000...
echo.
echo Your app will be accessible at a URL like:
echo   https://abc123.ngrok.io
echo.
echo IMPORTANT: Keep this window open to maintain global access!
echo Press Ctrl+C to stop the tunnel and app.
echo.
echo ========================================================================

ngrok http 5000

echo.
echo Tunnel closed. Stopping Flask app...
taskkill /f /im python.exe >nul 2>&1
pause