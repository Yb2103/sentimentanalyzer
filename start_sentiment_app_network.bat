@echo off
echo ========================================================================
echo             SENTIMENT ANALYSIS APP - NETWORK VERSION
echo ========================================================================
echo.
echo Checking network configuration...
echo.

REM Get the current IP address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /R /C:"IPv4 Address.*172\." ^| findstr /V "127.0.0.1"') do (
    set "LOCAL_IP=%%a"
    set "LOCAL_IP=!LOCAL_IP: =!"
)

REM Fallback to get any IPv4 address
if not defined LOCAL_IP (
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /R /C:"IPv4 Address" ^| findstr /V "127.0.0.1" ^| head -1') do (
        set "LOCAL_IP=%%a"
        set "LOCAL_IP=!LOCAL_IP: =!"
    )
)

echo ========================================================================
echo NETWORK ACCESS INFORMATION:
echo ========================================================================
echo.
echo LOCAL ACCESS (this computer):
echo   - http://localhost:5000
echo   - http://127.0.0.1:5000
echo.
echo NETWORK ACCESS (other devices on same WiFi/network):
if defined LOCAL_IP (
    echo   - http://%LOCAL_IP%:5000
) else (
    echo   - Check your IP address manually: ipconfig
)
echo.
echo MOBILE DEVICE ACCESS:
echo   Connect your phone/tablet to the same WiFi network, then use:
if defined LOCAL_IP (
    echo   - http://%LOCAL_IP%:5000
) else (
    echo   - http://[YOUR_IP_ADDRESS]:5000
)
echo.
echo ========================================================================
echo IMPORTANT SETUP STEPS (First Time Only):
echo ========================================================================
echo.
echo 1. FIREWALL CONFIGURATION (Run as Administrator):
echo    Right-click this file and "Run as Administrator", then run:
echo    netsh advfirewall firewall add rule name="Sentiment Analysis App" dir=in action=allow protocol=TCP localport=5000
echo.
echo 2. VERIFY NETWORK CONNECTION:
echo    - Ensure all devices are on the same WiFi network
echo    - Check Windows Defender Firewall settings
echo    - Try disabling antivirus temporarily if issues persist
echo.
echo ========================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org
    pause
    exit /b 1
)

echo [1/4] Python version check...
python --version

echo.
echo [2/4] Installing/updating dependencies...
pip install -r requirements.txt --quiet --disable-pip-version-check

echo.
echo [3/4] Starting Sentiment Analysis Server...
echo.
echo Server Status: STARTING...
echo.
echo ========================================================================
echo             🚀 SERVER IS NOW RUNNING 🚀
echo ========================================================================
echo.
echo Press Ctrl+C to stop the server
echo Close this window to stop the application
echo.
echo [4/4] Launching application...

REM Start the Flask application
python app.py

echo.
echo Server stopped.
pause