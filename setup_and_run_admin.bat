@echo off
echo ========================================================================
echo    SENTIMENT ANALYSIS APP - ADMINISTRATOR SETUP AND RUN
echo ========================================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✓ Running as Administrator - Good!
    echo.
) else (
    echo ❌ NOT running as Administrator!
    echo.
    echo Please right-click this file and select "Run as Administrator"
    echo This is required to configure Windows Firewall.
    echo.
    pause
    exit /b 1
)

echo [1/5] Configuring Windows Firewall...
echo.

REM Remove any existing rule first
netsh advfirewall firewall delete rule name="Sentiment Analysis App" >nul 2>&1

REM Add new firewall rule
netsh advfirewall firewall add rule name="Sentiment Analysis App" dir=in action=allow protocol=TCP localport=5000 >nul 2>&1

if %errorLevel% == 0 (
    echo ✓ Windows Firewall configured successfully!
    echo   Port 5000 is now open for incoming connections.
) else (
    echo ❌ Failed to configure Windows Firewall.
    echo   You may need to configure it manually.
)

echo.
echo [2/5] Getting network configuration...

REM Get the current WiFi IP address
for /f "tokens=14" %%a in ('ipconfig ^| findstr /C:"IPv4 Address"') do (
    set "WIFI_IP=%%a"
    goto :found_ip
)
:found_ip

echo.
echo ========================================================================
echo                    NETWORK ACCESS INFORMATION
echo ========================================================================
echo.
echo Your computer's IP address: %WIFI_IP%
echo.
echo ACCESS URLS:
echo   Local (this computer): http://localhost:5000
echo   Network access:        http://%WIFI_IP%:5000
echo.
echo MOBILE DEVICES:
echo   1. Connect your phone/tablet to the same WiFi network
echo   2. Open browser and go to: http://%WIFI_IP%:5000
echo.
echo OTHER COMPUTERS:
echo   1. Ensure they're on the same network
echo   2. Open browser and go to: http://%WIFI_IP%:5000
echo.
echo ========================================================================

echo.
echo [3/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
) else (
    python --version
    echo ✓ Python is available
)

echo.
echo [4/5] Installing/updating dependencies...
pip install -r requirements.txt --quiet --disable-pip-version-check
echo ✓ Dependencies updated

echo.
echo [5/5] Starting Sentiment Analysis Application...
echo.
echo ========================================================================
echo                    🚀 APPLICATION STARTING 🚀
echo ========================================================================
echo.
echo The application is now accessible at:
echo   - http://localhost:5000 (local)
echo   - http://%WIFI_IP%:5000 (network)
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the application
python app.py

echo.
echo Application stopped.
pause