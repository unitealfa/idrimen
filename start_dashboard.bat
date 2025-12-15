@echo off
echo ========================================
echo   Forex Live Dashboard - Startup
echo ========================================
echo.

REM Check if Next.js dependencies are installed
if not exist "next-forex-app\node_modules" (
    echo Installing Next.js dependencies...
    cd next-forex-app
    call npm install
    cd ..
    echo.
)

REM Start Next.js dev server in a new window
echo Starting Next.js dashboard...
start "Forex Dashboard - Next.js" cmd /k "cd next-forex-app && npm run dev"

REM Wait a bit for the server to start
timeout /t 3 /nobreak >nul

REM Start Python MT5 script in a new window
echo Starting MT5 data collector...
start "Forex Dashboard - MT5 Script" cmd /k "python mt5_forex_live.py"

REM Wait a bit before opening browser
timeout /t 5 /nobreak >nul

REM Open the dashboard in default browser
echo Opening dashboard in browser...
start http://localhost:3005

echo.
echo ========================================
echo   Both services are running!
echo ========================================
echo   - Next.js Dashboard: http://localhost:3005
echo   - Close this window to keep services running
echo   - Close service windows to stop them
echo ========================================
echo.
pause
