@echo off
chcp 65001 >nul
title Forex ML Prediction System

:menu
cls
echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║         FOREX ML PREDICTION SYSTEM                     ║
echo ╠════════════════════════════════════════════════════════╣
echo ║                                                        ║
echo ║   1. TRAIN   - Entraîner le modèle LSTM                ║
echo ║                                                        ║
echo ║   2. PREDICT - Mode prédiction en temps réel           ║
echo ║                                                        ║
echo ║   3. DASHBOARD - Lancer le dashboard complet           ║
echo ║                                                        ║
echo ║   0. QUIT    - Quitter                                 ║
echo ║                                                        ║
echo ╚════════════════════════════════════════════════════════╝
echo.

set /p choice="Votre choix (0-3): "

if "%choice%"=="1" goto train
if "%choice%"=="2" goto predict
if "%choice%"=="3" goto dashboard
if "%choice%"=="0" goto quit

echo.
echo Choix invalide! Entrez 0, 1, 2 ou 3.
timeout /t 2 >nul
goto menu

:train
cls
echo.
echo ========================================
echo   MODE ENTRAINEMENT
echo ========================================
echo.
cd forex_ml
python train.py
cd ..
echo.
pause
goto menu

:predict
cls
echo.
echo ========================================
echo   MODE PREDICTION
echo ========================================
echo.
cd forex_ml
python predict_live.py
cd ..
echo.
pause
goto menu

:dashboard
cls
echo.
echo ========================================
echo   LANCEMENT DU DASHBOARD COMPLET
echo ========================================
echo.

REM Check if Next.js dependencies are installed
if not exist "next-forex-app\node_modules" (
    echo Installation des dépendances Next.js...
    cd next-forex-app
    call npm install
    cd ..
    echo.
)

REM Start Next.js dev server in a new window
echo Démarrage du dashboard Next.js...
start "Forex Dashboard - Next.js" cmd /k "cd next-forex-app && npm run dev"

REM Wait a bit for the server to start
timeout /t 3 /nobreak >nul

REM Start Python MT5 script in a new window
echo Démarrage du collecteur MT5...
start "Forex Dashboard - MT5 Script" cmd /k "python mt5_forex_live.py"

REM Wait a bit before opening browser
timeout /t 5 /nobreak >nul

REM Open the dashboard in default browser
echo Ouverture du dashboard dans le navigateur...
start http://localhost:3005

echo.
echo ========================================
echo   Services démarrés!
echo ========================================
echo   - Dashboard: http://localhost:3005
echo   - Fermez les fenêtres pour arrêter
echo ========================================
echo.
pause
goto menu

:quit
cls
echo.
echo Au revoir!
timeout /t 1 >nul
exit
