@echo off
REM Cross-View Gait Recognition Windows One-Click Runner
echo =======================================================
echo Cross-View Gait Recognition (ResNet-50 + CBAM + HPM)
echo =======================================================

echo Checking Python environment...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo.
echo Select an option:
echo   1. Run Training Pipeline (12 Epochs) and Evaluation
echo   2. Run Evaluation Only
echo   3. Run Fast Test Run (2 Epochs on CPU)
echo   4. Exit
echo.
set /p choice="Enter choice [1-4]: "

if "%choice%"=="1" (
    echo Starting Full Training Pipeline...
    python main.py --mode all
) else if "%choice%"=="2" (
    echo Starting Evaluation Only...
    python main.py --mode eval
) else if "%choice%"=="3" (
    echo Starting Fast Test Run...
    python main.py --mode all --cpu --epochs 2 --fs_epochs 1
) else (
    echo Exiting...
)

pause
