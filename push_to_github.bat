@echo off
echo ========================================================
echo Pushing Cross-View Gait Recognition to GitHub
echo ========================================================

REM Add MinGit to PATH if not globally registered yet
set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Git.MinGit_Microsoft.Winget.Source_8wekyb3d8bbwe\cmd;C:\Program Files\Git\cmd;%PATH%"

git --version
if errorlevel 1 (
    echo [ERROR] Git was not found. Please install Git from https://git-scm.com/download/win
    pause
    exit /b 1
)

echo.
echo Pushing code to https://github.com/Satishkumar683/capstoneV.git ...
git push -u origin main

if errorlevel 1 (
    echo.
    echo [NOTE] If you were prompted to log in, please complete the sign-in.
    echo If pushing was denied, ensure you have write access to the repository.
) else (
    echo.
    echo ========================================================
    echo [SUCCESS] Code successfully pushed to GitHub!
    echo Check your repository at: https://github.com/Satishkumar683/capstoneV
    echo ========================================================
)

pause
