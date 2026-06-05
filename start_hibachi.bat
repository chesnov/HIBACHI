@echo off
setlocal enabledelayedexpansion
:: Ensure we are running in the HIBACHI folder
cd /d "%~dp0"

echo Checking for updates...
call git fetch

:: Check how many commits behind the remote branch we are
set BEHIND=0
for /f %%i in ('git rev-list HEAD..@{u} --count 2^>nul') do set BEHIND=%%i

if !BEHIND! GTR 0 (
    echo ========================================================
    echo  [NOTICE] HIBACHI is behind by !BEHIND! updates!
    echo ========================================================
    set /p UPDATE="Would you like to download and install the update now? (Y/N): "
    if /i "!UPDATE!"=="Y" (
        echo.
        echo Pulling latest code...
        call git pull
        
        echo.
        echo Updating conda environment ^(this may take a minute^)...
        call conda env update -f environment.yaml --prune
    )
) else (
    echo HIBACHI is up to date!
)

echo.
echo Launching HIBACHI GUI...
call conda activate hibachi
call python segment.py

:: Keep the terminal open if the app crashes so the user can see the error
if %ERRORLEVEL% NEQ 0 pause