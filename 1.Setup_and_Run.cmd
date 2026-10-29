
@echo off
setlocal
title Multimodal Lung Disease Classifier

REM ============================================================
REM Configuration - paths relative to this script
REM ============================================================

cd /d "%~dp0"

set "VENV=%~dp0.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "REQUIREMENTS=%~dp0requirements.txt"
set "GUI=%~dp0GUI\GUI_fusion.py"

echo ============================================
echo Multimodal Lung Disease Classifier
echo ============================================

REM ============================================================
REM 1. Check required project files
REM ============================================================

if not exist "%REQUIREMENTS%" (
    echo ERROR: requirements.txt not found.
    goto :error
)

if not exist "%GUI%" (
    echo ERROR: GUI\GUI.py not found.
    goto :error
)

REM ============================================================
REM 2. Create virtual environment if missing
REM ============================================================

if not exist "%PYTHON%" (
    echo Creating Python virtual environment...

    py -3 -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment.
        echo Check that Python is installed.
        goto :error
    )
)

REM ============================================================
REM 3. Activate virtual environment
REM ============================================================

call "%VENV%\Scripts\activate.bat"

if errorlevel 1 (
    echo ERROR: Virtual environment activation failed.
    goto :error
)

echo.
echo Active Python:
python --version

REM ============================================================
REM 4. Install dependencies
REM ============================================================

echo.
echo Installing required packages...

"%PYTHON%" -m pip install -r "%REQUIREMENTS%"

if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    goto :error
)

REM ============================================================
REM 5. Start Gradio application
REM ============================================================

echo.
echo Starting Gradio GUI...
echo.

"%PYTHON%" "%GUI%"

if errorlevel 1 (
    echo ERROR: GUI exited with an error.
    goto :error
)

goto :end

:error
echo.
echo Setup or execution failed.
pause
exit /b 1

:end
echo.
echo Application closed.
pause
endlocal
