
@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title Multimodal Lung Disease Classifier - Setup and Run

echo ============================================================
echo   MULTIMODAL LUNG DISEASE CLASSIFIER
echo   Environment Setup and GUI Launcher
echo ============================================================
echo.

:: ============================================================
:: 1. CHECK PYTHON
:: ============================================================

echo [1/5] Checking Python installation...

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    goto :failed
)

python --version
if errorlevel 1 goto :failed
echo [OK] Python available.
echo.


:: ============================================================
:: 2. CREATE VIRTUAL ENVIRONMENT
:: ============================================================

echo [2/5] Checking virtual environment...

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :failed
) else (
    echo [OK] Existing virtual environment found.
)

echo.


:: ============================================================
:: 3. ACTIVATE VIRTUAL ENVIRONMENT
:: ============================================================

echo [3/5] Activating virtual environment...

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :failed

echo [OK] Virtual environment activated.
python -c "import sys; print('Python:', sys.executable)"
echo.


:: ============================================================
:: 4. INSTALL DEPENDENCIES
:: ============================================================

echo [4/5] Installing required dependencies...

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found.
    goto :failed
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    goto :failed
)

echo [OK] Dependencies installed.
echo.


:: ============================================================
:: 5. LAUNCH GRADIO GUI
:: ============================================================

echo [5/5] Starting Gradio application...

if not exist "GUI\GUI_demo.py" (
    echo [ERROR] GUI\GUI_demo.py not found.
    goto :failed
)

echo [INFO] Initializing application...
echo [INFO] Local URL: http://127.0.0.1:7860
echo [INFO] Press CTRL+C to stop the server.
echo.

python "GUI\GUI_demo.py"

if errorlevel 1 (
    echo [ERROR] GUI exited with an error.
    goto :failed
)

goto :complete


:: ============================================================
:: ERROR HANDLING
:: ============================================================

:failed
echo.
echo ============================================================
echo   SETUP OR EXECUTION FAILED
echo   Review the error messages above.
echo ============================================================
pause
exit /b 1


:: ============================================================
:: COMPLETION
:: ============================================================

:complete
echo.
echo ============================================================
echo   APPLICATION STOPPED
echo ============================================================
pause
exit /b 0
