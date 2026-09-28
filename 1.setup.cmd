@echo off

REM Create virtual environment (only if it does not exist)
if not exist ".venv" (
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate

REM Install dependencies
REM pip install -r requirements.txt
python -m pip install -r requirements.txt

pause