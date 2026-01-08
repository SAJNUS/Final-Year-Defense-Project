@echo off
echo Starting Bangla NLP Demo Website...
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
    echo.
)

REM Activate virtual environment and install requirements
echo Installing/Updating dependencies (this may take a few minutes)...
call .venv\Scripts\activate.bat
pip install -r requirements.txt
echo.

REM Run the application
cd /d "%~dp0backend"
start "FastAPI Server" cmd /k "..\.venv\Scripts\python.exe app.py"

timeout /t 3 /nobreak >nul

echo Opening browser...
start chrome http://localhost:8000

echo.
echo Website running at http://localhost:8000
echo Press Ctrl+C in the server window to stop.
