@echo off
echo Starting Bangla NLP Demo Website...
echo.

cd /d "%~dp0backend"
start "FastAPI Server" cmd /k "..\\.venv\\Scripts\\python.exe app.py"

timeout /t 3 /nobreak >nul

echo Opening browser...
start chrome http://localhost:8000

echo.
echo Website running at http://localhost:8000
echo Press Ctrl+C in the server window to stop.
