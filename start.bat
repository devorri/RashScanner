@echo off
title Rashilience 100% AI Scanner
cd /d "C:\Users\Tom Pc\Desktop\Commissions\RashScannerAI"

set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"

rem Start Flask without its own browser thread; this launcher opens one window after readiness.
start "Rashilience Server" /b "%PYTHON%" app.py --no-browser

rem Wait until Flask has finished loading the AI model and is accepting requests.
for /l %%I in (1,1,30) do (
	powershell -NoProfile -ExecutionPolicy Bypass -Command "try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 http://127.0.0.1:5000/).StatusCode | Out-Null; exit 0 } catch { exit 1 }"
	if not errorlevel 1 goto :launch
	timeout /t 1 /nobreak >nul
)

echo Rashilience server did not start on port 5000.
pause
exit /b 1

:launch
where msedge.exe >nul 2>&1 && start "" msedge.exe --kiosk http://127.0.0.1:5000 --edge-kiosk-type=fullscreen || start "" http://127.0.0.1:5000
exit /b 0
