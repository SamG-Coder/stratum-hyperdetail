@echo off
cd /d "%~dp0"
call npm run build
if errorlevel 1 (pause & exit /b 1)
node server.mjs
pause
