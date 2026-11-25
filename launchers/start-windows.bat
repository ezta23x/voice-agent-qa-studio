@echo off
chcp 65001 >nul
cd /d "%~dp0.."

if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" set "%%A=%%~B"
  )
)

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js not found. Please install Node.js from https://nodejs.org
  pause
  exit /b 1
)

if not exist "node_modules\playwright" (
  echo Installing dependencies...
  call npm install
)

echo Starting Voice Agent QA Studio...
node src\app.mjs
if errorlevel 1 pause
