@echo off
chcp 65001 >nul
cd /d "%~dp0.."

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
node --env-file-if-exists=.env src\app.mjs
if errorlevel 1 pause
