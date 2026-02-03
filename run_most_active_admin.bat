@echo off
:: One-click script to run most_active.py as admin in the correct environment
setlocal

:: Get current script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: Activate venv and run script as admin
:: This will prompt for admin if not already elevated
powershell -Command "Start-Process cmd -ArgumentList '/c .\\.venv313\\Scripts\\activate && python most_active.py & pause' -Verb RunAs"
