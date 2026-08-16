@echo off
setlocal

rem poke.AI launcher - runs the AI against the running VBA-RR emulator.
rem Logs all output to ai.log in the project root.

cd /d "%~dp0ai"
set "LOG=%~dp0ai.log"

tasklist /FI "IMAGENAME eq VBA_VS2010_V82XD.exe" | find /I "VBA_VS2010_V82XD" >nul
if errorlevel 1 (
    echo WARNING: VBA_VS2010_V82XD.exe emulator is NOT running.
    echo Start the emulator, load the ROM, and finish the intro BEFORE running the AI.
    echo Press any key to continue anyway, or close this window to abort.
    pause >nul
)

echo [start.bat] %date% %time% starting poke.AI - logging to %LOG%
if exist "%LOG%" del "%LOG%"
"%~dp0.venv\Scripts\python.exe" standalone_backend.py > "%LOG%" 2>&1
echo [start.bat] %date% %time% poke.AI exited with code %errorlevel%
endlocal
