@echo off
setlocal ENABLEEXTENSIONS

set "LOG_ROOT=C:\ProgramData\Bimba3D\Logs"
if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%" >nul 2>nul
set "LOG_FILE=%LOG_ROOT%\prereq-check-wrapper.log"
echo ==== %DATE% %TIME% START run-prereq-check.cmd ====>>"%LOG_FILE%"

set "PS_SCRIPT=%~dp0prereq-check.ps1"
echo PS_SCRIPT=%PS_SCRIPT%>>"%LOG_FILE%"

if not exist "%PS_SCRIPT%" (
  echo ERROR: Missing helper script: "%PS_SCRIPT%"
  echo ERROR helper script missing>>"%LOG_FILE%"
  exit /b 10
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" >>"%LOG_FILE%" 2>>&1
if errorlevel 1 (
  set "RC=1"
) else (
  set "RC=0"
)
echo RC=%RC%>>"%LOG_FILE%"
echo ==== %DATE% %TIME% END run-prereq-check.cmd ====>>"%LOG_FILE%"
exit /b %RC%
