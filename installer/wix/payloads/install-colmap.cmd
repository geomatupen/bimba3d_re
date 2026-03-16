@echo off
setlocal ENABLEEXTENSIONS

set "LOG_ROOT=C:\ProgramData\Bimba3D\Logs"
if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%" >nul 2>nul
set "LOG_FILE=%LOG_ROOT%\colmap-wrapper.log"
echo ==== %DATE% %TIME% START install-colmap.cmd ====>>"%LOG_FILE%"

if "%~1"=="" (
  echo Usage: install-colmap.cmd ^<install-dir^> ^<colmap-zip^>
  echo ERROR missing arg1>>"%LOG_FILE%"
  exit /b 2
)

if "%~2"=="" (
  echo Usage: install-colmap.cmd ^<install-dir^> ^<colmap-zip^>
  echo ERROR missing arg2>>"%LOG_FILE%"
  exit /b 2
)

set "INSTALL_DIR=%~1"
set "ZIP_FILE=%~2"
set "PS_SCRIPT=%~dp0install-colmap.ps1"
echo INSTALL_DIR=%INSTALL_DIR%>>"%LOG_FILE%"
echo ZIP_FILE=%ZIP_FILE%>>"%LOG_FILE%"
echo PS_SCRIPT=%PS_SCRIPT%>>"%LOG_FILE%"

if not exist "%PS_SCRIPT%" (
  echo ERROR: Missing helper script: "%PS_SCRIPT%"
  echo ERROR helper script missing>>"%LOG_FILE%"
  exit /b 10
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -ZipPath "%ZIP_FILE%" -InstallDir "%INSTALL_DIR%" >>"%LOG_FILE%" 2>>&1
if errorlevel 1 (
  set "RC=1"
) else (
  set "RC=0"
)
echo RC=%RC%>>"%LOG_FILE%"
echo ==== %DATE% %TIME% END install-colmap.cmd ====>>"%LOG_FILE%"
exit /b %RC%
