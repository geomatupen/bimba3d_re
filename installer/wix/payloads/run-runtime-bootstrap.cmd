@echo off
setlocal

set "WRAPPER_LOG=%ProgramData%\Bimba3D\Logs\runtime-bootstrap-wrapper.log"
if not exist "%ProgramData%\Bimba3D\Logs" mkdir "%ProgramData%\Bimba3D\Logs" >nul 2>nul

set "INSTALL_DIR=%~1"
set "BOOTSTRAP_PHASE=%~2"

if /I "%~nx1"=="run-runtime-bootstrap.cmd" (
    set "INSTALL_DIR=%~2"
    set "BOOTSTRAP_PHASE=%~3"
)

if "%BOOTSTRAP_PHASE%"=="" (
    if /I "%INSTALL_DIR%"=="prepare" set "BOOTSTRAP_PHASE=%INSTALL_DIR%"
    if /I "%INSTALL_DIR%"=="torch" set "BOOTSTRAP_PHASE=%INSTALL_DIR%"
    if /I "%INSTALL_DIR%"=="gsplat" set "BOOTSTRAP_PHASE=%INSTALL_DIR%"
    if /I "%INSTALL_DIR%"=="requirements" set "BOOTSTRAP_PHASE=%INSTALL_DIR%"
    if not "%BOOTSTRAP_PHASE%"=="" set "INSTALL_DIR="
)

if "%INSTALL_DIR%"=="" (
    set "INSTALL_DIR=%ProgramFiles%\Bimba3D"
)

if not exist "%INSTALL_DIR%\bimba3d_backend\requirements.windows.txt" (
    if exist "%ProgramFiles(x86)%\Bimba3D\bimba3d_backend\requirements.windows.txt" (
        set "INSTALL_DIR=%ProgramFiles(x86)%\Bimba3D"
    )
)

if "%BOOTSTRAP_PHASE%"=="" (
    echo Missing bootstrap phase argument.
    >>"%WRAPPER_LOG%" echo [%date% %time%] ERROR missing phase. raw1="%~1" raw2="%~2" raw3="%~3"
    exit /b 1
)

>>"%WRAPPER_LOG%" echo [%date% %time%] START raw1="%~1" raw2="%~2" raw3="%~3" install="%INSTALL_DIR%" phase="%BOOTSTRAP_PHASE%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0runtime-bootstrap.ps1" -InstallDir "%INSTALL_DIR%" -Phase "%BOOTSTRAP_PHASE%"
>>"%WRAPPER_LOG%" echo [%date% %time%] EXIT code=%ERRORLEVEL% phase="%BOOTSTRAP_PHASE%"
exit /b %ERRORLEVEL%
