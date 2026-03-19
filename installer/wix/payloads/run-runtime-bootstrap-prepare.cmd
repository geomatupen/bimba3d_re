@echo off
setlocal

set "WRAPPER_LOG=%ProgramData%\Bimba3D\Logs\runtime-bootstrap-wrapper.log"
set "PHASE_LOG=%ProgramData%\Bimba3D\Logs\runtime-bootstrap-prepare.log"
set "BOOTSTRAP_ROOT=%ProgramData%\Bimba3D\runtime\bootstrap"
if not exist "%ProgramData%\Bimba3D\Logs" mkdir "%ProgramData%\Bimba3D\Logs" >nul 2>nul
if not exist "%BOOTSTRAP_ROOT%" mkdir "%BOOTSTRAP_ROOT%" >nul 2>nul

set "INSTALL_DIR=%~1"
if "%INSTALL_DIR%"=="" set "INSTALL_DIR=%ProgramFiles%\Bimba3D"
if not exist "%INSTALL_DIR%\bimba3d_backend\requirements.windows.txt" (
	if exist "%ProgramFiles(x86)%\Bimba3D\bimba3d_backend\requirements.windows.txt" (
		set "INSTALL_DIR=%ProgramFiles(x86)%\Bimba3D"
	)
)

>>"%WRAPPER_LOG%" echo [%date% %time%] START install="%INSTALL_DIR%" phase="prepare"
copy /Y "%~dp0runtime-bootstrap.ps1" "%BOOTSTRAP_ROOT%\runtime-bootstrap.ps1" >nul
if errorlevel 1 (
	>>"%WRAPPER_LOG%" echo [%date% %time%] ERROR copy runtime-bootstrap.ps1 failed
	exit /b 1
)

set "RESOLVER_SRC="
if exist "%~dp0compatibility-resolver.ps1" set "RESOLVER_SRC=%~dp0compatibility-resolver.ps1"
if "%RESOLVER_SRC%"=="" if exist "%~dp0compatibility-resolver-runtime.ps1" set "RESOLVER_SRC=%~dp0compatibility-resolver-runtime.ps1"
if "%RESOLVER_SRC%"=="" if exist "%~dp0compatibility-resolver-prereq.ps1" set "RESOLVER_SRC=%~dp0compatibility-resolver-prereq.ps1"
if "%RESOLVER_SRC%"=="" if exist "%~dp0compatibility-resolver-colmap.ps1" set "RESOLVER_SRC=%~dp0compatibility-resolver-colmap.ps1"
if "%RESOLVER_SRC%"=="" (
	>>"%WRAPPER_LOG%" echo [%date% %time%] ERROR compatibility resolver payload not found
	exit /b 1
)

copy /Y "%RESOLVER_SRC%" "%BOOTSTRAP_ROOT%\compatibility-resolver.ps1" >nul
if errorlevel 1 (
	>>"%WRAPPER_LOG%" echo [%date% %time%] ERROR copy compatibility resolver failed from "%RESOLVER_SRC%"
	exit /b 1
)

set "MATRIX_SRC="
if exist "%~dp0compatibility-matrix.json" set "MATRIX_SRC=%~dp0compatibility-matrix.json"
if "%MATRIX_SRC%"=="" if exist "%~dp0compatibility-matrix-runtime.json" set "MATRIX_SRC=%~dp0compatibility-matrix-runtime.json"
if "%MATRIX_SRC%"=="" if exist "%~dp0compatibility-matrix-prereq.json" set "MATRIX_SRC=%~dp0compatibility-matrix-prereq.json"
if "%MATRIX_SRC%"=="" if exist "%~dp0compatibility-matrix-colmap.json" set "MATRIX_SRC=%~dp0compatibility-matrix-colmap.json"
if "%MATRIX_SRC%"=="" (
	>>"%WRAPPER_LOG%" echo [%date% %time%] ERROR compatibility matrix payload not found
	exit /b 1
)

copy /Y "%MATRIX_SRC%" "%BOOTSTRAP_ROOT%\compatibility-matrix.json" >nul
if errorlevel 1 (
	>>"%WRAPPER_LOG%" echo [%date% %time%] ERROR copy compatibility matrix failed from "%MATRIX_SRC%"
	exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%BOOTSTRAP_ROOT%\runtime-bootstrap.ps1" -InstallDir "%INSTALL_DIR%" -Phase "prepare" >>"%PHASE_LOG%" 2>&1
set "RC=%ERRORLEVEL%"
>>"%WRAPPER_LOG%" echo [%date% %time%] EXIT code=%RC% phase="prepare"
exit /b %RC%
