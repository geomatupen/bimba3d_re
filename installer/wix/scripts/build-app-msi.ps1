param(
    [string]$WxsPath = "$PSScriptRoot\..\app-msi\Product.wxs",
    [string]$StagingDir = "$PSScriptRoot\..\staging",
    [string]$OutputMsi = "$PSScriptRoot\..\payloads\Bimba3D.msi",
    [switch]$SkipFrontendBuild
)

$ErrorActionPreference = 'Stop'

$WxsPath = [System.IO.Path]::GetFullPath($WxsPath)
$StagingDir = [System.IO.Path]::GetFullPath($StagingDir)
$OutputMsi = [System.IO.Path]::GetFullPath($OutputMsi)

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$backendRoot = Join-Path $repoRoot "bimba3d_backend"
$frontendRoot = Join-Path $repoRoot "bimba3d_frontend"
$frontendDist = Join-Path $frontendRoot "dist"

if (-not (Test-Path $backendRoot)) {
    throw "Backend folder not found: $backendRoot"
}

if (-not (Test-Path $frontendRoot)) {
    throw "Frontend folder not found: $frontendRoot"
}

$wix = Get-Command wix -ErrorAction SilentlyContinue
if (-not $wix) {
    throw "wix.exe not found in PATH. Install WiX CLI first."
}

if (-not $SkipFrontendBuild) {
    Push-Location $frontendRoot
    try {
        & npm run build
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend build failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

if (-not (Test-Path $frontendDist)) {
    throw "Frontend dist not found: $frontendDist"
}

if (Test-Path $StagingDir) {
    Remove-Item -Path $StagingDir -Recurse -Force
}
New-Item -ItemType Directory -Path $StagingDir | Out-Null

$stagingBackend = Join-Path $StagingDir "bimba3d_backend"
$stagingFrontendRoot = Join-Path $StagingDir "bimba3d_frontend"
$stagingFrontendDist = Join-Path $stagingFrontendRoot "dist"

New-Item -ItemType Directory -Path $stagingBackend | Out-Null
New-Item -ItemType Directory -Path $stagingFrontendRoot | Out-Null

Copy-Item -Path (Join-Path $backendRoot "app") -Destination $stagingBackend -Recurse -Force
Copy-Item -Path (Join-Path $backendRoot "worker") -Destination $stagingBackend -Recurse -Force
Copy-Item -Path (Join-Path $backendRoot "requirements.local.txt") -Destination $stagingBackend -Force
Copy-Item -Path (Join-Path $backendRoot "requirements.windows.txt") -Destination $stagingBackend -Force
Copy-Item -Path (Join-Path $backendRoot "requirements.txt") -Destination $stagingBackend -Force
Copy-Item -Path $frontendDist -Destination $stagingFrontendRoot -Recurse -Force

$launcherPath = Join-Path $StagingDir "start_bimba3d.bat"
@"
@echo off
setlocal

cd /d "%~dp0"

set "PYEXE="
set "PYEXE_ARGS="
if exist "C:\Program Files\Python312\python.exe" (
  set "PYEXE=C:\Program Files\Python312\python.exe"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
     where py >nul 2>nul
     if errorlevel 1 (
        echo Python is not installed. Please install Python 3.12+ and try again.
        pause
        exit /b 1
     )
      set "PYEXE=py"
      set "PYEXE_ARGS=-3"
  ) else (
     set "PYEXE=python"
  )
)

"%PYEXE%" %PYEXE_ARGS% -c "import sys; raise SystemExit(0 if sys.maxsize > 2**32 else 1)" >nul 2>nul
if errorlevel 1 (
    echo Python interpreter is not 64-bit. Please install/use x64 Python and try again.
    pause
    exit /b 1
)

set "RUNTIME_ROOT=%ProgramData%\Bimba3D\runtime"
set "VENV_DIR=%RUNTIME_ROOT%\.venv"

if not exist "%RUNTIME_ROOT%" (
    mkdir "%RUNTIME_ROOT%"
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating Python virtual environment...
    "%PYEXE%" %PYEXE_ARGS% -m venv "%VENV_DIR%"
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Failed to locate venv python at "%VENV_PY%".
    pause
    exit /b 1
)

set "PYTHONNOUSERSITE=1"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
set "TORCH_VERSION=2.5.1+cu121"
set "GSPLAT_VERSION=1.5.3"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
set "CUDA_PATH=%CUDA_HOME%"
set "DISTUTILS_USE_SDK=1"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

"%VENV_PY%" -m pip install --upgrade pip setuptools==69.5.1 wheel
if errorlevel 1 (
    echo Failed to install Python packaging tooling.
    pause
    exit /b 1
)

"%VENV_PY%" -m pip install --index-url %TORCH_INDEX% --force-reinstall "torch==%TORCH_VERSION%"
if errorlevel 1 (
    echo Failed to install CUDA-enabled torch %TORCH_VERSION%.
    pause
    exit /b 1
)

"%VENV_PY%" -m pip install -r bimba3d_backend\requirements.windows.txt
if errorlevel 1 (
    echo Failed to install backend requirements.
    pause
    exit /b 1
)

"%VENV_PY%" -c "import torch; import gsplat.cuda._wrapper as w; ok=torch.cuda.is_available() and (getattr(w,'_C',None) is not None or hasattr(w,'_make_lazy_cuda_obj')); raise SystemExit(0 if ok else 1)" >nul 2>nul
if errorlevel 1 (
    echo CUDA-enabled torch/gsplat not ready. Reinstalling pinned training dependencies...
    "%VENV_PY%" -m pip install --index-url %TORCH_INDEX% --force-reinstall "torch==%TORCH_VERSION%"
    if errorlevel 1 (
        echo Failed to reinstall CUDA-enabled torch %TORCH_VERSION%.
        pause
        exit /b 1
    )
    "%VENV_PY%" -m pip install --force-reinstall ninja
    if errorlevel 1 (
        echo Failed to install ninja.
        pause
        exit /b 1
    )
    "%VENV_PY%" -m pip install --force-reinstall --no-binary=gsplat "gsplat==%GSPLAT_VERSION%" --no-build-isolation -v
    if errorlevel 1 (
        echo Failed to build/install gsplat %GSPLAT_VERSION%.
        pause
        exit /b 1
    )
    "%VENV_PY%" -c "import torch; import gsplat.cuda._wrapper as w; ok=torch.cuda.is_available() and (getattr(w,'_C',None) is not None or hasattr(w,'_make_lazy_cuda_obj')); raise SystemExit(0 if ok else 1)"
    if errorlevel 1 (
        echo CUDA is still unavailable in runtime venv after reinstall.
        echo Please verify NVIDIA driver + CUDA Toolkit + Visual Studio Build Tools installation.
        pause
        exit /b 1
    )
)

set "WORKER_MODE=local"
set "FRONTEND_DIST=%CD%\bimba3d_frontend\dist"
set "BIMBA3D_DATA_DIR=%ProgramData%\Bimba3D\data\projects"

if not exist "%BIMBA3D_DATA_DIR%" (
    mkdir "%BIMBA3D_DATA_DIR%"
)

start "" http://127.0.0.1:8005
"%VENV_PY%" -m uvicorn bimba3d_backend.app.main:app --host 127.0.0.1 --port 8005
"@ | Set-Content -Path $launcherPath -Encoding ASCII

$readmePath = Join-Path $StagingDir "README.txt"
@(
    "Bimba3D packaged application"
    "Generated: $(Get-Date -Format o)"
    ""
    "Start the app using start_bimba3d.bat"
    "The launcher creates a local .venv on first run and installs Python dependencies."
) | Set-Content -Path $readmePath -Encoding UTF8

Push-Location (Split-Path -Parent $WxsPath)
try {
    & wix build "$(Split-Path -Leaf $WxsPath)" -o $OutputMsi
    if ($LASTEXITCODE -ne 0) {
        throw "wix build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "App MSI created: $OutputMsi" -ForegroundColor Green