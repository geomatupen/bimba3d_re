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

set "PYEXE=python"
where python >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 (
    echo Python is not installed. Please install Python 3.12+ and try again.
    pause
    exit /b 1
  )
  set "PYEXE=py -3"
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  %PYEXE% -m venv .venv
)

call .venv\Scripts\activate

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Failed to locate venv python at "%VENV_PY%".
    pause
    exit /b 1
)

set "PYTHONNOUSERSITE=1"

"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
"%VENV_PY%" -m pip install -r bimba3d_backend\requirements.windows.txt

"%VENV_PY%" -c "import torch; import gsplat.cuda._wrapper as w; ok=torch.cuda.is_available() and (getattr(w,'_C',None) is not None or hasattr(w,'_make_lazy_cuda_obj')); raise SystemExit(0 if ok else 1)" >nul 2>nul
if errorlevel 1 (
    echo CUDA-enabled torch/gsplat not ready. Installing training dependencies...
    "%VENV_PY%" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
    "%VENV_PY%" -m pip install ninja
    "%VENV_PY%" -m pip install --no-binary=gsplat gsplat --no-build-isolation -v
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