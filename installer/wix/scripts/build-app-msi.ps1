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

$bootstrapPath = Join-Path $StagingDir "bootstrap_runtime.bat"
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
set "BOOTSTRAP_STATE=%RUNTIME_ROOT%\bootstrap-state.txt"

if not exist "%RUNTIME_ROOT%" (
    mkdir "%RUNTIME_ROOT%"
)

set "PYTHONNOUSERSITE=1"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
set "TORCH_VERSION=2.5.1+cu121"
set "TORCH_CPU_VERSION=2.5.1"
set "GSPLAT_VERSION=1.5.3"
set "CUDA_HOME="
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "CUDA_HOME=%CUDA_PATH%"
if not defined CUDA_HOME for %%V in (12.8 12.7 12.6 12.5 12.4 12.3 12.2 12.1 12.0) do (
    if not defined CUDA_HOME if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%V\bin\nvcc.exe" set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%V"
)
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
set "CUDA_PATH=%CUDA_HOME%"
set "HAS_CUDA_TOOLKIT=0"
if exist "%CUDA_HOME%\bin\nvcc.exe" set "HAS_CUDA_TOOLKIT=1"
set "DISTUTILS_USE_SDK=1"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
set "NEED_BOOTSTRAP=0"
set "TORCH_FLAVOR=cu121"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    set "NEED_BOOTSTRAP=1"
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%BOOTSTRAP_STATE%" (
    set "NEED_BOOTSTRAP=1"
)

if exist "%BOOTSTRAP_STATE%" (
    findstr /C:"TORCH_VERSION=%TORCH_VERSION%" "%BOOTSTRAP_STATE%" >nul 2>nul
    if errorlevel 1 set "NEED_BOOTSTRAP=1"
    findstr /C:"GSPLAT_VERSION=%GSPLAT_VERSION%" "%BOOTSTRAP_STATE%" >nul 2>nul
    if errorlevel 1 set "NEED_BOOTSTRAP=1"
    if "%HAS_CUDA_TOOLKIT%"=="1" (
        findstr /C:"TORCH_FLAVOR=cpu" "%BOOTSTRAP_STATE%" >nul 2>nul
        if not errorlevel 1 set "NEED_BOOTSTRAP=1"
    )
)

if "%NEED_BOOTSTRAP%"=="0" (
    "%VENV_PY%" -c "import fastapi,uvicorn" >nul 2>nul
    if errorlevel 1 set "NEED_BOOTSTRAP=1"
    if "%HAS_CUDA_TOOLKIT%"=="1" (
        "%VENV_PY%" -c "import torch; import gsplat.cuda._wrapper as w; ok=torch.cuda.is_available() and (getattr(w,'_C',None) is not None or hasattr(w,'_make_lazy_cuda_obj')); raise SystemExit(0 if ok else 1)" >nul 2>nul
        if errorlevel 1 set "NEED_BOOTSTRAP=1"
    )
)

if "%NEED_BOOTSTRAP%"=="1" (
    echo Preparing runtime environment...

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

    "%VENV_PY%" -m pip install --upgrade pip setuptools==69.5.1 wheel
    if errorlevel 1 (
        echo Failed to install Python packaging tooling.
        pause
        exit /b 1
    )

    "%VENV_PY%" -m pip install --index-url %TORCH_INDEX% --force-reinstall "torch==%TORCH_VERSION%"
    if errorlevel 1 (
        echo CUDA torch install failed.
        set "TORCH_FLAVOR=cpu"
        "%VENV_PY%" -m pip install --force-reinstall "torch==%TORCH_CPU_VERSION%"
        if errorlevel 1 (
            echo Failed to install torch runtime.
            pause
            exit /b 1
        )
        if "%HAS_CUDA_TOOLKIT%"=="1" (
            echo Warning: CUDA toolkit is present but CUDA torch install failed. gsplat training will fail until CUDA torch is repaired.
        )
    ) else (
        "%VENV_PY%" -c "import torch; raise SystemExit(0 if getattr(torch.version,'cuda',None) else 1)" >nul 2>nul
        if errorlevel 1 (
            set "TORCH_FLAVOR=cpu"
            if "%HAS_CUDA_TOOLKIT%"=="1" (
                echo Warning: torch installed but reports CPU-only build. gsplat training will fail until CUDA torch is repaired.
            )
        ) else (
            set "TORCH_FLAVOR=cu121"
        )
    )

    if /I "%TORCH_FLAVOR%"=="cu121" (
        set "NEED_VC_X64=1"
        where cl >nul 2>nul
        if not errorlevel 1 (
            where cl > "%TEMP%\bimba3d_cl_where.txt" 2>nul
            findstr /I /C:"Hostx64\x64\cl.exe" "%TEMP%\bimba3d_cl_where.txt" >nul 2>nul
            if not errorlevel 1 set "NEED_VC_X64=0"
            del "%TEMP%\bimba3d_cl_where.txt" >nul 2>nul
        )

        if "%NEED_VC_X64%"=="1" (
            set "VSWHERE="
            set "VSINSTALL="
            set "VCVARS64="
            set "VSDEVCMD="

            for /f "delims=" %%I in ('where vswhere 2^>nul') do if not defined VSWHERE set "VSWHERE=%%~fI"
            if not defined VSWHERE if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
            if not defined VSWHERE if exist "%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

            if defined VSWHERE (
                for /f "delims=" %%I in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do if not defined VSINSTALL set "VSINSTALL=%%~I"
                if defined VSINSTALL (
                    set "VCVARS64=%VSINSTALL%\VC\Auxiliary\Build\vcvarsall.bat"
                    set "VSDEVCMD=%VSINSTALL%\Common7\Tools\VsDevCmd.bat"
                )
            )

            if defined VCVARS64 if exist "%VCVARS64%" (
                call "%VCVARS64%" amd64 >nul
            ) else if defined VSDEVCMD if exist "%VSDEVCMD%" (
                call "%VSDEVCMD%" -arch=x64 -host_arch=x64 -no_logo >nul
            ) else (
                echo Warning: Visual Studio Build Tools x64 environment scripts were not found automatically. gsplat build may fail.
            )
        )

        where cl >nul 2>nul
        if errorlevel 1 (
            echo Warning: cl.exe is not available in current shell. Ensure Visual Studio 2022 Build Tools x64 is installed.
        ) else (
            where cl > "%TEMP%\bimba3d_cl_where.txt" 2>nul
            findstr /I /C:"Hostx64\x64\cl.exe" "%TEMP%\bimba3d_cl_where.txt" >nul 2>nul
            if errorlevel 1 echo Warning: cl.exe does not appear to be Hostx64\x64. If gsplat build fails, run vcvarsall.bat amd64.
            del "%TEMP%\bimba3d_cl_where.txt" >nul 2>nul
        )

        "%VENV_PY%" -m pip install --force-reinstall ninja
        if errorlevel 1 (
            echo Warning: failed to install ninja. gsplat build may be unavailable.
        )
        "%VENV_PY%" -m pip install --no-binary=gsplat --no-deps "gsplat==%GSPLAT_VERSION%" --no-build-isolation -v
        if errorlevel 1 (
            echo Warning: failed to build/install gsplat %GSPLAT_VERSION%. gsplat training will fail until CUDA/toolchain is repaired.
        )
    )

    "%VENV_PY%" -m pip install --upgrade-strategy only-if-needed -r bimba3d_backend\requirements.windows.txt
    if errorlevel 1 (
        echo Failed to install backend requirements.
        pause
        exit /b 1
    )

    set "TORCH_INSTALLED=<unknown>"
    for /f "delims=" %%I in ('"%VENV_PY%" -c "import torch; print(getattr(torch,'__version__','unknown'))" 2^>nul') do set "TORCH_INSTALLED=%%I"
    > "%BOOTSTRAP_STATE%" echo TORCH_VERSION=%TORCH_VERSION%
    >> "%BOOTSTRAP_STATE%" echo TORCH_FLAVOR=%TORCH_FLAVOR%
    >> "%BOOTSTRAP_STATE%" echo TORCH_INSTALLED=%TORCH_INSTALLED%
    >> "%BOOTSTRAP_STATE%" echo GSPLAT_VERSION=%GSPLAT_VERSION%
)

"%VENV_PY%" -c "import torch; import gsplat.cuda._wrapper as w; ok=torch.cuda.is_available() and (getattr(w,'_C',None) is not None or hasattr(w,'_make_lazy_cuda_obj')); raise SystemExit(0 if ok else 1)" >nul 2>nul
if errorlevel 1 (
    echo Warning: CUDA-enabled gsplat runtime is not ready on this machine.
    echo Processing features requiring CUDA/gsplat may be unavailable until CUDA tooling is installed.
)

set "WORKER_MODE=local"
set "FRONTEND_DIST=%CD%\bimba3d_frontend\dist"
set "BIMBA3D_DATA_DIR=%ProgramData%\Bimba3D\data\projects"

if not exist "%BIMBA3D_DATA_DIR%" (
    mkdir "%BIMBA3D_DATA_DIR%"
)
"@ | Set-Content -Path $bootstrapPath -Encoding ASCII

$launcherPath = Join-Path $StagingDir "start_bimba3d.bat"
@"
@echo off
setlocal

cd /d "%~dp0"

set "VENV_PY=%ProgramData%\Bimba3D\runtime\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Bimba3D runtime is not initialized.
    echo Run the installer repair or reinstall Bimba3D.
    pause
    exit /b 1
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

$uninstallPs1Path = Join-Path $StagingDir "uninstall_bimba3d.ps1"
@"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing




$form = New-Object System.Windows.Forms.Form
$form.Text = "Uninstall Bimba3D"
$form.Size = New-Object System.Drawing.Size(520,220)
$form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
$form.MaximizeBox = $false
$form.MinimizeBox = $false

$label = New-Object System.Windows.Forms.Label
$label.Text = "Uninstall Bimba3D from this machine."
$label.AutoSize = $true
$label.Location = New-Object System.Drawing.Point(20,20)
$form.Controls.Add($label)

$checkbox = New-Object System.Windows.Forms.CheckBox
$checkbox.Text = "Also delete all project data (C:\ProgramData\Bimba3D\data\projects)"
$checkbox.AutoSize = $true
$checkbox.Checked = $false
$checkbox.Location = New-Object System.Drawing.Point(20,60)
$form.Controls.Add($checkbox)

$btnUninstall = New-Object System.Windows.Forms.Button
$btnUninstall.Text = "Uninstall"
$btnUninstall.Size = New-Object System.Drawing.Size(110,32)
$btnUninstall.Location = New-Object System.Drawing.Point(280,120)
$btnUninstall.DialogResult = [System.Windows.Forms.DialogResult]::OK
$form.Controls.Add($btnUninstall)

$btnCancel = New-Object System.Windows.Forms.Button
$btnCancel.Text = "Cancel"
$btnCancel.Size = New-Object System.Drawing.Size(110,32)
$btnCancel.Location = New-Object System.Drawing.Point(400,120)
$btnCancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
$form.Controls.Add($btnCancel)

$form.AcceptButton = $btnUninstall
$form.CancelButton = $btnCancel

$result = $form.ShowDialog()
if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
    exit 0
}

$deleteProjects = if ($checkbox.Checked) { "1" } else { "0" }

$uninstallKeys = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
)

$productCode = $null
foreach ($key in $uninstallKeys) {
    $entry = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue |
        Where-Object { $_.DisplayName -eq "Bimba3D" -and $_.PSChildName -match "^\{[0-9A-Fa-f-]+\}$" } |
        Select-Object -First 1
    if ($entry) {
        $productCode = $entry.PSChildName
        break
    }
}

if (-not $productCode) {
    [System.Windows.Forms.MessageBox]::Show(
        "Could not locate Bimba3D product code for uninstall.",
        "Uninstall Bimba3D",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
    ) | Out-Null
    exit 1
}

$args = @("/x", $productCode, "/passive", "DELETE_PROJECTS=$deleteProjects")
[System.Windows.Forms.MessageBox]::Show(
    "Uninstalling Bimba3D...`n`nThis may take several minutes, especially if runtime or project files are large.",
    "Uninstall Bimba3D",
    [System.Windows.Forms.MessageBoxButtons]::OK,
    [System.Windows.Forms.MessageBoxIcon]::Information
) | Out-Null
Start-Process -FilePath "msiexec.exe" -ArgumentList $args -Wait
exit $LASTEXITCODE
"@ | Set-Content -Path $uninstallPs1Path -Encoding UTF8

$uninstallCmdPath = Join-Path $StagingDir "uninstall_bimba3d.cmd"
@"
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall_bimba3d.ps1"
"@ | Set-Content -Path $uninstallCmdPath -Encoding ASCII

$readmePath = Join-Path $StagingDir "README.txt"
@(
    "Bimba3D packaged application"
    "Generated: $(Get-Date -Format o)"
    ""
    "Start the app using start_bimba3d.bat"
    "Runtime bootstrap executes during bundle installation in staged steps (venv/torch/gsplat/requirements)."
    "If installation fails, check Burn/MSI logs for runtime bootstrap output."
    "Use uninstall_bimba3d.cmd to uninstall with optional project-data deletion."
) | Set-Content -Path $readmePath -Encoding UTF8

Push-Location (Split-Path -Parent $WxsPath)
try {
    & wix build "$(Split-Path -Leaf $WxsPath)" -ext WixToolset.Util.wixext -o $OutputMsi
    if ($LASTEXITCODE -ne 0) {
        throw "wix build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "App MSI created: $OutputMsi" -ForegroundColor Green