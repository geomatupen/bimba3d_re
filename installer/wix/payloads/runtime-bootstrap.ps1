param(
    [Parameter(Mandatory = $true)]
    [string]$InstallDir,
    [Parameter(Mandatory = $true)]
    [ValidateSet("prepare", "torch", "gsplat", "requirements")]
    [string]$Phase
)

$ErrorActionPreference = 'Stop'

$runtimeRoot = Join-Path $env:ProgramData "Bimba3D\runtime"
$venvDir = Join-Path $runtimeRoot ".venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"
$bootstrapState = Join-Path $runtimeRoot "bootstrap-state.txt"
$torchFlavorFile = Join-Path $runtimeRoot "torch-flavor.txt"
$backendRequirements = Join-Path $InstallDir "bimba3d_backend\requirements.windows.txt"

$torchIndex = "https://download.pytorch.org/whl/cu121"
$torchVersion = "2.5.1+cu121"
$torchCpuVersion = "2.5.1"
$gsplatVersion = "1.5.3"

function Ensure-RuntimeRoot {
    if (-not (Test-Path $runtimeRoot)) {
        New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
    }
}

function Resolve-PythonExecutable {
    $candidates = @(
        "C:\Program Files\Python312\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return @{ exe = $candidate; args = @() }
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return @{ exe = "python"; args = @() }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        return @{ exe = "py"; args = @("-3") }
    }

    throw "Python is not installed. Please install Python 3.12+ and rerun setup."
}

function Ensure-Venv {
    Ensure-RuntimeRoot

    if (-not (Test-Path $venvPy)) {
        Write-Host "Creating Python virtual environment..."
        $python = Resolve-PythonExecutable
        & $python.exe @($python.args + @("-m", "venv", $venvDir))
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create Python virtual environment."
        }
    }

    if (-not (Test-Path $venvPy)) {
        throw "Venv python not found at $venvPy"
    }
}

function Install-PipTooling {
    Write-Host "Installing pip/setuptools/wheel..."
    & $venvPy -m pip install --upgrade pip setuptools==69.5.1 wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Python packaging tooling."
    }
}

function Detect-CudaToolkit {
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
        return $true
    }

    $versions = "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0"
    foreach ($version in $versions) {
        $cudaDir = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$version"
        if (Test-Path (Join-Path $cudaDir "bin\nvcc.exe")) {
            return $true
        }
    }

    return $false
}

function Install-Torch {
    Ensure-Venv

    $torchFlavor = "cu121"
    Write-Host "Installing CUDA torch..."
    & $venvPy -m pip install --index-url $torchIndex --force-reinstall "torch==$torchVersion"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "CUDA torch install failed, falling back to CPU torch."
        $torchFlavor = "cpu"
        & $venvPy -m pip install --force-reinstall "torch==$torchCpuVersion"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install torch runtime."
        }
    }
    else {
        & $venvPy -c "import torch; raise SystemExit(0 if getattr(torch.version,'cuda',None) else 1)"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Torch installed but reports CPU-only build."
            $torchFlavor = "cpu"
        }
    }

    Set-Content -Path $torchFlavorFile -Value $torchFlavor -Encoding ASCII
}

function Install-Gsplat {
    Ensure-Venv

    $torchFlavor = "cpu"
    if (Test-Path $torchFlavorFile) {
        $torchFlavor = (Get-Content $torchFlavorFile -Raw).Trim()
    }

    if ($torchFlavor -ne "cu121") {
        Write-Warning "Skipping gsplat build because CUDA torch is not active."
        return
    }

    Write-Host "Installing ninja..."
    & $venvPy -m pip install --force-reinstall ninja
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install ninja."
    }

    Write-Host "Building gsplat $gsplatVersion..."
    & $venvPy -m pip install --no-binary=gsplat --no-deps "gsplat==$gsplatVersion" --no-build-isolation -v
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build/install gsplat $gsplatVersion."
    }
}

function Install-Requirements {
    Ensure-Venv

    if (-not (Test-Path $backendRequirements)) {
        throw "Backend requirements file not found: $backendRequirements"
    }

    Write-Host "Installing backend requirements..."
    & $venvPy -m pip install --upgrade-strategy only-if-needed -r $backendRequirements
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install backend requirements."
    }

    $torchInstalled = & $venvPy -c "import torch; print(getattr(torch,'__version__','unknown'))"
    $torchFlavor = "unknown"
    if (Test-Path $torchFlavorFile) {
        $torchFlavor = (Get-Content $torchFlavorFile -Raw).Trim()
    }

    @(
        "TORCH_VERSION=$torchVersion"
        "TORCH_FLAVOR=$torchFlavor"
        "TORCH_INSTALLED=$torchInstalled"
        "GSPLAT_VERSION=$gsplatVersion"
    ) | Set-Content -Path $bootstrapState -Encoding ASCII

    $dataDir = Join-Path $env:ProgramData "Bimba3D\data\projects"
    if (-not (Test-Path $dataDir)) {
        New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
    }
}

Write-Host "Runtime bootstrap phase: $Phase"

switch ($Phase) {
    "prepare" {
        Ensure-Venv
        Install-PipTooling
    }
    "torch" {
        Install-Torch
    }
    "gsplat" {
        Install-Gsplat
    }
    "requirements" {
        Install-Requirements
    }
}

Write-Host "Runtime bootstrap phase '$Phase' completed."
