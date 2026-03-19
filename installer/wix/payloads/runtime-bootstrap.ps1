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

$resolverPath = Join-Path $PSScriptRoot 'compatibility-resolver.ps1'
if (-not (Test-Path $resolverPath)) {
    $resolverCandidate = Get-ChildItem -Path $PSScriptRoot -Filter 'compatibility-resolver*.ps1' -File -ErrorAction SilentlyContinue |
        Sort-Object -Property Name |
        Select-Object -First 1
    if ($resolverCandidate) {
        $resolverPath = $resolverCandidate.FullName
    }
}
if (-not (Test-Path $resolverPath)) {
    throw "Compatibility resolver not found in $PSScriptRoot"
}

. $resolverPath
$matrixPath = Join-Path $PSScriptRoot 'compatibility-matrix.json'
$compat = Resolve-Bimba3DCompatibility -MatrixPath $matrixPath

$torchTrack = [string]$compat.torchTrack
$torchIndex = [string]$compat.torchIndexUrl
$torchVersion = [string]$compat.torchVersion
$torchvisionVersion = [string]$compat.torchvisionVersion
$torchaudioVersion = [string]$compat.torchaudioVersion
$torchCpuVersion = [string]$compat.torchCpuVersion
$torchvisionCpuVersion = [string]$compat.torchvisionCpuVersion
$torchaudioCpuVersion = [string]$compat.torchaudioCpuVersion
$gsplatVersion = [string]$compat.gsplatVersion
$gsplatSupportedTracks = @($compat.gsplatSupportedTorchTracks)

function Ensure-RuntimeRoot {
    if (-not (Test-Path $runtimeRoot)) {
        New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
    }

    Ensure-RuntimeWritable -TargetPath $runtimeRoot
}

function Ensure-RuntimeWritable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    if (-not (Test-Path $TargetPath)) {
        return
    }

    try {
        & icacls $TargetPath /grant "*S-1-5-32-545:(OI)(CI)M" /T /C | Out-Null
    }
    catch {
        Write-Warning "Unable to update ACLs for runtime path '$TargetPath': $($_.Exception.Message)"
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

    if (Test-Path $venvDir) {
        Ensure-RuntimeWritable -TargetPath $venvDir
    }

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

    Ensure-RuntimeWritable -TargetPath $venvDir
}

function Install-PipTooling {
    Write-Host "Installing pip/setuptools/wheel..."
    & $venvPy -m pip install --upgrade pip setuptools==69.5.1 wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Python packaging tooling."
    }
}

function Install-CpuTorch {
    Write-Host "Installing CPU torch fallback..."
    & $venvPy -m pip install --force-reinstall "torch==$torchCpuVersion" "torchvision==$torchvisionCpuVersion" "torchaudio==$torchaudioCpuVersion"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install CPU torch fallback runtime."
    }
}

function Install-Torch {
    Ensure-Venv

    $torchFlavor = $torchTrack
    Write-Host "Detected compatibility profile: CUDA=$($compat.detectedCudaVersion) VS=$($compat.detectedVsMajor) Track=$torchTrack DefaultStack=$($compat.useDefaultStack)"
    Write-Host "Installing CUDA torch stack..."
    & $venvPy -m pip install --index-url $torchIndex --force-reinstall "torch==$torchVersion" "torchvision==$torchvisionVersion" "torchaudio==$torchaudioVersion"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "CUDA torch install failed, falling back to CPU torch."
        $torchFlavor = "cpu"
        Install-CpuTorch
    }
    else {
        $torchProbeOutput = (& $venvPy -c "import torch; import traceback; import sys; print(getattr(torch.version,'cuda',None) or 'none'); raise SystemExit(0 if getattr(torch.version,'cuda',None) else 1)" 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "CUDA torch installed but import/CUDA probe failed; falling back to CPU torch."
            if ($torchProbeOutput) {
                Write-Warning "Torch CUDA probe output: $torchProbeOutput"
            }
            $torchFlavor = "cpu"
            Install-CpuTorch
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

    if ($torchFlavor -eq "cpu") {
        Write-Warning "Skipping gsplat build because CUDA torch is not active."
        return
    }

    if ($gsplatSupportedTracks.Count -gt 0 -and ($gsplatSupportedTracks -notcontains $torchFlavor)) {
        Write-Warning "Skipping gsplat build because torch track '$torchFlavor' is outside supported tracks: $($gsplatSupportedTracks -join ', ')."
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
        "TORCHVISION_VERSION=$torchvisionVersion"
        "TORCHAUDIO_VERSION=$torchaudioVersion"
        "TORCH_FLAVOR=$torchFlavor"
        "TORCH_INSTALLED=$torchInstalled"
        "GSPLAT_VERSION=$gsplatVersion"
        "CUDA_DETECTED=$($compat.detectedCudaVersion)"
        "VS_DETECTED=$($compat.detectedVsMajor)"
        "DEFAULT_STACK=$($compat.useDefaultStack)"
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
