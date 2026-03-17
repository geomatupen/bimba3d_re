param(
    [string]$InstallerPath = "..\Bimba3D-Setup.exe"
)

$resolvedInstallerPath = Resolve-Path -Path $InstallerPath -ErrorAction SilentlyContinue
if (-not $resolvedInstallerPath) {
    Write-Error "Installer not found at '$InstallerPath'. Build the bundle first or pass -InstallerPath."
    exit 1
}

$resolvedInstallerPath = $resolvedInstallerPath.Path

$projectEulaUrl = "https://github.com/geomatupen/bimba3d/blob/main/installer/wix/EULA.md"
$thirdPartyNoticesUrl = "https://github.com/geomatupen/bimba3d/blob/main/installer/wix/THIRD_PARTY_NOTICES.md"
$vsTermsUrl = "https://visualstudio.microsoft.com/license-terms/vs2022-ga-diagnosticbuildtools/"
$cudaTermsUrl = "https://docs.nvidia.com/cuda/eula/index.html"

Write-Host "Bimba3D compliant installer launcher" -ForegroundColor Cyan
Write-Host ""
Write-Host "VC++ Runtime will be installed automatically if missing." -ForegroundColor Yellow
Write-Host "Build Tools and CUDA are opt-in and require acceptance." -ForegroundColor Yellow
Write-Host ""
Write-Host "Bimba3D EULA:      $projectEulaUrl"
Write-Host "Third-party terms: $thirdPartyNoticesUrl"
Write-Host "Build Tools terms: $vsTermsUrl"
Write-Host "CUDA terms:       $cudaTermsUrl"
Write-Host ""

$installBuildTools = Read-Host "Install Microsoft Visual Studio Build Tools 2022? (y/N)"
$installCuda = Read-Host "Install NVIDIA CUDA Toolkit via bundled network installer? (y/N)"

$buildToolsValue = if ($installBuildTools -match '^(y|yes)$') { "1" } else { "0" }
$cudaValue = if ($installCuda -match '^(y|yes)$') { "1" } else { "0" }

$arguments = @(
    "/InstallBuildTools=$buildToolsValue",
    "/InstallCudaToolkit=$cudaValue"
)

Write-Host ""
Write-Host "Launching installer with: InstallBuildTools=$buildToolsValue, InstallCudaToolkit=$cudaValue" -ForegroundColor Green
Start-Process -FilePath $resolvedInstallerPath -ArgumentList $arguments -Wait
