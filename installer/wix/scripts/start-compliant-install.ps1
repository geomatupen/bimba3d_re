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
Write-Host "For gsplat training, BOTH Build Tools and CUDA are required." -ForegroundColor Yellow
Write-Host "Recommendation: answer 'y' to both prompts on fresh machines." -ForegroundColor Yellow
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

if ($buildToolsValue -eq "0" -or $cudaValue -eq "0") {
    Write-Host "" 
    Write-Host "Warning: You skipped one or more components required for gsplat training." -ForegroundColor Red
    Write-Host "Bimba3D will install, but training features may fail until these are installed." -ForegroundColor Red
    $continueWithoutTrainingDeps = Read-Host "Continue anyway? (y/N)"
    if ($continueWithoutTrainingDeps -notmatch '^(y|yes)$') {
        Write-Host "Installation canceled by user. Re-run and accept Build Tools + CUDA for training-ready setup." -ForegroundColor Yellow
        exit 0
    }
}

$arguments = @(
    "/InstallBuildTools=$buildToolsValue",
    "/InstallCudaToolkit=$cudaValue"
)

Write-Host ""
Write-Host "Launching installer with: InstallBuildTools=$buildToolsValue, InstallCudaToolkit=$cudaValue" -ForegroundColor Green
Start-Process -FilePath $resolvedInstallerPath -ArgumentList $arguments -Wait
