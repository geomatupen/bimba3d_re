param(
    [string]$WxsPath = "$PSScriptRoot\..\Bimba3D.Bundle.wxs",
    [string]$OutputExe = "$PSScriptRoot\..\Bimba3D-Setup.exe",
    [switch]$CiStrict
)

$ErrorActionPreference = 'Stop'

$WxsPath = [System.IO.Path]::GetFullPath($WxsPath)
$OutputExe = [System.IO.Path]::GetFullPath($OutputExe)

$wix = Get-Command wix -ErrorAction SilentlyContinue
if (-not $wix) {
    throw "wix.exe not found in PATH. Install WiX CLI first."
}

$validator = Join-Path $PSScriptRoot "validate-payload-manifest.ps1"
if (-not (Test-Path $validator)) {
    throw "Missing validator script: $validator"
}

if ($CiStrict) {
    & powershell -ExecutionPolicy Bypass -File $validator -RequireLocalFiles
} else {
    & powershell -ExecutionPolicy Bypass -File $validator
}

if ($LASTEXITCODE -ne 0) {
    throw "Payload manifest validation failed."
}

$required = @(
    "$PSScriptRoot\..\payloads\vc_redist.x64.exe",
    "$PSScriptRoot\..\payloads\colmap-x64-windows-cuda.zip",
    "$PSScriptRoot\..\payloads\install-colmap.cmd",
    "$PSScriptRoot\..\payloads\run-runtime-bootstrap.cmd",
    "$PSScriptRoot\..\payloads\runtime-bootstrap.ps1",
    "$PSScriptRoot\..\payloads\Bimba3D.msi"
)

$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing.Count -gt 0) {
    Write-Host "Missing payload files:" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host " - $_" }
    throw "Missing required payloads. Run download-payloads.ps1 and place Bimba3D.msi."
}

Push-Location (Split-Path -Parent $WxsPath)
try {
    & wix build "$(Split-Path -Leaf $WxsPath)" `
        -ext WixToolset.BootstrapperApplications.wixext `
        -ext WixToolset.Util.wixext `
        -o $OutputExe

    if ($LASTEXITCODE -ne 0) {
        throw "wix build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Bundle created: $OutputExe"
