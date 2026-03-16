param(
    [Parameter(Mandatory = $true)]
    [string]$ZipPath,
    [string]$InstallDir = "C:\ProgramData\Bimba3D\third_party\colmap"
)

$ErrorActionPreference = 'Stop'

$logRoot = if ($env:TEMP -and (Test-Path -LiteralPath $env:TEMP)) { $env:TEMP } else { "C:\ProgramData\Bimba3D\Logs" }
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$logPath = Join-Path $logRoot ("bimba3d-colmap-install-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".log")

function Write-Log([string]$message) {
    $line = "$(Get-Date -Format o) $message"
    Write-Host $line
    Add-Content -Path $logPath -Value $line
}

try {
    $ZipPath = [System.IO.Path]::GetFullPath($ZipPath)
    Write-Log "Starting COLMAP install"
    Write-Log "ZipPath=$ZipPath"
    Write-Log "InstallDir=$InstallDir"

    if (-not (Test-Path -LiteralPath $ZipPath)) {
        throw "COLMAP zip not found: $ZipPath"
    }

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $InstallDir -Force

    $colmapBat = Join-Path $InstallDir "COLMAP.bat"
    if (-not (Test-Path -LiteralPath $colmapBat)) {
        $nested = Get-ChildItem -LiteralPath $InstallDir -Filter "COLMAP.bat" -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($nested) {
            $nestedRoot = Split-Path -Parent $nested.FullName
            Copy-Item -Path (Join-Path $nestedRoot "*") -Destination $InstallDir -Recurse -Force
        }
    }

    if (-not (Test-Path -LiteralPath $colmapBat)) {
        throw "COLMAP.bat not found after extraction."
    }

    try {
        [Environment]::SetEnvironmentVariable("COLMAP_EXE", $colmapBat, "Machine")
        Write-Log "Set machine environment variable COLMAP_EXE=$colmapBat"
    } catch {
        Write-Log ("WARNING: failed to set machine COLMAP_EXE: " + $_.Exception.Message)
        [Environment]::SetEnvironmentVariable("COLMAP_EXE", $colmapBat, "User")
        Write-Log "Set user environment variable COLMAP_EXE=$colmapBat"
    }
    Write-Log "COLMAP installed at $InstallDir"
    Write-Log "Completed successfully"
    exit 0
} catch {
    Write-Log ("ERROR: " + $_.Exception.Message)
    Write-Host "COLMAP install failed. See log: $logPath" -ForegroundColor Red
    exit 1
}
