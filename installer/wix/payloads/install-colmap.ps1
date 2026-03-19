param(
    [Parameter(Mandatory = $true)]
    [string]$CudaZipPath,
    [string]$NoCudaZipPath,
    [string]$InstallDir = "C:\ProgramData\Bimba3D\third_party\colmap",
    [string]$LogPath,
    [string]$BurnLogPath
)

$ErrorActionPreference = 'Stop'

function Get-SafeTempRoot {
    if ($env:TEMP -and $env:TEMP.Trim()) {
        return $env:TEMP
    }
    if ($env:TMP -and $env:TMP.Trim()) {
        return $env:TMP
    }
    if ($env:WINDIR -and $env:WINDIR.Trim()) {
        return (Join-Path $env:WINDIR 'Temp')
    }
    return 'C:\Windows\Temp'
}

$logRoot = "C:\ProgramData\Bimba3D\Logs"
try {
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
} catch {
    $logRoot = Join-Path (Get-SafeTempRoot) "Bimba3D\Logs"
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
}
if ($LogPath) {
    $logPath = $LogPath
} else {
    $logPath = Join-Path $logRoot ("bimba3d-colmap-install-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".log")
}

try {
    Add-Content -Path $logPath -Value ("{0} bootstrap" -f (Get-Date -Format o)) -ErrorAction Stop
} catch {
    $logRoot = Join-Path (Get-SafeTempRoot) "Bimba3D\Logs"
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    if ($LogPath) {
        $logPath = Join-Path $logRoot ([System.IO.Path]::GetFileName($LogPath))
    } else {
        $logPath = Join-Path $logRoot ("bimba3d-colmap-install-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".log")
    }
}

$compat = $null

function Resolve-CompatibilityContext {
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
    return Resolve-Bimba3DCompatibility
}

function Write-Log([string]$message) {
    $line = "$(Get-Date -Format o) $message"
    Write-Host $line
    try {
        Add-Content -Path $logPath -Value $line -ErrorAction Stop
    } catch {
        Write-Host "[COLMAP-LOG-FALLBACK] $line"
    }

    if ($BurnLogPath) {
        try {
            Add-Content -Path $BurnLogPath -Value ("[COLMAP] " + $line) -ErrorAction Stop
        } catch {
        }
    }
}

function Download-ColmapArchive {
    param(
        [string]$Url,
        [string]$AssetName
    )

    $downloadDir = Join-Path $env:ProgramData "Bimba3D\cache"
    New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null

    $destination = Join-Path $downloadDir $AssetName
    Write-Log "Downloading COLMAP asset: $Url"
    Invoke-WebRequest -Uri $Url -OutFile $destination -UseBasicParsing
    if (-not (Test-Path $destination)) {
        throw "Failed to download COLMAP archive: $Url"
    }

    return $destination
}

function Get-ArchivePathForVariant {
    param(
        [string]$Variant,
        [ValidateSet('offline', 'online')]
        [string]$Source
    )

    if ($Variant -eq 'cuda') {
        if ($Source -eq 'offline') {
            if ($CudaZipPath -and (Test-Path -LiteralPath $CudaZipPath)) {
                return [System.IO.Path]::GetFullPath($CudaZipPath)
            }

            throw "Offline CUDA COLMAP payload is missing."
        }

        return Download-ColmapArchive -Url $compat.colmapCudaUrl -AssetName ([System.IO.Path]::GetFileName($compat.colmapCudaUrl))
    }

    if ($Source -eq 'offline') {
        if ($NoCudaZipPath -and (Test-Path -LiteralPath $NoCudaZipPath)) {
            return [System.IO.Path]::GetFullPath($NoCudaZipPath)
        }

        throw "Offline no-CUDA COLMAP payload is missing."
    }

    return Download-ColmapArchive -Url $compat.colmapNoCudaUrl -AssetName ([System.IO.Path]::GetFileName($compat.colmapNoCudaUrl))
}

function Expand-ColmapArchive {
    param([string]$ArchivePath)

    if (-not (Test-Path -LiteralPath $ArchivePath)) {
        throw "COLMAP zip not found: $ArchivePath"
    }

    if (Test-Path -LiteralPath $InstallDir) {
        Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $InstallDir -Force

    $colmapBat = Join-Path $InstallDir "COLMAP.bat"
    if (-not (Test-Path -LiteralPath $colmapBat)) {
        $nested = Get-ChildItem -LiteralPath $InstallDir -Filter "COLMAP.bat" -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($nested) {
            Write-Log "Found nested COLMAP.bat at $($nested.FullName)"
            $nestedBat = $nested.FullName
            $launcher = @"
@echo off
call "$nestedBat" %*
"@
            Set-Content -Path $colmapBat -Value $launcher -Encoding ascii -Force
            Write-Log "Created launcher at $colmapBat -> $nestedBat"
        }
    }

    if (-not (Test-Path -LiteralPath $colmapBat)) {
        throw "COLMAP.bat not found after extraction."
    }

    return $colmapBat
}

function Test-ColmapExecutable {
    param([string]$ColmapBatPath)

    & $ColmapBatPath -h | Out-Null
    return ($LASTEXITCODE -eq 0)
}

try {
    Write-Host "[COLMAP] Log file: $logPath"
    $compat = Resolve-CompatibilityContext
    Write-Log "Starting COLMAP install"
    Write-Log "CudaZipPath=$CudaZipPath"
    Write-Log "NoCudaZipPath=$NoCudaZipPath"
    Write-Log "InstallDir=$InstallDir"
    Write-Log "Resolved compatibility: CUDA=$($compat.detectedCudaVersion) VS=$($compat.detectedVsMajor) Variant=$($compat.colmapPreferredVariant) Profile=$($compat.colmapAssetProfile) CudaColmap=$($compat.colmapCudaVersion) NoCudaColmap=$($compat.colmapNoCudaVersion) DefaultStack=$($compat.useDefaultStack)"

    $installPlan = New-Object System.Collections.Generic.List[object]
    if ($compat.colmapPreferredVariant -eq 'cuda' -and $compat.useDefaultStack) {
        $installPlan.Add([pscustomobject]@{ Variant = 'cuda'; Source = 'offline'; Reason = 'default stack match' })
        $installPlan.Add([pscustomobject]@{ Variant = 'cuda'; Source = 'online'; Reason = 'refresh from online if needed' })
        $installPlan.Add([pscustomobject]@{ Variant = 'nocuda'; Source = 'online'; Reason = 'fallback when CUDA COLMAP is incompatible' })
        $installPlan.Add([pscustomobject]@{ Variant = 'nocuda'; Source = 'offline'; Reason = 'offline fallback when internet is unavailable' })
    } else {
        $installPlan.Add([pscustomobject]@{ Variant = 'cuda'; Source = 'online'; Reason = 'non-default CUDA stack: prefer online candidate first' })
        $installPlan.Add([pscustomobject]@{ Variant = 'nocuda'; Source = 'online'; Reason = 'fallback when CUDA candidate fails' })
        $installPlan.Add([pscustomobject]@{ Variant = 'nocuda'; Source = 'offline'; Reason = 'offline fallback when internet is unavailable' })
    }

    $selectedVariant = $null
    $selectedSource = $null
    $colmapBat = $null
    foreach ($plan in $installPlan) {
        $variant = [string]$plan.Variant
        $source = [string]$plan.Source
        $reason = [string]$plan.Reason

        Write-Host "[COLMAP] Trying variant=$variant source=$source ($reason)..."
        Write-Log "Trying variant='$variant' source='$source' reason='$reason'"

        try {
            $archivePath = Get-ArchivePathForVariant -Variant $variant -Source $source
            Write-Log "Installing COLMAP variant '$variant' from $archivePath"
            $colmapBat = Expand-ColmapArchive -ArchivePath $archivePath
            if (Test-ColmapExecutable -ColmapBatPath $colmapBat) {
                Write-Host "[COLMAP] Installed variant=$variant source=$source successfully."
                Write-Log "COLMAP smoke test succeeded for variant '$variant' source '$source'"
                $selectedVariant = $variant
                $selectedSource = $source
                break
            }

            Write-Host "[COLMAP] Smoke test failed for variant=$variant source=$source."
            Write-Log "COLMAP smoke test failed for variant '$variant' source '$source'"
        } catch {
            Write-Host "[COLMAP] Attempt failed for variant=$variant source=$source."
            Write-Log "COLMAP variant '$variant' source '$source' failed: $($_.Exception.Message)"
        }
    }

    if (-not $selectedVariant) {
        throw 'Failed to install a compatible COLMAP variant.'
    }

    try {
        [Environment]::SetEnvironmentVariable("COLMAP_EXE", $colmapBat, "Machine")
        Write-Log "Set machine environment variable COLMAP_EXE=$colmapBat"
    } catch {
        Write-Log ("WARNING: failed to set machine COLMAP_EXE: " + $_.Exception.Message)
        try {
            [Environment]::SetEnvironmentVariable("COLMAP_EXE", $colmapBat, "User")
            Write-Log "Set user environment variable COLMAP_EXE=$colmapBat"
        } catch {
            Write-Log ("WARNING: failed to set user COLMAP_EXE: " + $_.Exception.Message)
            Write-Log "Continuing without persisted COLMAP_EXE environment variable."
        }
    }
    Write-Log "COLMAP installed at $InstallDir (variant=$selectedVariant source=$selectedSource)"
    Write-Log "Completed successfully"
    exit 0
} catch {
    Write-Log ("ERROR: " + $_.Exception.Message)
    if ($_.Exception.StackTrace) {
        Write-Log ("STACK: " + $_.Exception.StackTrace)
    }
    if ($_.Exception.InnerException) {
        Write-Log ("INNER: " + $_.Exception.InnerException.Message)
    }
    Write-Host "COLMAP install failed. See log: $logPath" -ForegroundColor Red
    exit 1
}
