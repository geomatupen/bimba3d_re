$ErrorActionPreference = 'Stop'

function ConvertTo-Bimba3DVersion {
    param([string]$VersionText)
    if (-not $VersionText) {
        return $null
    }

    try {
        return [Version]$VersionText
    } catch {
        return $null
    }
}

function Test-Bimba3DVersionAtLeast {
    param(
        [string]$Actual,
        [string]$Minimum
    )

    $actualVersion = ConvertTo-Bimba3DVersion -VersionText $Actual
    $minimumVersion = ConvertTo-Bimba3DVersion -VersionText $Minimum
    if (-not $actualVersion -or -not $minimumVersion) {
        return $false
    }

    return $actualVersion -ge $minimumVersion
}

function Test-Bimba3DVersionInRange {
    param(
        [string]$Actual,
        [string]$Minimum,
        [string]$Maximum
    )

    if (-not (Test-Bimba3DVersionAtLeast -Actual $Actual -Minimum $Minimum)) {
        return $false
    }

    if (-not $Maximum) {
        return $true
    }

    $actualVersion = ConvertTo-Bimba3DVersion -VersionText $Actual
    $maximumVersion = ConvertTo-Bimba3DVersion -VersionText $Maximum
    if (-not $actualVersion -or -not $maximumVersion) {
        return $false
    }

    return $actualVersion -le $maximumVersion
}

function Get-Bimba3DRegistryValue {
    param(
        [Microsoft.Win32.RegistryHive]$Hive,
        [string]$SubKey,
        [string]$ValueName
    )

    foreach ($view in @([Microsoft.Win32.RegistryView]::Registry64, [Microsoft.Win32.RegistryView]::Registry32)) {
        try {
            $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey($Hive, $view)
            $key = $base.OpenSubKey($SubKey)
            if ($key) {
                $value = $key.GetValue($ValueName, $null)
                if ($null -ne $value -and [string]$value -ne '') {
                    return [string]$value
                }
            }
        } catch {
        }
    }

    return $null
}

function Get-Bimba3DVsWherePath {
    $cmd = Get-Command vswhere.exe -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }

    $candidates = @()
    if ($env:ProgramFiles) {
        $candidates += (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\Installer\vswhere.exe')
    }
    if (${env:ProgramFiles(x86)}) {
        $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe')
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Get-Bimba3DInstalledVsMajor {
    function Convert-VsCatalogValueToMajor {
        param([int]$Value)

        $yearToMajor = @{
            2017 = 15
            2019 = 16
            2022 = 17
        }

        if ($yearToMajor.ContainsKey($Value)) {
            return [int]$yearToMajor[$Value]
        }

        if ($Value -gt 0 -and $Value -lt 100) {
            return $Value
        }

        return $null
    }

    $vswherePath = Get-Bimba3DVsWherePath
    if ($vswherePath) {
        try {
            $versionText = & $vswherePath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property catalog_productLineVersion -latest 2>$null
            if ($LASTEXITCODE -eq 0 -and $versionText -and $versionText.Trim()) {
                $major = Convert-VsCatalogValueToMajor -Value ([int]$versionText.Trim())
                if ($major) {
                    return $major
                }
            }
        } catch {
        }

        try {
            $installationVersion = & $vswherePath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion -latest 2>$null
            if ($LASTEXITCODE -eq 0 -and $installationVersion -and $installationVersion.Trim()) {
                $parsed = ConvertTo-Bimba3DVersion -VersionText $installationVersion.Trim()
                if ($parsed) {
                    return $parsed.Major
                }
            }
        } catch {
        }
    }

    $vs17Path = Get-Bimba3DRegistryValue -Hive LocalMachine -SubKey 'SOFTWARE\Microsoft\VisualStudio\SxS\VS7' -ValueName '17.0'
    if ($vs17Path) {
        return 17
    }

    return $null
}

function Get-Bimba3DInstalledCudaVersion {
    $candidates = New-Object System.Collections.Generic.List[string]

    if ($env:CUDA_PATH) {
        $match = [regex]::Match($env:CUDA_PATH, 'CUDA\\v(?<v>\d+\.\d+)')
        if ($match.Success) {
            $candidates.Add($match.Groups['v'].Value)
        }
    }

    foreach ($cudaEnv in (Get-ChildItem Env:CUDA_PATH_V* -ErrorAction SilentlyContinue)) {
        if ($cudaEnv.Value) {
            $match = [regex]::Match($cudaEnv.Value, 'CUDA\\v(?<v>\d+\.\d+)')
            if ($match.Success) {
                $candidates.Add($match.Groups['v'].Value)
            }
        }
    }

    $nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($nvcc -and $nvcc.Source) {
        try {
            $nvccOut = & $nvcc.Source --version 2>$null | Out-String
            $match = [regex]::Match($nvccOut, 'release\s+(?<v>\d+\.\d+)')
            if ($match.Success) {
                $candidates.Add($match.Groups['v'].Value)
            }
        } catch {
        }

        $pathMatch = [regex]::Match($nvcc.Source, 'CUDA\\v(?<v>\d+\.\d+)')
        if ($pathMatch.Success) {
            $candidates.Add($pathMatch.Groups['v'].Value)
        }
    }

    $cudaRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $cudaRoot) {
        foreach ($dir in (Get-ChildItem -Path $cudaRoot -Directory -ErrorAction SilentlyContinue)) {
            $match = [regex]::Match($dir.Name, '^v(?<v>\d+\.\d+)$')
            if ($match.Success) {
                $candidates.Add($match.Groups['v'].Value)
            }
        }
    }

    $registryCudaRoot = 'SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA'
    foreach ($view in @([Microsoft.Win32.RegistryView]::Registry64, [Microsoft.Win32.RegistryView]::Registry32)) {
        try {
            $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, $view)
            $root = $base.OpenSubKey($registryCudaRoot)
            if ($root) {
                foreach ($subName in $root.GetSubKeyNames()) {
                    $match = [regex]::Match($subName, '^v(?<v>\d+\.\d+)$')
                    if ($match.Success) {
                        $candidates.Add($match.Groups['v'].Value)
                    }
                }
            }
        } catch {
        }
    }

    $bestVersion = $null
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        $parsed = ConvertTo-Bimba3DVersion -VersionText $candidate
        if ($parsed -and ((-not $bestVersion) -or $parsed -gt $bestVersion)) {
            $bestVersion = $parsed
        }
    }

    if ($bestVersion) {
        return "$($bestVersion.Major).$($bestVersion.Minor)"
    }

    return $null
}

function Get-Bimba3DCompatibilityMatrix {
    param([string]$MatrixPath)

    if ([string]::IsNullOrWhiteSpace([string]$MatrixPath)) {
        $MatrixPath = $null
    }

    if (-not $MatrixPath) {
        $variantCandidates = @()
        try {
            $variantCandidates = Get-ChildItem -Path $PSScriptRoot -Filter 'compatibility-matrix*.json' -File -ErrorAction SilentlyContinue |
                Sort-Object -Property Name |
                ForEach-Object { $_.FullName }
        } catch {
        }

        $candidates = @(
            (Join-Path $PSScriptRoot 'compatibility-matrix.json')
        ) + @($variantCandidates) + @(
            (Join-Path $PSScriptRoot '..\..\..\compatibility-matrix.json')
        )

        $candidates = $candidates | Select-Object -Unique

        foreach ($candidate in $candidates) {
            if (Test-Path $candidate) {
                $MatrixPath = $candidate
                break
            }
        }
    }

    if ([string]::IsNullOrWhiteSpace([string]$MatrixPath) -or -not (Test-Path -LiteralPath $MatrixPath)) {
        throw "Compatibility matrix not found: $MatrixPath"
    }

    return Get-Content -Path $MatrixPath -Raw | ConvertFrom-Json
}

function Resolve-Bimba3DCompatibility {
    param(
        [string]$MatrixPath,
        [string]$CudaVersion,
        [Nullable[int]]$VsMajor
    )

    $matrix = Get-Bimba3DCompatibilityMatrix -MatrixPath $MatrixPath

    if (-not $CudaVersion) {
        $CudaVersion = Get-Bimba3DInstalledCudaVersion
    }
    if (-not $VsMajor) {
        $VsMajor = Get-Bimba3DInstalledVsMajor
    }

    $defaults = $matrix.defaults
    $cudaMeetsMinimum = $false
    if ($CudaVersion) {
        $cudaMeetsMinimum = Test-Bimba3DVersionAtLeast -Actual $CudaVersion -Minimum $defaults.cudaMin
    }

    $cudaNeedsUpgrade = -not $cudaMeetsMinimum
    $vsMeetsDefault = $false
    if ($VsMajor) {
        $vsMeetsDefault = ([int]$VsMajor -ge [int]$defaults.vsMajor)
    }

    $resolved = [ordered]@{
        matrix = $matrix
        detectedCudaVersion = $CudaVersion
        detectedVsMajor = $VsMajor
        cudaMin = [string]$defaults.cudaMin
        cudaPreferred = [string]$defaults.cudaPreferred
        defaultVsMajor = [int]$defaults.vsMajor
        requiredVsMajor = [int]$defaults.vsMajor
        recommendedVsInstallerUrl = [string]$defaults.vsRecommendedInstallerUrl
        selectedVsProfile = 'default'
        cudaIsInstalled = [bool]$CudaVersion
        cudaMeetsMinimum = [bool]$cudaMeetsMinimum
        cudaNeedsUpgrade = [bool]$cudaNeedsUpgrade
        vsMeetsDefault = [bool]$vsMeetsDefault
        useDefaultStack = $false
        torchTrack = [string]$defaults.torchTrack
        torchIndexUrl = [string]$defaults.torchIndexUrl
        torchVersion = [string]$defaults.torchVersion
        torchvisionVersion = [string]$defaults.torchvisionVersion
        torchaudioVersion = [string]$defaults.torchaudioVersion
        torchCpuVersion = [string]$defaults.torchCpuVersion
        torchvisionCpuVersion = [string]$defaults.torchvisionCpuVersion
        torchaudioCpuVersion = [string]$defaults.torchaudioCpuVersion
        gsplatVersion = [string]$matrix.gsplat.version
        gsplatSupportedTorchTracks = @($matrix.gsplat.supportedTorchTracks)
        colmapCudaVersion = [string]$matrix.colmap.cuda.version
        colmapCudaUrl = [string]$matrix.colmap.cuda.url
        colmapNoCudaVersion = [string]$matrix.colmap.nocuda.version
        colmapNoCudaUrl = [string]$matrix.colmap.nocuda.url
        colmapPreferredVariant = 'nocuda'
    }

    if ($resolved.cudaMeetsMinimum) {
        $resolved.colmapPreferredVariant = 'cuda'
    }

    if ($resolved.cudaMeetsMinimum -and $matrix.vsProfiles) {
        foreach ($vsProfile in $matrix.vsProfiles) {
            if (Test-Bimba3DVersionInRange -Actual $CudaVersion -Minimum $vsProfile.minCuda -Maximum $vsProfile.maxCuda) {
                if ($vsProfile.vsMajor) {
                    $resolved.requiredVsMajor = [int]$vsProfile.vsMajor
                }
                if ($vsProfile.vsRecommendedInstallerUrl) {
                    $resolved.recommendedVsInstallerUrl = [string]$vsProfile.vsRecommendedInstallerUrl
                }
                if ($vsProfile.name) {
                    $resolved.selectedVsProfile = [string]$vsProfile.name
                }
                break
            }
        }
    }

    if ($VsMajor) {
        $resolved.vsMeetsDefault = ([int]$VsMajor -ge [int]$resolved.requiredVsMajor)
    } else {
        $resolved.vsMeetsDefault = $false
    }

    if ($resolved.cudaMeetsMinimum -and $VsMajor -eq [int]$defaults.vsMajor -and $CudaVersion -eq [string]$defaults.cudaPreferred) {
        $resolved.useDefaultStack = $true
    }

    if ($resolved.cudaMeetsMinimum) {
        foreach ($profile in $matrix.torchProfiles) {
            if (Test-Bimba3DVersionInRange -Actual $CudaVersion -Minimum $profile.minCuda -Maximum $profile.maxCuda) {
                $resolved.torchTrack = [string]$profile.track
                $resolved.torchIndexUrl = [string]$profile.indexUrl
                $resolved.torchVersion = [string]$profile.torchVersion
                $resolved.torchvisionVersion = [string]$profile.torchvisionVersion
                $resolved.torchaudioVersion = [string]$profile.torchaudioVersion
                break
            }
        }
    }

    [pscustomobject]$resolved
}
