param(
    [switch]$ProbeOnly
)

$ErrorActionPreference = 'Stop'

$logRoot = 'C:\ProgramData\Bimba3D\Logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$logPath = Join-Path $logRoot 'prereq-check.log'

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
$compat = Resolve-Bimba3DCompatibility

function Write-Log([string]$message) {
    Add-Content -Path $logPath -Value "$(Get-Date -Format o) $message"
}

$vsRecommended = [string]$compat.recommendedVsInstallerUrl
$vsAllVersions = 'https://visualstudio.microsoft.com/vs/older-downloads/'
$cudaRecommended = [string]$compat.matrix.defaults.cudaRecommendedInstallerUrl
$cudaAllVersions = 'https://developer.nvidia.com/cuda-toolkit-archive'
$cudaMinimum = [string]$compat.cudaMin

function Test-SupportedCudaVersion {
    param([string]$VersionText)

    if (-not $VersionText) {
        return $false
    }

    return Test-Bimba3DVersionAtLeast -Actual $VersionText -Minimum $cudaMinimum
}

function Get-CudaVersionFromPath {
    param([string]$PathText)

    if (-not $PathText) {
        return $null
    }

    $match = [regex]::Match($PathText, 'CUDA\\v(?<v>\d+\.\d+)')
    if ($match.Success) {
        return $match.Groups['v'].Value
    }

    return $null
}

function Get-RegistryValue {
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

function Get-VsWherePath {
    $cmd = Get-Command vswhere.exe -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }

    $candidates = @()
    if ($env:ProgramFiles -and $env:ProgramFiles.Trim()) {
        $candidates += (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\Installer\vswhere.exe')
    }
        if (${env:ProgramFiles(x86)} -and ${env:ProgramFiles(x86)}.Trim()) {
            $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe')
    }

    $candidates = $candidates | Select-Object -Unique
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Test-BuildToolsInstalled {
    $buildToolsPath = Get-RegistryValue -Hive LocalMachine -SubKey 'SOFTWARE\Microsoft\VisualStudio\Setup\Instances\BuildTools_17' -ValueName 'InstallationPath'
    if ($buildToolsPath) {
        Write-Log "VS detected via BuildTools registry path: $buildToolsPath"
        return $true
    }

    $vs2022Sxs = Get-RegistryValue -Hive LocalMachine -SubKey 'SOFTWARE\Microsoft\VisualStudio\SxS\VS7' -ValueName '17.0'
    if ($vs2022Sxs) {
        Write-Log "VS detected via VS7 SxS registry path: $vs2022Sxs"
        return $true
    }

    $vswherePath = Get-VsWherePath
    if ($vswherePath) {
        try {
            $vcToolsPath = & $vswherePath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -latest 2>$null
            if ($LASTEXITCODE -eq 0 -and $vcToolsPath -and $vcToolsPath.Trim()) {
                Write-Log "VS detected via vswhere VC tools component: $($vcToolsPath.Trim())"
                return $true
            }
        } catch {
            Write-Log ("vswhere check failed: " + $_.Exception.Message)
        }
    }

    Write-Log 'VS not detected by registry or vswhere VC tools component check.'

    return $false
}

function Test-CudaInstalled {
    $cudaVersion = Get-Bimba3DInstalledCudaVersion
    if (-not $cudaVersion) {
        Write-Log 'CUDA not detected by resolver checks.'
        return $false
    }

    if (Test-SupportedCudaVersion -VersionText $cudaVersion) {
        Write-Log "CUDA detected: version $cudaVersion (minimum required: $cudaMinimum)"
        return $true
    }

    Write-Log "CUDA detected but too old: version $cudaVersion (minimum required: $cudaMinimum)"
    return $false
}

function Open-DependencyLinks {
    param(
        [bool]$vsMissing,
        [bool]$cudaMissing
    )

    if ($vsMissing) {
        Start-Process $vsRecommended
        Start-Process $vsAllVersions
    }

    if ($cudaMissing) {
        Start-Process $cudaRecommended
        Start-Process $cudaAllVersions
    }
}

function Show-PrereqDialog {
    param(
        [bool]$vsInstalled,
        [bool]$cudaInstalled,
        [Nullable[int]]$detectedVsMajor,
        [int]$requiredVsMajor
    )

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    $form = New-Object System.Windows.Forms.Form
    $form.Text = 'Bimba3D Setup - Prerequisites'
    $form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
    $form.Size = New-Object System.Drawing.Size(760, 380)
    $form.MinimumSize = $form.Size
    $form.MaximizeBox = $false
    $form.MinimizeBox = $false
    $form.TopMost = $true

    $title = New-Object System.Windows.Forms.Label
    $title.Text = 'Bimba3D requires Visual Studio Build Tools and NVIDIA CUDA Toolkit before continuing.'
    $title.AutoSize = $true
    $title.Location = New-Object System.Drawing.Point(16, 16)
    $form.Controls.Add($title)

    $status = New-Object System.Windows.Forms.Label
    $status.AutoSize = $true
    $status.Location = New-Object System.Drawing.Point(16, 46)
    $vsDetectedText = if ($detectedVsMajor) { "$detectedVsMajor" } else { 'Not detected' }
    $status.Text = "Visual Studio Build Tools (required major >= $requiredVsMajor): " + ($(if ($vsInstalled) { 'Installed' } else { 'Missing or too old' })) + "`n" +
                   "Detected VS major: $vsDetectedText`n" +
                   "NVIDIA CUDA Toolkit (minimum $cudaMinimum): " + ($(if ($cudaInstalled) { 'Installed' } else { 'Missing or too old' }))
    $form.Controls.Add($status)

    $y = 98
    if (-not $vsInstalled) {
        $vsLabel = New-Object System.Windows.Forms.Label
        $vsLabel.AutoSize = $true
        $vsLabel.Location = New-Object System.Drawing.Point(16, $y)
        $vsLabel.Text = 'Visual Studio Build Tools links:'
        $form.Controls.Add($vsLabel)

        $vsRec = New-Object System.Windows.Forms.LinkLabel
        $vsRec.AutoSize = $true
        $vsRec.Location = New-Object System.Drawing.Point(36, ($y + 22))
        $vsRec.Text = "Recommended for this CUDA profile: VS Build Tools $requiredVsMajor"
        $vsRec.Tag = $vsRecommended
        $vsRec.add_LinkClicked({ Start-Process $this.Tag })
        $form.Controls.Add($vsRec)

        $vsAny = New-Object System.Windows.Forms.LinkLabel
        $vsAny.AutoSize = $true
        $vsAny.Location = New-Object System.Drawing.Point(36, ($y + 44))
        $vsAny.Text = 'Any versions page'
        $vsAny.Tag = $vsAllVersions
        $vsAny.add_LinkClicked({ Start-Process $this.Tag })
        $form.Controls.Add($vsAny)

        $y += 78
    }

    if (-not $cudaInstalled) {
        $cudaLabel = New-Object System.Windows.Forms.Label
        $cudaLabel.AutoSize = $true
        $cudaLabel.Location = New-Object System.Drawing.Point(16, $y)
        $cudaLabel.Text = 'NVIDIA CUDA Toolkit links:'
        $form.Controls.Add($cudaLabel)

        $cudaRec = New-Object System.Windows.Forms.LinkLabel
        $cudaRec.AutoSize = $true
        $cudaRec.Location = New-Object System.Drawing.Point(36, ($y + 22))
        $cudaRec.Text = 'Recommended (validated): CUDA 12.5 network installer (.exe)'
        $cudaRec.Tag = $cudaRecommended
        $cudaRec.add_LinkClicked({ Start-Process $this.Tag })
        $form.Controls.Add($cudaRec)

        $cudaAny = New-Object System.Windows.Forms.LinkLabel
        $cudaAny.AutoSize = $true
        $cudaAny.Location = New-Object System.Drawing.Point(36, ($y + 44))
        $cudaAny.Text = 'Any versions page'
        $cudaAny.Tag = $cudaAllVersions
        $cudaAny.add_LinkClicked({ Start-Process $this.Tag })
        $form.Controls.Add($cudaAny)

        $y += 78
    }

    $hint = New-Object System.Windows.Forms.Label
    $hint.AutoSize = $true
    $hint.Location = New-Object System.Drawing.Point(16, [Math]::Min($y + 8, 275))
    $hint.Text = 'Install prerequisites, then click Retry Detection. Click Cancel to exit setup.'
    $form.Controls.Add($hint)

    $openBtn = New-Object System.Windows.Forms.Button
    $openBtn.Text = 'Open Missing Download Pages'
    $openBtn.Size = New-Object System.Drawing.Size(220, 30)
    $openBtn.Location = New-Object System.Drawing.Point(16, 305)
    $openBtn.Enabled = (-not $vsInstalled) -or (-not $cudaInstalled)
    $openBtn.Add_Click({ Open-DependencyLinks -vsMissing (-not $vsInstalled) -cudaMissing (-not $cudaInstalled) })
    $form.Controls.Add($openBtn)

    $retryBtn = New-Object System.Windows.Forms.Button
    $retryBtn.Text = 'Retry Detection'
    $retryBtn.Size = New-Object System.Drawing.Size(130, 30)
    $retryBtn.Location = New-Object System.Drawing.Point(470, 305)
    $retryBtn.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $form.Controls.Add($retryBtn)

    $cancelBtn = New-Object System.Windows.Forms.Button
    $cancelBtn.Text = 'Cancel Setup'
    $cancelBtn.Size = New-Object System.Drawing.Size(110, 30)
    $cancelBtn.Location = New-Object System.Drawing.Point(614, 305)
    $cancelBtn.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $form.Controls.Add($cancelBtn)

    $form.AcceptButton = $retryBtn
    $form.CancelButton = $cancelBtn

    $result = $form.ShowDialog()
    $form.Dispose()
    return $result
}

while ($true) {
    $compat = Resolve-Bimba3DCompatibility
    $requiredVsMajor = [int]$compat.requiredVsMajor
    $vsRecommended = [string]$compat.recommendedVsInstallerUrl

    $detectedVsMajor = Get-Bimba3DInstalledVsMajor
    $vsInstalled = $false
    if ($detectedVsMajor) {
        $vsInstalled = ([int]$detectedVsMajor -ge $requiredVsMajor)
    }

    $cudaInstalled = Test-CudaInstalled
    Write-Log "Detected VS=$vsInstalled (detectedMajor=$detectedVsMajor requiredMajor=$requiredVsMajor profile=$($compat.selectedVsProfile)) CUDA=$cudaInstalled"

    if ($ProbeOnly) {
        Write-Host "VSInstalled=$vsInstalled"
        Write-Host "VSDetectedMajor=$detectedVsMajor"
        Write-Host "VSRequiredMajor=$requiredVsMajor"
        Write-Host "CUDAInstalled=$cudaInstalled"
        if ($vsInstalled -and $cudaInstalled) {
            exit 0
        }
        exit 1
    }

    if ($vsInstalled -and $cudaInstalled) {
        Write-Log 'All prerequisites detected. Continuing setup.'
        exit 0
    }

    $result = Show-PrereqDialog -vsInstalled $vsInstalled -cudaInstalled $cudaInstalled -detectedVsMajor $detectedVsMajor -requiredVsMajor $requiredVsMajor
    if ($result -eq [System.Windows.Forms.DialogResult]::Cancel) {
        throw 'User cancelled setup during prerequisite check.'
    }
}
