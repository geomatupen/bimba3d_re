param(
  [string]$OutputRoot = "C:\\temp\\bimba3d-shutdown-evidence",
  [int]$HoursBack = 12
)

$ErrorActionPreference = "SilentlyContinue"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $OutputRoot $timestamp
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Write-Section {
  param([string]$Title, [string]$File)
  "`n===== $Title =====`n" | Out-File -FilePath $File -Append -Encoding utf8
}

$summaryFile = Join-Path $outDir "summary.txt"
"Bimba3D shutdown evidence" | Out-File -FilePath $summaryFile -Encoding utf8
"Collected at: $(Get-Date -Format o)" | Out-File -FilePath $summaryFile -Append -Encoding utf8
"Computer: $env:COMPUTERNAME" | Out-File -FilePath $summaryFile -Append -Encoding utf8
"User: $env:USERNAME" | Out-File -FilePath $summaryFile -Append -Encoding utf8

Write-Section "Power configuration" $summaryFile
powercfg /a | Out-File -FilePath $summaryFile -Append -Encoding utf8
Write-Section "Power requests" $summaryFile
powercfg /requests | Out-File -FilePath $summaryFile -Append -Encoding utf8
Write-Section "Last wake" $summaryFile
powercfg /lastwake | Out-File -FilePath $summaryFile -Append -Encoding utf8
Write-Section "Wake armed devices" $summaryFile
powercfg /devicequery wake_armed | Out-File -FilePath $summaryFile -Append -Encoding utf8

$start = (Get-Date).AddHours(-[math]::Abs($HoursBack))
$systemFile = Join-Path $outDir "system_power_events.txt"
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$start} |
  Where-Object { $_.Id -in 41,42,1074,6008,6006,6005,109,6009,1001 } |
  Select-Object TimeCreated, Id, ProviderName, LevelDisplayName, Message |
  Format-List | Out-File -FilePath $systemFile -Encoding utf8

$werFile = Join-Path $outDir "application_wer_events.txt"
Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=$start} |
  Where-Object { $_.ProviderName -eq 'Windows Error Reporting' } |
  Select-Object -First 120 TimeCreated, Id, ProviderName, LevelDisplayName, Message |
  Format-List | Out-File -FilePath $werFile -Encoding utf8

$reliabilityFile = Join-Path $outDir "reliability_monitor_events.txt"
Get-CimInstance -Namespace root\cimv2 -ClassName Win32_ReliabilityRecords |
  Where-Object { $_.TimeGenerated -ge $start } |
  Select-Object TimeGenerated, SourceName, EventIdentifier, Message |
  Sort-Object TimeGenerated -Descending |
  Format-List | Out-File -FilePath $reliabilityFile -Encoding utf8

$kernelDumpInfoFile = Join-Path $outDir "kernel_dump_info.txt"
"Crash control registry:" | Out-File -FilePath $kernelDumpInfoFile -Encoding utf8
Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl" |
  Format-List CrashDumpEnabled, DumpFile, MinidumpDir, Overwrite, AlwaysKeepMemoryDump |
  Out-File -FilePath $kernelDumpInfoFile -Append -Encoding utf8

"`nCollected files:" | Out-File -FilePath $summaryFile -Append -Encoding utf8
Get-ChildItem -Path $outDir -File | Select-Object Name, Length, LastWriteTime |
  Format-Table -AutoSize | Out-String | Out-File -FilePath $summaryFile -Append -Encoding utf8

Write-Output "Evidence written to: $outDir"
