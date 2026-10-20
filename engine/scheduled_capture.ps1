# Record one hour of Kraken BTC/USD, then update the study.
# Run by the scheduled task that install_schedule.ps1 registers; also safe to
# run by hand:  powershell -ExecutionPolicy Bypass -File .\scheduled_capture.ps1
#
# Everything it prints goes to data\schedule.log, since a scheduled task has
# no window to show it in.

param(
    [int]$Seconds = 3600,
    [string]$Python = "python"
)

# Continue, not Stop: Python writes progress notes to stderr, and in Windows
# PowerShell 5.1 a native command's stderr under "Stop" ends the script.
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "data") | Out-Null
$log = Join-Path $PSScriptRoot "data\schedule.log"

function Write-Log([string]$text) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $text" | Add-Content -Path $log
}

# Pipe through Add-Content rather than redirecting with *>>: in Windows
# PowerShell 5.1, *>> writes UTF-16 while Add-Content writes the system
# encoding, and mixing the two in one file garbles the log.
function Invoke-Logged([string[]]$cmd) {
    & $Python @cmd 2>&1 | ForEach-Object { "$_" } | Add-Content -Path $log
}

Write-Log "start: recording $Seconds s"
Invoke-Logged @("capture.py", "--live", "kraken", "BTC/USD", "$Seconds")
Write-Log "capture exited with code $LASTEXITCODE"
Invoke-Logged @("study.py")
Write-Log "study updated (exit $LASTEXITCODE)"
