# Register (or update) a daily Windows scheduled task that records one hour of
# Kraken BTC/USD at each of the given times, then updates the study.
#
#   powershell -ExecutionPolicy Bypass -File .\install_schedule.ps1
#   powershell -ExecutionPolicy Bypass -File .\install_schedule.ps1 -Remove
#
# Default times are one per session of the day (local time), so after five
# days every session has the ~5 recordings study.py wants before trusting a
# difference between sessions. Kraken trades around the clock.
#
# The computer has to be on. -WakeToRun asks Windows to wake it from sleep,
# which only works if wake timers are allowed in your power plan; if the
# machine was off at a start time, -StartWhenAvailable runs it when it's back.

param(
    [string[]]$Times = @("03:00", "09:30", "14:00", "21:00"),
    [string]$TaskName = "Bullpen Kraken capture",
    [switch]$Remove
)

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed '$TaskName'."
    return
}

$script = Join-Path $PSScriptRoot "scheduled_capture.ps1"
# Resolve python now, from the PATH you have in this window, because a
# scheduled task's PATH can differ.
$python = (Get-Command python -ErrorAction Stop).Source

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -Python `"$python`"" `
    -WorkingDirectory $PSScriptRoot
$triggers = $Times | ForEach-Object { New-ScheduledTaskTrigger -Daily -At $_ }
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers `
    -Settings $settings -Description "Records Kraken BTC/USD for the Bullpen order-flow study" `
    -Force | Out-Null

Write-Host "Registered '$TaskName' at $($Times -join ', ') daily, using $python."
Write-Host "Log: $(Join-Path $PSScriptRoot 'data\schedule.log')"
Write-Host "Results: python study.py   (from this folder)"
