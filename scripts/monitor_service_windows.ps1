param(
    [switch]$Register,
    [switch]$Run,
    [switch]$Remove,
    [string]$Distribution = 'Ubuntu-24.04'
)
$ErrorActionPreference = 'Stop'
$taskName = 'SiguoZero Monitor'
$scriptPath = $PSCommandPath
$projectRoot = Split-Path -Parent $PSScriptRoot
$outputDirectory = Join-Path $projectRoot 'output/local_console'
$null = New-Item -ItemType Directory -Path $outputDirectory -Force
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$taskIdentity = $identity.User.Value
$powershellPath = Join-Path $env:WINDIR 'System32/WindowsPowerShell/v1.0/powershell.exe'

if ($Register) {
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing -and $existing.Description -ne 'Start the SiguoZero monitor service and keep its WSL instance alive.') {
        throw 'A task with this name exists and is not owned by this installer.'
    }
    if ($scriptPath.Contains('"') -or $Distribution -match '["\r\n]') { throw 'Unsupported path or distribution name.' }
    $arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $scriptPath + '" -Run -Distribution "' + $Distribution + '"'
    $action = New-ScheduledTaskAction -Execute $powershellPath -Argument $arguments -WorkingDirectory $projectRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskIdentity
    $principal = New-ScheduledTaskPrincipal -UserId $taskIdentity -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    $null = Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
        -Description 'Start the SiguoZero monitor service and keep its WSL instance alive.' -Force
    Start-ScheduledTask -TaskName $taskName
    Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State
} elseif ($Remove) {
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing -and $existing.Description -ne 'Start the SiguoZero monitor service and keep its WSL instance alive.') {
        throw 'Refusing to remove a task not owned by this installer.'
    }
    if ($existing) {
        Stop-ScheduledTask -TaskName $taskName
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }
} elseif ($Run) {
    $linuxScript = (& wsl.exe --distribution $Distribution --user root --exec wslpath -u (Join-Path $projectRoot 'tools/monitor_keepalive.py')).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $linuxScript -or $linuxScript -match '["\r\n]') { throw 'Cannot resolve the WSL keepalive script.' }
    # This PowerShell host is already hidden by the scheduled-task action. Keep
    # wsl.exe in the foreground and let PowerShell pass native arguments: WSL's
    # option parser can treat manually added quotes as part of a distro name.
    & (Join-Path $env:WINDIR 'System32/wsl.exe') --distribution $Distribution --user root --exec /usr/bin/python3 $linuxScript `
        1> (Join-Path $outputDirectory 'windows-keeper.stdout.log') `
        2> (Join-Path $outputDirectory 'windows-keeper.stderr.log')
    exit $LASTEXITCODE
} else {
    Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | Select-Object TaskName, State
}
