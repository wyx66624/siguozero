param(
    [string]$Server = '192.168.130.250',
    [string]$User = 'third_onto_keeper'
)
$ErrorActionPreference = 'Stop'
Write-Host 'Connect the VPN first. Keep this window open, then visit http://127.0.0.1:8768/'
& ssh -N -L '127.0.0.1:8768:127.0.0.1:8768' -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "$User@$Server"
exit $LASTEXITCODE
