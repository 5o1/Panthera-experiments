$ErrorActionPreference = "Stop"

$ruleName = "Panthera SRT Camera UDP 9000"
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule

New-NetFirewallRule `
    -DisplayName $ruleName `
    -Description "Allow the experiment camera subnet to push SRT to Panthera orchestration." `
    -Direction Inbound `
    -Action Allow `
    -Protocol UDP `
    -LocalPort 9000 `
    -RemoteAddress "192.168.31.0/24" `
    -Profile Any | Out-Null

Get-NetFirewallRule -DisplayName $ruleName |
    Select-Object DisplayName, Enabled, Direction, Action
