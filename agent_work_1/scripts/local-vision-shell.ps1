# Use the current PowerShell installation, independent of PATH or PowerShell 7.
function Get-LocalVisionShell {
    foreach ($shellName in @('pwsh.exe', 'powershell.exe')) {
        $shellPath = Join-Path $PSHOME $shellName
        if (Test-Path -LiteralPath $shellPath -PathType Leaf) { return $shellPath }
    }
    throw 'Cannot locate PowerShell in the current PowerShell installation.'
}

function Get-LocalVisionShellArguments {
    param([Parameter(Mandatory=$true)][string]$ScriptPath)
    # Start-Process joins ArgumentList; explicit quoting preserves paths with spaces.
    # Preserve the parent's effective policy, including a session-only setting.
    # A PowerShell 7 parent may pass its module search path to Windows PowerShell.
    Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Security/Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
    return @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', [string](Get-ExecutionPolicy), '-File', ('"' + $ScriptPath + '"'))
}
