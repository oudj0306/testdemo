param([Parameter(Mandatory=$true)][string]$InputPath, [Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
$app = $null
$document = $null
$ownsApp = $false
$security = $null
$alerts = $null
$updateLinks = $null
try {
    foreach ($progId in @('KWPS.Application', 'Word.Application')) {
        try { $app = New-Object -ComObject $progId; break } catch { $app = $null }
    }
    if (-not $app) { throw 'No Word/WPS automation available' }
    $ownsApp = ($app.Documents.Count -eq 0)
    if ($ownsApp) { $app.Visible = $false }
    $security = $app.AutomationSecurity
    $app.AutomationSecurity = 3
    if ([int]$app.AutomationSecurity -ne 3) { throw 'Cannot disable document macros' }
    $alerts = $app.DisplayAlerts
    $app.DisplayAlerts = 0
    try { $updateLinks = $app.Options.UpdateLinksAtOpen; $app.Options.UpdateLinksAtOpen = $false } catch {}
    # Read-only, no recent-files entry, no macro execution, no password dialog.
    $document = $app.Documents.Open($InputPath, $false, $true, $false, 'grading-no-password-dialog')
    if (-not $document) { throw 'Document could not be opened' }
    $document.ExportAsFixedFormat($OutputPath, 17)
    if (-not (Test-Path -LiteralPath $OutputPath)) { throw 'PDF export did not produce output' }
} finally {
    if ($document) { try { $document.Close(0) } catch {}; [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document) }
    if ($app) {
        try { if ($null -ne $security) { $app.AutomationSecurity = $security } } catch {}
        try { if ($null -ne $alerts) { $app.DisplayAlerts = $alerts } } catch {}
        try { if ($null -ne $updateLinks) { $app.Options.UpdateLinksAtOpen = $updateLinks } } catch {}
        if ($ownsApp) { try { $app.Quit(0) } catch {} }
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)
    }
}
