param(
    [int]$Port = 8001,
    [switch]$InitDbOnly
)
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$candidates = @((Join-Path $projectPath '.venv\Scripts\python.exe'))
$ancestor = Split-Path $projectPath -Parent
for ($level = 0; $level -lt 4 -and $ancestor; $level++) {
    $candidates += Join-Path $ancestor '.venv\Scripts\python.exe'
    $ancestor = Split-Path $ancestor -Parent
}
$pythonExe = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $pythonExe) {
    throw 'Create project/.venv and install dependencies with: python -m pip install -e ".[dev]"'
}
$savedPath = $env:PYTHONPATH
$savedDataDir = $env:GRADING_DATA_DIR
Push-Location -LiteralPath $projectPath
try {
    $env:PYTHONPATH = Join-Path $projectPath 'src'
    $env:GRADING_DATA_DIR = Join-Path $projectPath 'data'
    if ($InitDbOnly) {
        & $pythonExe -m grading_agent.cli --init-db
    } else {
        & $pythonExe -m grading_agent.cli --host 127.0.0.1 --port $Port
    }
    if ($LASTEXITCODE -ne 0) { throw "Optimization service exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $savedPath
    $env:GRADING_DATA_DIR = $savedDataDir
    Pop-Location
}
