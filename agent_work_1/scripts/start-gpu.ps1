param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectPath
if (-not (Test-Path -LiteralPath '.venv-gpu\Scripts\python.exe')) { throw 'Run scripts/setup-gpu.ps1 first' }
$env:PADDLE_PDX_CACHE_HOME = Join-Path $projectPath '.cache\paddlex'
$env:HF_HOME = Join-Path $projectPath '.cache\huggingface'
if ($CheckOnly) {
    & .\.venv-gpu\Scripts\python.exe scripts/check-gpu.py
    if ($LASTEXITCODE -ne 0) { throw 'GPU OCR check failed; see output above' }
    exit 0
}
& .\.venv-gpu\Scripts\python.exe -m grading_agent.cli
if ($LASTEXITCODE -ne 0) { throw 'Grading service exited with an error; see output above' }
