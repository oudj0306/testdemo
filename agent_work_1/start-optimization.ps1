param([int]$Port = 8001, [switch]$InitDbOnly, [switch]$CheckOcr, [switch]$PrintRuntime, [string]$PythonExe = '')
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$roots = @($projectPath)
$ancestor = Split-Path $projectPath -Parent
for ($level = 0; $level -lt 4 -and $ancestor; $level++) {
    if (Test-Path -LiteralPath (Join-Path $ancestor 'src\grading_agent\config.py')) { $roots += $ancestor }
    $ancestor = Split-Path $ancestor -Parent
}
$candidates = @()
foreach ($root in $roots) {
    foreach ($environment in @('.venv-gpu', '.venv-ocr', '.venv')) {
        $candidate = Join-Path $root "$environment\Scripts\python.exe"
        if (Test-Path -LiteralPath $candidate) { $candidates += $candidate }
    }
}
if ($PythonExe) {
    if (-not (Test-Path -LiteralPath $PythonExe)) { throw 'PythonExe does not exist' }
    $pythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
} else {
    $pythonExe = $null
    foreach ($candidate in $candidates) {
        & $candidate -c "import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in ('paddle','paddleocr','fastapi')) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonExe = $candidate; break }
    }
    if (-not $pythonExe) { $pythonExe = $candidates | Select-Object -First 1 }
}
if (-not $pythonExe) { throw 'Create .venv and install dependencies with: python -m pip install -e ".[dev]"' }
$names = @('PYTHONPATH','GRADING_DATA_DIR','PADDLE_PDX_CACHE_HOME','HF_HOME')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
Push-Location -LiteralPath $projectPath
try {
    $env:PYTHONPATH = Join-Path $projectPath 'src'
    $env:GRADING_DATA_DIR = Join-Path $projectPath 'data'
    foreach ($root in $roots) {
        if (Test-Path -LiteralPath (Join-Path $root '.cache\paddlex\official_models')) {
            if (-not $env:PADDLE_PDX_CACHE_HOME) { $env:PADDLE_PDX_CACHE_HOME = Join-Path $root '.cache\paddlex' }
            if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $root '.cache\huggingface' }
            break
        }
    }
    if ($PrintRuntime) {
        & $pythonExe -c "import json,sys,os; from grading_agent.config import get_settings; s=get_settings(); print(json.dumps(dict(python=sys.executable,data_dir=str(s.data_dir),ocr_provider=s.ocr_provider,ocr_device=s.ocr_device,cache=os.environ.get('PADDLE_PDX_CACHE_HOME'))))"
    } elseif ($CheckOcr) {
        & $pythonExe -c "import json; from grading_agent.config import get_settings; from grading_agent.services.ocr_diagnostics import check_ocr; s=get_settings(); r=check_ocr(s.ocr_device,s.data_dir/'runtime-check'); print(json.dumps(r,ensure_ascii=True)); raise SystemExit(0 if r['ok'] else 1)"
    } elseif ($InitDbOnly) {
        & $pythonExe -m grading_agent.cli --init-db
    } else {
        Write-Host "Python: $pythonExe"
        Write-Host "Data: $env:GRADING_DATA_DIR"
        & $pythonExe -m grading_agent.cli --host 127.0.0.1 --port $Port
    }
    if ($LASTEXITCODE -ne 0) { throw "Optimization service/check exited with code $LASTEXITCODE" }
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
    Pop-Location
}