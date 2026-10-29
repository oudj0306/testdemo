$ErrorActionPreference = "Stop"

$projectPath = Split-Path $PSScriptRoot -Parent
Push-Location -LiteralPath $projectPath
try {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $launcher) {
        throw "没有找到 Python Launcher。请先从 python.org 安装 64 位 Python 3.12，并勾选 Install launcher for all users。"
    }

    & py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12); print(sys.version)"
    if ($LASTEXITCODE -ne 0) {
        throw "没有找到 Python 3.12。PaddlePaddle 当前不能安装到本项目现有的 Python 3.14 环境。"
    }

    & py -3.12 -m venv .venv-ocr
    if ($LASTEXITCODE -ne 0) { throw 'OCR environment creation failed' }
    & .\.venv-ocr\Scripts\python -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
    & .\.venv-ocr\Scripts\python -m pip install -e ".[dev,local_recognition,ocr]"
    if ($LASTEXITCODE -ne 0) { throw 'OCR dependency installation failed' }
    & .\.venv-ocr\Scripts\python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'OCR dependency validation failed' }

    Write-Host "OCR 环境安装完成。"
    Write-Host '$env:OCR_PROVIDER="paddleocr"'
    Write-Host '.\.venv-ocr\Scripts\python.exe -m grading_agent.cli'
} finally {
    Pop-Location
}
