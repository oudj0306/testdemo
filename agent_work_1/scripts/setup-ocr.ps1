$ErrorActionPreference = "Stop"

$launcher = Get-Command py -ErrorAction SilentlyContinue
if (-not $launcher) {
    throw "没有找到 Python Launcher。请先从 python.org 安装 64 位 Python 3.12，并勾选 Install launcher for all users。"
}

& py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12); print(sys.version)"
if ($LASTEXITCODE -ne 0) {
    throw "没有找到 Python 3.12。PaddlePaddle 当前不能安装到本项目现有的 Python 3.14 环境。"
}

& py -3.12 -m venv .venv-ocr
& .\.venv-ocr\Scripts\python -m pip install --upgrade pip setuptools wheel
& .\.venv-ocr\Scripts\python -m pip install -e ".[dev,ocr]"
& .\.venv-ocr\Scripts\python -m pip check

Write-Host "OCR 环境安装完成。"
Write-Host '$env:OCR_PROVIDER="paddleocr"'
Write-Host '.\.venv-ocr\Scripts\python.exe -m grading_agent.cli'
