param([string]$PythonExe = "", [string]$Cuda = "cu118")
$ErrorActionPreference = "Stop"
$projectPath = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectPath
if ($Cuda -notin @('cu118', 'cu126')) { throw 'Cuda must be cu118 or cu126' }
if (-not $PythonExe) {
    if (Test-Path -LiteralPath '.venv-gpu\Scripts\python.exe') {
        $PythonExe = Join-Path $projectPath '.venv-gpu\Scripts\python.exe'
    } else {
        $PythonExe = (& py -3.12 -c "import sys; print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -ne 0 -or -not $PythonExe) { throw 'Install Python 3.12 x64 or pass -PythonExe with a Python 3.11-3.13 path.' }
    }
}
& $PythonExe -c "import sys,struct; assert (3,11)<=sys.version_info[:2]<=(3,13) and struct.calcsize('P')==8, 'Requires Python 3.11-3.13 x64'"
if ($LASTEXITCODE -ne 0) { throw 'Unsupported Python version' }
if (-not (Test-Path -LiteralPath '.venv-gpu\Scripts\python.exe')) {
    & $PythonExe -m venv .venv-gpu
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
$gpuPython = Join-Path $projectPath '.venv-gpu\Scripts\python.exe'
& $gpuPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
# Do not use the CPU [ocr] extra in this environment: CPU/GPU wheels share paddle files.
$runtimePackage = if ($Cuda -eq 'cu118') { 'nvidia-cuda-runtime-cu11' } else { 'nvidia-cuda-runtime-cu12' }
# Both indexes publish version 3.3.0. pip otherwise keeps an installed wheel
# from the wrong CUDA index because its version number already matches.
& $gpuPython -c "import importlib.metadata as m,sys; sys.exit(0 if any(d.metadata['Name']=='paddlepaddle-gpu' and d.version=='3.3.0' and any('$runtimePackage' in r for r in (d.requires or [])) for d in m.distributions()) else 1)"
if ($LASTEXITCODE -ne 0) {
    & $gpuPython -m pip install --force-reinstall --no-deps paddlepaddle-gpu==3.3.0 --index-url "https://www.paddlepaddle.org.cn/packages/stable/$Cuda/"
    if ($LASTEXITCODE -ne 0) { throw 'Paddle CUDA build selection failed' }
}
& $gpuPython -m pip install paddlepaddle-gpu==3.3.0 --index-url "https://www.paddlepaddle.org.cn/packages/stable/$Cuda/"
if ($LASTEXITCODE -ne 0) { throw 'Paddle GPU installation failed' }
& $gpuPython -m pip install -e '.[dev]' 'paddleocr>=3.7,<4'
if ($LASTEXITCODE -ne 0) { throw 'Application/OCR installation failed' }
& $gpuPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed' }
& $gpuPython -c "import paddle; assert paddle.is_compiled_with_cuda(); assert paddle.device.cuda.device_count()>0; paddle.set_device('gpu:0'); print('GPU:', paddle.device.cuda.get_device_name()); print('GPU tensor:', (paddle.ones([2,2]) @ paddle.ones([2,2])).numpy()); paddle.utils.run_check()"
if ($LASTEXITCODE -ne 0) { throw 'GPU execution check failed; check NVIDIA driver and runtime DLLs' }
Write-Host 'GPU environment verified. Start with: .\.venv-gpu\Scripts\python.exe -m grading_agent.cli'
Write-Host 'In Settings choose PaddleOCR and gpu:0. First OCR call downloads public model weights.'
