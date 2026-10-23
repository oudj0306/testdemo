param([string]$PythonExe='',[ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._:/-]*$')][string]$Model='qwen3-vl:4b-instruct')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'scripts/local-vision-shell.ps1')
if($Model -match 'cloud'){throw 'Cloud models are not permitted'}
$visionRoot=Join-Path $PSScriptRoot 'data/local-vision'
if(-not $PythonExe){$candidateRoot=$PSScriptRoot;for($i=0;$i -lt 5 -and $candidateRoot;$i++){$candidate=Join-Path $candidateRoot '.venv-gpu/Scripts/python.exe';if(Test-Path -LiteralPath $candidate){$PythonExe=$candidate;break};$candidateRoot=Split-Path $candidateRoot -Parent}}
if(-not $PythonExe){$PythonExe=(Get-Command python -ErrorAction Stop).Source}
& $PythonExe (Join-Path $PSScriptRoot 'scripts/install_local_vision.py') (Join-Path $visionRoot 'runtime')
if($LASTEXITCODE -ne 0){throw 'Official portable runtime setup failed'}
$serverStarted=$null
try{
 try{$status=Invoke-RestMethod 'http://127.0.0.1:11435/api/version' -TimeoutSec 2}catch{
  $serverStarted=Start-Process -FilePath (Get-LocalVisionShell) -ArgumentList (Get-LocalVisionShellArguments -ScriptPath (Join-Path $PSScriptRoot 'start-local-vision.ps1')) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru
  for($attempt=0;$attempt -lt 30;$attempt++){try{$status=Invoke-RestMethod 'http://127.0.0.1:11435/api/version' -TimeoutSec 2;break}catch{Start-Sleep -Seconds 1}}
 }
 if(-not $status.version){throw 'Local vision server did not start'}
 Write-Host "Downloading local model $Model (no student data is sent)"
 $savedHost=$env:OLLAMA_HOST
 try{$env:OLLAMA_HOST='127.0.0.1:11435';& (Join-Path $visionRoot 'runtime/ollama.exe') pull $Model;if($LASTEXITCODE -ne 0){throw 'Local model download failed; rerun to resume'}}finally{$env:OLLAMA_HOST=$savedHost}
 Write-Host 'Setup complete. Start vision with .\start-local-vision.ps1, then start grading in another terminal.'
}finally{
 if($serverStarted){
  # Stop only this launcher's own descendant; never stop an existing user service.
  Get-CimInstance Win32_Process -Filter "ParentProcessId=$($serverStarted.Id)" | Where-Object {$_.CommandLine -like '*ollama.exe*serve*'} | ForEach-Object {Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue}
  Stop-Process -Id $serverStarted.Id -ErrorAction SilentlyContinue
 }
}
