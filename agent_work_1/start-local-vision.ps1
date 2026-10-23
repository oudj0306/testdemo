param([int]$Port=11435)
$ErrorActionPreference='Stop'
$visionRoot=Join-Path $PSScriptRoot 'data/local-vision'
$runtimeExe=Join-Path $visionRoot 'runtime/ollama.exe'
if(-not (Test-Path -LiteralPath $runtimeExe)){throw 'Run setup-local-vision.ps1 first'}
try{$ready=Invoke-RestMethod "http://127.0.0.1:$Port/api/version" -TimeoutSec 2;if($ready.version){Write-Host "Local vision service already running on $Port";return}}catch{}
$names=@('OLLAMA_HOST','OLLAMA_MODELS','OLLAMA_NO_CLOUD','OLLAMA_NUM_PARALLEL','OLLAMA_MAX_LOADED_MODELS','OLLAMA_CONTEXT_LENGTH','OLLAMA_VULKAN')
$previous=@{};foreach($name in $names){$previous[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try{
 $env:OLLAMA_HOST="127.0.0.1:$Port";$env:OLLAMA_MODELS=Join-Path $visionRoot 'models';$env:OLLAMA_NO_CLOUD='1'
 $env:OLLAMA_NUM_PARALLEL='1';$env:OLLAMA_MAX_LOADED_MODELS='1';$env:OLLAMA_CONTEXT_LENGTH='8192';$env:OLLAMA_VULKAN='1'
 Write-Host "Local-only vision service: http://127.0.0.1:$Port (close with Ctrl+C)"
 & $runtimeExe serve
 if($LASTEXITCODE -ne 0){throw "Local vision service exited: $LASTEXITCODE"}
}finally{foreach($name in $names){[Environment]::SetEnvironmentVariable($name,$previous[$name],'Process')}}
