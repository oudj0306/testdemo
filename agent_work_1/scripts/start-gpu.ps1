param([switch]$CheckOnly, [int]$Port = 8001)
$ErrorActionPreference = 'Stop'
& (Join-Path (Split-Path $PSScriptRoot -Parent) 'start-optimization.ps1') -Port $Port -CheckOcr:$CheckOnly