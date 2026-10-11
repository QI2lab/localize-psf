param(
    [string]$CudaPath = (Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA\v12.8'),
    [Parameter(Mandatory = $true)][string[]]$CommandArguments
)

# Use the selected SDK and local caches only within this process.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$OriginalPath = $env:PATH
$OriginalCuda = $env:CUDA_PATH
$OriginalCupyCache = $env:CUPY_CACHE_DIR
$OriginalUvCache = $env:UV_CACHE_DIR
if (-not (Test-Path -LiteralPath (Join-Path $CudaPath 'include\cuda.h'))) {
    throw "CUDA SDK headers are missing under $CudaPath. Supply -CudaPath."
}
$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
$UvPath = if ($UvCommand) { $UvCommand.Source } else { Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
try {
    $env:CUDA_PATH = $CudaPath
    $env:PATH = "$(Join-Path $CudaPath 'bin');$OriginalPath"
    $env:CUPY_CACHE_DIR = Join-Path $ProjectRoot 'build\cupy-cache'
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot '.uv-cache'
    & $UvPath run --project (Join-Path $ProjectRoot 'build\local-gpu') --locked --no-sync @CommandArguments
    if ($LASTEXITCODE -ne 0) { throw "GPU command failed with exit code $LASTEXITCODE" }
} finally {
    $env:PATH = $OriginalPath
    $env:CUDA_PATH = $OriginalCuda
    $env:CUPY_CACHE_DIR = $OriginalCupyCache
    $env:UV_CACHE_DIR = $OriginalUvCache
}
