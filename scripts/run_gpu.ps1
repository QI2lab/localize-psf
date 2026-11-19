param(
    [Parameter(Mandatory = $true)][string[]]$CommandArguments
)

# Use package CUDA components and local caches only within this process.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$CudaVariables = @('CUDA_PATH', 'CUDA_HOME', 'CUDAToolkit_ROOT', 'CUDACXX', 'CUDAHOSTCXX')
$OriginalCuda = @{}
foreach ($Name in $CudaVariables) {
    $OriginalCuda[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
}
$OriginalCupyCache = $env:CUPY_CACHE_DIR
$OriginalUvCache = $env:UV_CACHE_DIR
$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
$UvPath = if ($UvCommand) { $UvCommand.Source } else { Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
try {
    foreach ($Name in $CudaVariables) {
        [Environment]::SetEnvironmentVariable($Name, $null, 'Process')
    }
    $env:CUPY_CACHE_DIR = Join-Path $ProjectRoot 'build\cupy-cache'
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot '.uv-cache'
    & $UvPath run --project $ProjectRoot --locked --no-sync @CommandArguments
    if ($LASTEXITCODE -ne 0) { throw "GPU command failed with exit code $LASTEXITCODE" }
} finally {
    foreach ($Name in $CudaVariables) {
        [Environment]::SetEnvironmentVariable($Name, $OriginalCuda[$Name], 'Process')
    }
    $env:CUPY_CACHE_DIR = $OriginalCupyCache
    $env:UV_CACHE_DIR = $OriginalUvCache
}
