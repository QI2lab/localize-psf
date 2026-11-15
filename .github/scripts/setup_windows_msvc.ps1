# Select the MSVC toolset used by the tested Windows CUDA 12.8 profile.
$ErrorActionPreference = 'Stop'
$installerRoot = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer'
$vswhere = Join-Path $installerRoot 'vswhere.exe'
$component = 'Microsoft.VisualStudio.Component.VC.14.38.17.8.x86.x64'
$vsRoot = & $vswhere -latest -products '*' -version '[17.0,18.0)' -property installationPath
if (-not $vsRoot) { throw 'Visual Studio 2022 is required.' }
$compatibleRoot = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires $component -property installationPath
if (-not $compatibleRoot) {
    $setup = Join-Path $installerRoot 'setup.exe'
    $installation = Start-Process -FilePath $setup -ArgumentList @(
        'modify', '--installPath', "`"$vsRoot`"", '--add', $component,
        '--quiet', '--norestart', '--nocache'
    ) -Wait -PassThru
    if ($installation.ExitCode -notin @(0, 3010)) {
        throw "MSVC 14.38 installation failed: $($installation.ExitCode)"
    }
    $compatibleRoot = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires $component -property installationPath
}
if (-not $compatibleRoot) { throw 'MSVC 14.38 is unavailable.' }
$vcvars = Join-Path $compatibleRoot 'VC\Auxiliary\Build\vcvarsall.bat'
$variables = & $env:ComSpec /d /c "call `"$vcvars`" x64 -vcvars_ver=14.38 >nul && set"
if ($LASTEXITCODE -ne 0) { throw "MSVC initialization failed: $LASTEXITCODE" }
$compilerVariables = @(
    'PATH', 'INCLUDE', 'LIB', 'LIBPATH', 'VCINSTALLDIR', 'VSINSTALLDIR',
    'VCToolsInstallDir', 'VCToolsVersion', 'WindowsSdkDir', 'WindowsSDKVersion',
    'UniversalCRTSdkDir', 'UCRTVersion', 'VSCMD_ARG_TGT_ARCH', 'VSCMD_ARG_HOST_ARCH'
)
foreach ($variable in $variables) {
    if ($variable -match '^([^=]+)=(.*)$' -and $Matches[1] -in $compilerVariables) {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        $variable | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    }
}
if (-not $env:VCToolsVersion.StartsWith('14.38.')) {
    throw "Expected MSVC 14.38, got $env:VCToolsVersion"
}
# Clear inherited CUDA settings before exercising the declared SDK. The build
# helper also overrides them when called from an initialized user environment.
foreach ($name in @('CUDA_PATH', 'CUDA_HOME', 'CUDAToolkit_ROOT', 'CUDACXX', 'CUDAHOSTCXX')) {
    "$name=" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
Write-Host "Initialized x64 MSVC $env:VCToolsVersion"
