# Assemble the Windows gfx1201 package: the build's executables and DLLs, the ROCm runtime DLLs the build loads, the hipBLASLt
# kernels for gfx1201, LICENSES.txt, README.txt and run-server-sample.ps1; then a zip and its SHA-256.
# Written with Claude (AI).
# Usage: package.ps1 -Build <build\bin> -Rocm <ROCm SDK root> -Out <dir> -Name <package name>
param(
    [Parameter(Mandatory = $true)][string]$Build,
    [Parameter(Mandatory = $true)][string]$Rocm,
    [Parameter(Mandatory = $true)][string]$Out,
    [Parameter(Mandatory = $true)][string]$Name
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$dst  = Join-Path $Out $Name
New-Item -ItemType Directory -Force $dst | Out-Null
# a multi-config generator puts the binaries in bin\Release
$src = if (Test-Path (Join-Path $Build 'Release\llama-server.exe')) { Join-Path $Build 'Release' } else { $Build }
Get-ChildItem $src -File | Where-Object { $_.Extension -in '.exe', '.dll' } | Copy-Item -Destination $dst
foreach ($dll in 'amdhip64_7.dll', 'amd_comgr.dll', 'hipblas.dll', 'rocblas.dll', 'rocsolver.dll', 'libhipblaslt.dll', 'origami.dll', 'rocm_kpack.dll') {
    Copy-Item (Join-Path $Rocm "bin\$dll") $dst
}
Copy-Item -Recurse (Join-Path $Rocm 'bin\hipblaslt\library\gfx1201') (Join-Path $dst 'gfx1201')
Copy-Item (Join-Path $here 'LICENSES.txt') $dst
Copy-Item (Join-Path $here 'README.txt') $dst
Copy-Item (Join-Path $here 'run-server-sample.ps1') $dst
if (-not (Test-Path (Join-Path $dst 'llama-server.exe'))) { throw "no llama-server.exe in $src" }
$zip = Join-Path $Out "$Name.zip"
Compress-Archive -Path (Join-Path $dst '*') -DestinationPath $zip -Force
$h = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
Set-Content -Path "$zip.sha256" -Value "$h  $Name.zip" -NoNewline
Write-Host "packaged $zip ($([math]::Round((Get-Item $zip).Length / 1MB)) MiB), sha256 $h"
