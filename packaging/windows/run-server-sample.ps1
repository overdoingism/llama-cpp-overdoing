# Starts llama-server from this package with the settings we use on an R9700 (gfx1201).  Written with Claude (AI), after
# roger's own launcher.  A sample: fill in the two paths below (here or in a copy), then run it in PowerShell:
#   .\run-server-sample.ps1
# Anything you add after it goes to llama-server as it is (.\run-server-sample.ps1 --ctx-size 65536);
# -DryRun prints the command without starting.

$modelPath = "C:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf"
# DFlash2 draft model, Qwen3.8-27B only (https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF, Q4_K_M).
# For any other model set it to "" (no speculative decoding).
$draftPath = "C:\models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

# ---- the settings we use; nothing to edit below ----
$PSNativeCommandUseErrorActionPreference = $false
$here = $PSScriptRoot
# hipBLASLt does not find its kernels by itself on Windows: the gfx1201 folder of this package
$env:HIPBLASLT_TENSILE_LIBPATH = Join-Path $here "gfx1201"

$llamaArgs = @(
    "--model", $modelPath
    "--n-gpu-layers", "99"
    "--ctx-size", "131072"
    "--parallel", "1"
    "--kv-unified"
    # K1 (fp8 prefill) is fastest at 2048-token batches; F8 (fp8 prefill attention) needs flash attention and a q8_0 KV cache
    "-ub", "2048"
    "-b", "2048"
    "--flash-attn", "on"
    "--cache-type-k", "q8_0"
    "--cache-type-v", "q8_0"
    "--jinja"
    "--host", "127.0.0.1"
    "--port", "8080"
)
if ($draftPath) {
    $llamaArgs += @(
        "--model-draft", $draftPath
        "--n-gpu-layers-draft", "99"
        "--spec-draft-n-max", "5"
        # n-gram drafts (up to 15 tokens, at least 8) take over only while the reply copies text (0014)
        "--spec-type", "ngram-mod,draft-dflash"
        "--spec-ngram-mod-n-max", "15"
        "--spec-ngram-mod-n-min", "8"
        # sampled drafts, verified exactly by rejection sampling (llama.cpp's own; matters at temperature > 0)
        "--spec-draft-sampling", "probabilistic"
    )
}
$dry = $args -contains "-DryRun"
$llamaArgs += @($args | Where-Object { $_ -ne "-DryRun" })

foreach ($f in @($modelPath, $draftPath) | Where-Object { $_ }) {
    if (-not (Test-Path $f)) { Write-Host "not found: $f  (fill in the paths at the top of $(Split-Path -Leaf $PSCommandPath))"; exit 1 }
}
Write-Host "HIPBLASLT_TENSILE_LIBPATH=$env:HIPBLASLT_TENSILE_LIBPATH"
Write-Host ("llama-server.exe " + (($llamaArgs | ForEach-Object { if ("$_" -match '\s') { '"' + $_ + '"' } else { "$_" } }) -join ' '))
if ($dry) { exit 0 }
& (Join-Path $here "llama-server.exe") @llamaArgs
