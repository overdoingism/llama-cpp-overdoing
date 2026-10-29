llama-cpp-overdoing for Windows, AMD gfx1201 (Radeon AI PRO R9700, RX 9070 / 9070 XT)
====================================================================================

llama.cpp + llama-cpp-rdna-boosts + the patches of https://github.com/overdoingism/llama-cpp-overdoing, built for gfx1201
only, with the ROCm 10.0.0 runtime DLLs and the hipBLASLt kernels for gfx1201 included.  Other GPUs: use rdna-boosts
directly (this package's speed-ups are RDNA4-only).  An RX 9060 XT (gfx1200) is RDNA4 too but untested here: build it
yourself (BUILD-windows.md in the repository; an AI assistant can do it for you).

Run from this folder, in PowerShell:

    $env:HIPBLASLT_TENSILE_LIBPATH = ".\gfx1201"
    .\llama-server.exe -m <model.gguf> -ngl 99 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0

- If HIP also sees the CPU's integrated GPU (the log says "found 2 ROCm devices"), hide it with
  $env:HIP_VISIBLE_DEVICES = "0" (the R9700's index), or the fp8 prefill (K1) stays off.
- The log should show "K1 warm-up (2048 tokens, ...): fp8 GEMM ready".
- All switches: SWITCHES.md in the repository.

Check the download: compare the SHA-256 of the zip with the .sha256 file (PowerShell: Get-FileHash <zip>), and with the
GitHub CLI: gh attestation verify <zip> --repo overdoingism/llama-cpp-overdoing (it was built by the repository's GitHub
Actions workflow from the public sources).

Licenses: LICENSES.txt (llama.cpp, rdna-boosts, this repository: MIT; the ROCm components: MIT, BSD, Apache 2.0 with LLVM
exceptions).  No warranty.  Written with Claude (AI).
