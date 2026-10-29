llama-cpp-overdoing for Windows, AMD gfx1201 (Radeon AI PRO R9700, RX 9070 / 9070 XT)
====================================================================================

llama.cpp + llama-cpp-rdna-boosts + the patches of https://github.com/overdoingism/llama-cpp-overdoing, built for gfx1201
only, with the ROCm 10.0.0 runtime DLLs and the hipBLASLt kernels for gfx1201 included.  Other GPUs: use rdna-boosts
directly (this package's speed-ups are RDNA4-only).  An RX 9060 XT (gfx1200) is RDNA4 too but untested here: build it
yourself (BUILD-windows.md in the repository; an AI assistant can do it for you).

Run
---

The patches' speed-ups are on by default.  Speculative decoding (much of the generation speed) needs a draft model and
server parameters; run-server-sample.ps1 has them.  Open it (or a copy of it) in a text editor, fill in the two paths at
the top (the model, and the DFlash2 draft model), then in PowerShell, in this folder:

    .\run-server-sample.ps1

- The draft model is for Qwen3.8-27B only: https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF (Q4_K_M).  For any other
  model set the draft path to "" (no speculative decoding).
- The script runs llama-server with a 131072-token context, 1 slot, at http://127.0.0.1:8080.  Other llama-server
  arguments can follow it and win over its own, e.g.  .\run-server-sample.ps1 --ctx-size 65536 --port 8100
  .\run-server-sample.ps1 -DryRun prints the command without starting.
- If Windows refuses to run the script (downloaded files are blocked): Unblock-File .\run-server-sample.ps1, or
  powershell -ExecutionPolicy Bypass -File .\run-server-sample.ps1
- If HIP also sees the CPU's integrated GPU (the log says "found 2 ROCm devices"), run $env:HIP_VISIBLE_DEVICES = "0"
  (the R9700's index) first, or the fp8 prefill (K1) stays off.
- The log should show "K1 warm-up (2048 tokens, ...): fp8 GEMM ready".
- Optional, lossy: $env:LLAMA_SPEC_LENIENCE = "0.8663" keeps more sampled-draft tokens at temperature > 0 (SWITCHES.md).

Without the script:

    $env:HIPBLASLT_TENSILE_LIBPATH = ".\gfx1201"
    .\llama-server.exe -m <model.gguf> -ngl 99 -c 131072 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 --kv-unified `
      -md <Qwen3.8-27B-DFlash2-Q4_K_M.gguf> -ngld 99 --spec-draft-n-max 5 `
      --spec-type ngram-mod,draft-dflash --spec-ngram-mod-n-max 15 --spec-ngram-mod-n-min 8 `
      --spec-draft-sampling probabilistic

All switches: SWITCHES.md in the repository.

Check the download
------------------

Compare the SHA-256 of the zip with the .sha256 file (PowerShell: Get-FileHash <zip>), and with the GitHub CLI:
gh attestation verify <zip> --repo overdoingism/llama-cpp-overdoing (it was built by the repository's GitHub Actions
workflow from the public sources).

Licenses: LICENSES.txt (llama.cpp, rdna-boosts, this repository: MIT; the ROCm components: MIT, BSD, Apache 2.0 with LLVM
exceptions).  No warranty.  Written with Claude (AI).

----------------------------------------------------------------------------------------------------------------------------

中文說明
--------

llama.cpp＋llama-cpp-rdna-boosts＋本 repo 的 patch，只為 gfx1201 編譯，附 ROCm 10.0.0 執行期 DLL 與 gfx1201 的
hipBLASLt kernel。其他顯卡請直接用 rdna-boosts（本包的加速只在 RDNA4 上作用）。RX 9060 XT（gfx1200）也是 RDNA4，但沒測過，
請自行編譯（repo 裡的 BUILD-windows.md；可以請 AI 助理代勞）。

執行：patch 的加速預設就開。推測解碼（生成速度的一大部分）需要草稿模型與 server 參數，run-server-sample.ps1 裡都設好了。
用文字編輯器打開 run-server-sample.ps1（或它的複本），填好開頭的兩個路徑（模型、DFlash2 草稿模型），再在這個資料夾開 PowerShell：

    .\run-server-sample.ps1

- 草稿模型只適用 Qwen3.8-27B：https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF（Q4_K_M）。其他模型請把草稿路徑設成 ""
  （不做推測解碼）。
- 腳本以 131072 token 上下文、1 個 slot 啟動，網址 http://127.0.0.1:8080。後面可以接其他 llama-server 參數，會蓋過腳本的設定，
  例如  .\run-server-sample.ps1 --ctx-size 65536 --port 8100 ；.\run-server-sample.ps1 -DryRun 只印出指令、不啟動。
- Windows 不讓腳本執行時（下載的檔案會被封鎖）：Unblock-File .\run-server-sample.ps1，或
  powershell -ExecutionPolicy Bypass -File .\run-server-sample.ps1
- HIP 也看到 CPU 內顯時（log 出現 "found 2 ROCm devices"），先執行 $env:HIP_VISIBLE_DEVICES = "0"（R9700 的編號），
  否則 fp8 prefill（K1）不會開。
- log 要出現 "K1 warm-up (2048 tokens, ...): fp8 GEMM ready"。
- 選用、有損：$env:LLAMA_SPEC_LENIENCE = "0.8663" 在溫度大於 0 時多收一些抽樣草稿（見 SWITCHES.md）。

不用腳本時的完整指令見上方英文段落。下載檢查：zip 的 SHA-256 要和 .sha256 檔相同；有 GitHub CLI 可再跑
gh attestation verify <zip> --repo overdoingism/llama-cpp-overdoing。授權見 LICENSES.txt，不提供任何保證。
本說明由 Claude（AI）撰寫。
