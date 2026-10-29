# Build on Windows

For an AI assistant.  ROCm SDK for Windows (TheRock, `_rocm_sdk_devel`) with hipBLASLt, Ninja, Git for Windows (for `bash`); GPU gfx1201.
Use PowerShell.  Sources: [llama.cpp](https://github.com/ggml-org/llama.cpp),
[llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts).

```powershell
git config --global core.autocrlf false        # before cloning: patches need LF line endings
git clone https://github.com/ggml-org/llama.cpp; cd llama.cpp
git checkout 6050b467c
git clone https://github.com/stew675/llama-cpp-rdna-boosts ..\rdna-boosts
git -C ..\rdna-boosts checkout v15-6050b467c-r156     # older releases: see below
bash ../rdna-boosts/scripts/apply-all.sh .            # Git for Windows bash; makes branch rdna-boosts
Get-ChildItem C:\path\to\llama-cpp-overdoing\patches\0*-605-*.patch | Sort-Object Name | ForEach-Object { git apply --check $_.FullName; if ($?) { git apply $_.FullName } }
$root = "C:\path\to\_rocm_sdk_devel"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON `
  "-DCMAKE_PREFIX_PATH=$root" "-DCMAKE_HIP_COMPILER=$root\lib\llvm\bin\clang++.exe" `
  "-DCMAKE_C_COMPILER=$root\lib\llvm\bin\clang.exe" "-DCMAKE_CXX_COMPILER=$root\lib\llvm\bin\clang++.exe"
cmake --build build --target llama-server
$k = Get-ChildItem (Split-Path $root) -Recurse -Directory -Filter gfx1201 | Where-Object { $_.Parent.Name -eq "library" -and $_.Parent.Parent.Name -eq "hipblaslt" } | Select-Object -First 1
Copy-Item -Recurse $k.FullName build\bin\       # hipBLASLt kernels for gfx1201, next to llama-server.exe
```

- Apply in number order (0003 needs 0001; 0006 needs 0001 and 0003; 0007 needs 0004; 0008 needs 0001; 0009 needs 0002; 0015 needs 0012 and 0014; the others stand alone).  The
  patches add `.cu` files: after applying, always run the configure step again.
- r144 to r153: `0015` (v2) needs r154 or later, so the loop skips it; then apply `patches/r144/0015-605-spec-align-v1.patch`.
  On Windows use r147 or later: r143 to r146 do not compile there (`close` undeclared in `moe-expert-cache.cu`, rdna-boosts
  issue #231).
- Older releases: `v12-097cf33d9-r51` to `r59` use `patches/r59/` (check out `097cf33d9`), `v16-097cf33d9-r50` uses
  `patches/r50/`, a `v16-a55e952b8-rN` release uses `patches/a55/` (check out `a55e952b8`).  Those sets are off unless set.
- The configure step may log `UI: download dist.tar.gz... failed` (it tries to fetch the web UI); harmless, the build
  works offline, and `llama-server --no-webui` does not need it.
- **Keep `GGML_HIP_NO_VMM=ON`** (default).  With `OFF`, a long context overflows into shared GPU memory on Windows and prefill drops
  to about a quarter.  Keep `GGML_HIP_LF_HIPBLASLT=ON` (default).  Leave out `-DGGML_HIP_RCCL=1`: it matters only with
  several GPUs, and the Windows SDK may not have RCCL (the configure step then fails).
- **Copy the `gfx1201` folder next to `llama-server.exe`** (last two lines above).  On Windows hipBLASLt does not find its kernels
  in the SDK by itself: take the SDK's `hipblaslt\library\gfx1201` folder, copy it into the folder of `llama-server.exe`
  (`build\bin`), and point `HIPBLASLT_TENSILE_LIBPATH` at the copy.  Without it the log says `hipBLASLt has no ... kernels`
  and K1 stays off.
- **K1 is on by default only when every GPU the program sees is a gfx1201.**  If the integrated GPU of the CPU is visible to
  HIP (`ggml_cuda_init: found 2 ROCm devices`), either hide it (`$env:HIP_VISIBLE_DEVICES = "0"`, the R9700's index) or
  set `$env:GGML_LF_KQ_BLAS = "fp8"`.
- Run from the folder of `llama-server.exe`, e.g. `$env:HIPBLASLT_TENSILE_LIBPATH = ".\gfx1201"` (relative to the current
  folder; give the full path when starting from elsewhere), then `.\llama-server.exe ... -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0`.
  With the draft model and n-gram drafts: the whole command in [README](README.md#run).  All switches: [SWITCHES.md](SWITCHES.md).  The log must show `K1 warm-up (2048 tokens, ...): fp8 GEMM ready for 8 shapes` (the line
  `GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on` is printed in either mode).

---

# Windows 編譯

給 AI 助理看。需要 Windows 版 ROCm SDK（TheRock 的 `_rocm_sdk_devel`，含 hipBLASLt）、Ninja、Git for Windows（提供 `bash`）；GPU 為 gfx1201。在 PowerShell 操作。

1. **clone 之前**先設 `git config --global core.autocrlf false`，patch 需要 LF 換行。
2. clone [llama.cpp](https://github.com/ggml-org/llama.cpp)，checkout `6050b467c`；clone [rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)，checkout release tag `v15-6050b467c-r156`，用 Git for Windows 的 `bash` 在 llama.cpp 目錄執行 `apply-all.sh .`。舊版：`v12-097cf33d9-r51`～`r59` 用 `patches/r59/`（checkout `097cf33d9`），`v16-097cf33d9-r50` 用 `patches/r50/`，`v16-a55e952b8-rN` 用 `patches/a55/`（checkout `a55e952b8`）；這些舊版不設就是關。
3. 依編號順序 `git apply` 本 repo 的 patch（0003 需要 0001；0006 需要 0001 與 0003；0007 需要 0004；0008 需要 0001；0009 需要 0002；0015 需要 0012 與 0014；其餘可單獨套）。patch 會新增 `.cu` 檔，套完一定要重跑 cmake 設定步驟。設定時若出現 `UI: download dist.tar.gz... failed`（嘗試下載網頁介面），可以忽略。
4. r144～r153：`0015`（v2）需要 r154 以上，`git apply --check` 會失敗、跳過即可，之後改套 `patches/r144/0015-605-spec-align-v1.patch`。Windows 請用 r147 以上：r143～r146 在 Windows 上編不過（`moe-expert-cache.cu` 的 `close` 未宣告，rdna-boosts issue #231）。
5. cmake 用 Ninja，`-DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON`，編譯器與 `CMAKE_PREFIX_PATH` 指向 ROCm SDK（見上方指令）。不要加 `-DGGML_HIP_RCCL=1`：它只在多卡時有作用，而 Windows SDK 可能沒有 RCCL（設定步驟會失敗）。
6. **`GGML_HIP_NO_VMM` 一定要保持 ON**：設成 OFF 時，長上下文會溢出到共用 GPU 記憶體，prefill 掉到約四分之一。`GGML_HIP_LF_HIPBLASLT` 保持預設 ON。
7. 編 `llama-server`。
8. **把 `gfx1201` 資料夾複製到 `llama-server.exe` 旁邊**（上方指令最後兩行）：Windows 上 hipBLASLt 不會自己到 SDK 裡找 kernel。把 SDK 裡的 `hipblaslt\library\gfx1201` 資料夾複製到 `llama-server.exe` 所在的資料夾（`build\bin`），再用 `HIPBLASLT_TENSILE_LIBPATH` 指向這份複本。沒做的話 log 會出現 `hipBLASLt has no ... kernels`，K1 不會啟用。
9. **K1 只有在程式看到的 GPU 全是 gfx1201 時才預設開。** CPU 的內顯如果也被 HIP 看到（`ggml_cuda_init: found 2 ROCm devices`），就把它藏起來（`$env:HIP_VISIBLE_DEVICES = "0"`，填 R9700 的編號），或直接設 `$env:GGML_LF_KQ_BLAS = "fp8"`。
10. 在 `llama-server.exe` 所在的資料夾啟動，例如 `$env:HIPBLASLT_TENSILE_LIBPATH = ".\gfx1201"`（相對於目前資料夾；從別的地方啟動就寫完整路徑），再用 `-ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0` 啟動 `llama-server.exe`。加上草稿模型與 n-gram 草稿的完整指令見 [README](README.md#run)。所有開關見 [SWITCHES.md](SWITCHES.md)。log 要出現 `K1 warm-up (2048 tokens, ...): fp8 GEMM ready for 8 shapes`（`GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on` 這行兩種模式都會印）。
