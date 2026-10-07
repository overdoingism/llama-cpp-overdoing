# Build on Windows

For an AI assistant.  ROCm SDK for Windows (TheRock, `_rocm_sdk_devel`) with hipBLASLt, Ninja, Git for Windows (for `bash`); GPU gfx1201.
Use PowerShell.  Sources: [llama.cpp](https://github.com/ggml-org/llama.cpp),
[llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts).

```powershell
git config --global core.autocrlf false        # before cloning: patches need LF line endings
git clone https://github.com/ggml-org/llama.cpp; cd llama.cpp
git checkout a55e952b8
git clone https://github.com/stew675/llama-cpp-rdna-boosts ..\rdna-boosts
git -C ..\rdna-boosts checkout v16-a55e952b8-r8      # or another v16-a55e952b8-rN
bash ../rdna-boosts/scripts/apply-all.sh .            # Git for Windows bash; makes branch rdna-boosts
Get-ChildItem C:\path\to\llama-cpp-overdoing\patches\0*-a55-*.patch | Sort-Object Name | ForEach-Object { git apply --check $_.FullName; if ($?) { git apply $_.FullName } }
$root = "C:\path\to\_rocm_sdk_devel"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON `
  "-DCMAKE_PREFIX_PATH=$root" "-DCMAKE_HIP_COMPILER=$root\lib\llvm\bin\clang++.exe" `
  "-DCMAKE_C_COMPILER=$root\lib\llvm\bin\clang.exe" "-DCMAKE_CXX_COMPILER=$root\lib\llvm\bin\clang++.exe"
cmake --build build --target llama-server
$k = Get-ChildItem (Split-Path $root) -Recurse -Directory -Filter gfx1201 | Where-Object { $_.Parent.Name -eq "library" -and $_.Parent.Parent.Name -eq "hipblaslt" } | Select-Object -First 1
Copy-Item -Recurse $k.FullName build\bin\       # hipBLASLt kernels for gfx1201, next to llama-server.exe
```

- Apply in number order (0003 needs 0001).  The patches add `.cu` files: after applying, always run the configure step again.
- **Keep `GGML_HIP_NO_VMM=ON`** (default).  With `OFF`, a long context overflows into shared GPU memory on Windows and prefill drops
  to about a quarter.  Keep `GGML_HIP_LF_HIPBLASLT=ON` (default).
- **Copy the `gfx1201` folder next to `llama-server.exe`** (last two lines above).  On Windows hipBLASLt does not find its kernels
  in the SDK by itself: take the SDK's `hipblaslt\library\gfx1201` folder, copy it into the folder of `llama-server.exe`
  (`build\bin`), and point `HIPBLASLT_TENSILE_LIBPATH` at the copy.  Without it the log says
  `GGML_LF_KQ_BLAS: hipBLASLt has no f16 kernels ...` and K1 stays off.
- Run from the folder of `llama-server.exe`.  Set variables before starting, e.g.
  `$env:HIPBLASLT_TENSILE_LIBPATH = ".\gfx1201"; $env:GGML_LF_KQ_BLAS = "f16"; $env:GGML_LF_FA_FP8 = "1"` (`.\gfx1201` is relative
  to the current folder; give the full path when starting from elsewhere).  All switches: [SWITCHES.md](SWITCHES.md).  The log must show
  `GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on`.

---

# Windows 編譯

給 AI 助理看。需要 Windows 版 ROCm SDK（TheRock 的 `_rocm_sdk_devel`，含 hipBLASLt）、Ninja、Git for Windows（提供 `bash`）；GPU 為 gfx1201。在 PowerShell 操作。

1. **clone 之前**先設 `git config --global core.autocrlf false`，patch 需要 LF 換行。
2. clone [llama.cpp](https://github.com/ggml-org/llama.cpp)，checkout `a55e952b8`；clone [rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)，checkout release tag（例如 `v16-a55e952b8-r8`），用 Git for Windows 的 `bash` 在 llama.cpp 目錄執行 `apply-all.sh .`。
3. 依編號順序 `git apply` 本 repo 的 patch（0003 需要 0001）。patch 會新增 `.cu` 檔，套完一定要重跑 cmake 設定步驟。
4. cmake 用 Ninja，`-DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON`，編譯器與 `CMAKE_PREFIX_PATH` 指向 ROCm SDK（見上方指令）。
5. **`GGML_HIP_NO_VMM` 一定要保持 ON**：設成 OFF 時，長上下文會溢出到共用 GPU 記憶體，prefill 掉到約四分之一。`GGML_HIP_LF_HIPBLASLT` 保持預設 ON。
6. 編 `llama-server`。
7. **把 `gfx1201` 資料夾複製到 `llama-server.exe` 旁邊**（上方指令最後兩行）：Windows 上 hipBLASLt 不會自己到 SDK 裡找 kernel。把 SDK 裡的 `hipblaslt\library\gfx1201` 資料夾複製到 `llama-server.exe` 所在的資料夾（`build\bin`），再用 `HIPBLASLT_TENSILE_LIBPATH` 指向這份複本。沒做的話 log 會出現 `GGML_LF_KQ_BLAS: hipBLASLt has no f16 kernels ...`，K1 不會啟用。
8. 在 `llama-server.exe` 所在的資料夾啟動。啟動前用 `$env:名稱 = "值"` 設環境變數（所有開關見 [SWITCHES.md](SWITCHES.md)），例如 `$env:HIPBLASLT_TENSILE_LIBPATH = ".\gfx1201"`（相對於目前資料夾；從別的地方啟動就寫完整路徑）。log 要出現 `GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on`。
