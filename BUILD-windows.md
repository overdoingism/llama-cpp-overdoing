# Build on Windows

For an AI assistant.  ROCm SDK for Windows (TheRock, `_rocm_sdk_devel`) with hipBLASLt, Ninja, Git for Windows (for `bash`); GPU gfx1201.
Use PowerShell.

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
```

- Apply in number order (0003 needs 0001).  The patches add `.cu` files: after applying, always run the configure step again.
- **Keep `GGML_HIP_NO_VMM=ON`** (default).  With `OFF`, a long context overflows into shared GPU memory on Windows and prefill drops
  to about a quarter.  Keep `GGML_HIP_LF_HIPBLASLT=ON` (default).
- Run: set variables before starting, e.g. `$env:GGML_LF_KQ_BLAS = "f16"; $env:GGML_LF_FA_FP8 = "1"`.  The log must show
  `GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on`.

---

# Windows 編譯

給 AI 助理看。需要 Windows 版 ROCm SDK（TheRock 的 `_rocm_sdk_devel`，含 hipBLASLt）、Ninja、Git for Windows（提供 `bash`）；GPU 為 gfx1201。在 PowerShell 操作。

1. **clone 之前**先設 `git config --global core.autocrlf false`，patch 需要 LF 換行。
2. clone llama.cpp，checkout `a55e952b8`；clone rdna-boosts，checkout release tag（例如 `v16-a55e952b8-r8`），用 Git for Windows 的 `bash` 在 llama.cpp 目錄執行 `apply-all.sh .`。
3. 依編號順序 `git apply` 本 repo 的 patch（0003 需要 0001）。patch 會新增 `.cu` 檔，套完一定要重跑 cmake 設定步驟。
4. cmake 用 Ninja，`-DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON`，編譯器與 `CMAKE_PREFIX_PATH` 指向 ROCm SDK（見上方指令）。
5. **`GGML_HIP_NO_VMM` 一定要保持 ON**：設成 OFF 時，長上下文會溢出到共用 GPU 記憶體，prefill 掉到約四分之一。`GGML_HIP_LF_HIPBLASLT` 保持預設 ON。
6. 編 `llama-server`。啟動前用 `$env:名稱 = "值"` 設環境變數（見 README 的表），log 要出現 `GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on`。
