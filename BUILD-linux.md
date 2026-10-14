# Build on Linux

For an AI assistant.  ROCm with hipBLASLt installed; GPU gfx1201.  Sources: [llama.cpp](https://github.com/ggml-org/llama.cpp),
[llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts).

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 097cf33d9
git clone https://github.com/stew675/llama-cpp-rdna-boosts ../rdna-boosts
git -C ../rdna-boosts checkout v16-097cf33d9-r50         # r50 only: r51 and later are not supported yet
bash ../rdna-boosts/scripts/apply-all.sh .                 # makes branch rdna-boosts
for p in /path/to/llama-cpp-overdoing/patches/0*-097-*.patch; do git apply --check "$p" && git apply "$p"; done
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON
cmake --build build --target llama-server -j
```

- Apply in number order (0003 needs 0001).  The patches add `.cu` files: after applying, always run the configure step again.
- For a `v16-a55e952b8-rN` release, check out `a55e952b8` and use the `0*-a55-*` files instead.
- On r50 the configure step may log `UI: download dist.tar.gz... failed` (it tries to fetch the web UI); harmless, the build
  works offline, and `llama-server --no-webui` does not need it.
- Keep `GGML_HIP_NO_VMM=ON` and `GGML_HIP_LF_HIPBLASLT=ON` (both default).  Other flags as in the rdna-boosts README
  (`-DGGML_HIP_RCCL=1` only for multi-GPU).  If cmake cannot find the HIP compiler, add `-DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++`.
- hipBLASLt loads its kernels from the `hipblaslt/library` folder next to `libhipblaslt.so` (TheRock: `hipblaslt/library/gfx1201/`);
  nothing to copy as long as the loader uses that ROCm `lib` folder (for a pip / TheRock install, put it in `LD_LIBRARY_PATH`).
  If the log says `GGML_LF_KQ_BLAS: hipBLASLt has no f16 kernels ...`, set `HIPBLASLT_TENSILE_LIBPATH` to the `gfx1201` folder.
- Run: `GGML_LF_KQ_BLAS=f16 GGML_LF_FA_FP8=1 ./build/bin/llama-server ...` (all switches: [SWITCHES.md](SWITCHES.md)).  The log must show
  `GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on`.

---

# Linux 編譯

給 AI 助理看。需要已安裝 ROCm 與 hipBLASLt；GPU 為 gfx1201。

1. clone [llama.cpp](https://github.com/ggml-org/llama.cpp)，checkout `097cf33d9`。
2. clone [rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)，checkout release tag `v16-097cf33d9-r50`（r51 以後尚未支援；舊基底 `v16-a55e952b8-rN` 要 checkout `a55e952b8`，改用 `0*-a55-*` 的 patch），在 llama.cpp 目錄裡執行它的 `scripts/apply-all.sh .`。
3. 依編號順序 `git apply` 本 repo 的 patch（0003 需要 0001）。patch 會新增 `.cu` 檔，套完一定要重跑 cmake 設定步驟。r50 設定時若出現 `UI: download dist.tar.gz... failed`（嘗試下載網頁介面），可以忽略。
4. cmake：`-DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON`，`GGML_HIP_LF_HIPBLASLT` 保持預設 ON；其他參數照 rdna-boosts README（多卡才需要 `-DGGML_HIP_RCCL=1`）。找不到 HIP 編譯器時加 `-DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++`。
5. hipBLASLt 會從 `libhipblaslt.so` 旁邊的 `hipblaslt/library` 載入 kernel（TheRock 在 `hipblaslt/library/gfx1201/`），只要載入的是那個 ROCm `lib` 資料夾就不必複製（pip／TheRock 安裝要把它放進 `LD_LIBRARY_PATH`）。log 出現 `GGML_LF_KQ_BLAS: hipBLASLt has no f16 kernels ...` 時，把 `HIPBLASLT_TENSILE_LIBPATH` 設成那個 `gfx1201` 資料夾。
6. 編 `llama-server`。執行時設環境變數（所有開關見 [SWITCHES.md](SWITCHES.md)），log 要出現 `GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on`。
