# Build on Linux

For an AI assistant.  ROCm with hipBLASLt installed; GPU gfx1201.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout a55e952b8
git clone https://github.com/stew675/llama-cpp-rdna-boosts ../rdna-boosts
git -C ../rdna-boosts checkout v16-a55e952b8-r8          # or another v16-a55e952b8-rN
bash ../rdna-boosts/scripts/apply-all.sh .                 # makes branch rdna-boosts
for p in /path/to/llama-cpp-overdoing/patches/0*-a55-*.patch; do git apply --check "$p" && git apply "$p"; done
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON
cmake --build build --target llama-server -j
```

- Apply in number order (0003 needs 0001).  The patches add `.cu` files: after applying, always run the configure step again.
- Keep `GGML_HIP_NO_VMM=ON` and `GGML_HIP_LF_HIPBLASLT=ON` (both default).  Other flags as in the rdna-boosts README
  (`-DGGML_HIP_RCCL=1` only for multi-GPU).  If cmake cannot find the HIP compiler, add `-DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++`.
- Run: `GGML_LF_KQ_BLAS=f16 GGML_LF_FA_FP8=1 ./build/bin/llama-server ...`.  The log must show
  `GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on`.

---

# Linux 編譯

給 AI 助理看。需要已安裝 ROCm 與 hipBLASLt；GPU 為 gfx1201。

1. clone llama.cpp，checkout `a55e952b8`。
2. clone rdna-boosts，checkout release tag（例如 `v16-a55e952b8-r8`），在 llama.cpp 目錄裡執行它的 `scripts/apply-all.sh .`。
3. 依編號順序 `git apply` 本 repo 的 patch（0003 需要 0001）。patch 會新增 `.cu` 檔，套完一定要重跑 cmake 設定步驟。
4. cmake：`-DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON`，`GGML_HIP_LF_HIPBLASLT` 保持預設 ON；其他參數照 rdna-boosts README（多卡才需要 `-DGGML_HIP_RCCL=1`）。找不到 HIP 編譯器時加 `-DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++`。
5. 編 `llama-server`。執行時設環境變數（見 README 的表），log 要出現 `GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on`。
