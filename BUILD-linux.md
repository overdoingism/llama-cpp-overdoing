# Build on Linux

For an AI assistant.  ROCm with hipBLASLt installed; GPU gfx1201.  Sources: [llama.cpp](https://github.com/ggml-org/llama.cpp),
[llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts).

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 6050b467c
git clone https://github.com/stew675/llama-cpp-rdna-boosts ../rdna-boosts
git -C ../rdna-boosts checkout v16-6050b467c-r184
bash ../rdna-boosts/scripts/apply-all.sh .                 # makes branch rdna-boosts
for p in /path/to/llama-cpp-overdoing/patches/0*-605-*.patch; do git apply --check "$p" && git apply "$p"; done
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON
cmake --build build --target llama-server -j
```

- Apply in number order (0003 needs 0001; 0006 needs 0001 and 0003; 0007 needs 0004; 0008 needs 0001; 0009 needs 0002; 0015 needs 0012; the others stand alone).  The
  patches add `.cu` files: after applying, always run the configure step again.
- The configure step may log `UI: download dist.tar.gz... failed` (it tries to fetch the web UI); harmless, the build
  works offline, and `llama-server --no-webui` does not need it.
- Keep `GGML_HIP_NO_VMM=ON` and `GGML_HIP_LF_HIPBLASLT=ON` (both default).  Other flags as in the rdna-boosts README
  (`-DGGML_HIP_RCCL=1` matters only with several GPUs).  If cmake cannot find the HIP compiler, add
  `-DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++`.
- hipBLASLt loads its kernels from the `hipblaslt/library` folder next to `libhipblaslt.so` (TheRock: `hipblaslt/library/gfx1201/`);
  nothing to copy as long as the loader uses that ROCm `lib` folder (for a pip / TheRock install, put it in `LD_LIBRARY_PATH`).
  If the log says `hipBLASLt has no ... kernels`, set `HIPBLASLT_TENSILE_LIBPATH` to the `gfx1201` folder.
- Run `HSA_USERPTR_FOR_PAGED_MEM=0 ./build/bin/llama-server ... -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0` (the ROCm
  variable keeps the GPU from stalling when the kernel swaps or moves host memory: [README](README.md#switches)); with the draft model, the whole command is in [README](README.md#run).  The patches are on by default
  ([SWITCHES.md](SWITCHES.md)); the log must show `K1 warm-up (2048 tokens, ...): fp8 GEMM ready for 8 shapes` (the line
  `GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on` is printed in either mode).

---

# Linux 編譯

給 AI 助理看。需要已安裝 ROCm 與 hipBLASLt；GPU 為 gfx1201。

1. clone [llama.cpp](https://github.com/ggml-org/llama.cpp)，checkout `6050b467c`。
2. clone [rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)，checkout release tag `v16-6050b467c-r184`，在 llama.cpp 目錄裡執行它的 `scripts/apply-all.sh .`。
3. 依編號順序 `git apply` 本 repo 的 patch（0003 需要 0001；0006 需要 0001 與 0003；0007 需要 0004；0008 需要 0001；0009 需要 0002；0015 需要 0012；其餘可單獨套）。patch 會新增 `.cu` 檔，套完一定要重跑 cmake 設定步驟。設定時若出現 `UI: download dist.tar.gz... failed`（嘗試下載網頁介面），可以忽略。r144～r153：`0015`（v2）需要 r154 以上，`git apply --check` 會失敗、跳過即可，之後改套 `patches/r144/0015-605-spec-align-v1.patch`。
4. cmake：`-DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON`，`GGML_HIP_LF_HIPBLASLT` 保持預設 ON；其他參數照 rdna-boosts README（`-DGGML_HIP_RCCL=1` 只在多卡時有作用）。找不到 HIP 編譯器時加 `-DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++`。
5. hipBLASLt 會從 `libhipblaslt.so` 旁邊的 `hipblaslt/library` 載入 kernel（TheRock 在 `hipblaslt/library/gfx1201/`），只要載入的是那個 ROCm `lib` 資料夾就不必複製（pip／TheRock 安裝要把它放進 `LD_LIBRARY_PATH`）。log 出現 `hipBLASLt has no ... kernels` 時，把 `HIPBLASLT_TENSILE_LIBPATH` 設成那個 `gfx1201` 資料夾。
6. 編 `llama-server`，用 `HSA_USERPTR_FOR_PAGED_MEM=0 ./build/bin/llama-server ... -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0` 啟動（這個 ROCm 變數讓核心換出或搬動主機記憶體時 GPU 不會停住，見 [README](README.md#switches)）；加上草稿模型的完整指令見 [README](README.md#run)。patch 預設都是開的（開關見 [SWITCHES.md](SWITCHES.md)）；log 要出現 `K1 warm-up (2048 tokens, ...): fp8 GEMM ready for 8 shapes`（`GGML_LF_KQ_BLAS: f16 prefill GEMM through hipBLASLt on` 這行兩種模式都會印）。
