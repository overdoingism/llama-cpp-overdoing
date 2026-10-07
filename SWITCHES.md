# Switches

Written with Claude (AI).  Every switch is an environment variable, read once when `llama-server` starts.  Unset means off: with
nothing set, the build behaves like plain rdna-boosts.  Set them in the shell that starts the server (`NAME=value` before the
command in bash, `$env:NAME = "value"` in PowerShell); to turn one off, unset it.

## Main switches

| variable | patch | values | what it does |
|---|---|---|---|
| `GGML_LF_KQ_BLAS` | 0001 K1 | `f16`, `fp8` | Prefill matmuls of quantized weights (not MXFP4; dense layers, not MoE experts) through hipBLASLt.  `f16`: weights converted to f16, KLD at the floor.  `fp8`: weights and activations in fp8, faster, a little more error.  Only batches of 1024 tokens or more, so it needs `-ub` ≥ 1024. |
| `GGML_LF_DRAFT_UB` | 0001 K1 | a number, e.g. `512` | Caps the draft model's ubatch (only ever lowers it).  With `-ub 2048`, `512` saves about 0.7 GiB of VRAM. |
| `GGML_LF_FA_FP8` | 0002 F8 | `1` | Prefill flash attention in fp8.  Runs only with `-fa on` and a q8_0 KV cache (`-ctk q8_0 -ctv q8_0`), on batches of 32 tokens or more over 512 or more KV cells. |
| `GGML_LF_MXFP4_DIRECT` | 0003 M4 | `blaslt` | MXFP4 weights only: prefill converts them exactly to fp8 and runs 0001's fp8 GEMM (`GGML_LF_KQ_BLAS` does not have to be set).  Batches of 64 tokens or more.  (`fp8` and `f16` select older kernels, a little slower.) |

Decode and draft verification are not affected: their batches are below every minimum above.

## Server parameters they need

| parameter | why |
|---|---|
| `-ub 2048 -b 2048` | K1 takes only batches of 1024 tokens or more; 2048 is the fastest.  The main model's compute buffer grows by about 1.1 MiB per ubatch token. |
| `-fa on -ctk q8_0 -ctv q8_0` | F8 runs only on a q8_0 KV cache with flash attention. |

## All on (as in the benchmark)

```
GGML_LF_KQ_BLAS=fp8
GGML_LF_FA_FP8=1
GGML_LF_MXFP4_DIRECT=blaslt
GGML_LF_DRAFT_UB=512
GGML_LF_DFLASH_DEV=1
llama-server ... -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0
```

## Algorithm choice (normally unset)

| variable | values | what it does |
|---|---|---|
| `GGML_LF_KQ_BLAS_ALGOS` | unset, `def.gfx1201v1`, `K1a2.<...>`, `tune`, `0` | K1 `f16`: how hipBLASLt algorithms are picked.  Unset: the built-in gfx1201 list (`def.gfx1201v1`; other GPUs: `tune`).  `K1a2.<...>`: reuse a table the log printed earlier.  `tune`: time the candidates at startup (about 9 s; the choice, and so the output, can change between runs).  `0`: hipBLASLt's own first choice.  The log prints the table in use as `GGML_LF_KQ_BLAS_ALGOS=K1a2.<...>`. |
| `GGML_LF_MXFP4_ALGOS` | unset, `heur`, `timed` | The fp8 GEMM (K1 `fp8` and M4).  Unset: the built-in gfx1201 fp8 list.  `heur`: hipBLASLt's first choice.  `timed`: time the candidates on the first full batch (can change between runs). |
| `GGML_LF_KQ_BLAS_CHUNK_MB` | a number | Converts weights in pieces of that many MB.  Saves only about 150 MB of VRAM; prefill gets slower (about 9% on Windows) and the output changes.  Not recommended. |

## Diagnostics (normally unset)

| variable | what it does |
|---|---|
| `GGML_LF_KQ_BLAS_LOG=1`, `GGML_LF_MXFP4_LOG=1` | Print each GEMM shape's algorithm choice (K1) / one line per weight tensor (M4). |
| `GGML_LF_KQ_BLAS_MIN_T` (1024), `GGML_LF_FA_FP8_MIN_T` (32), `GGML_LF_FA_FP8_MIN_KV` (512), `GGML_LF_MXFP4_MIN_T` (64) | The minimum batch (tokens) or KV length each path takes; defaults in brackets. |
| `GGML_LF_KQ_BLAS_SHARE=0`, `GGML_LF_MXFP4_SHARE=0` | Convert the activations for every matmul instead of once per input: same result, slower. |
| `GGML_LF_MXFP4_AMAX=row`, `GGML_LF_MXFP4_CONV=row` | Alternative kernels for the activation scale / the MXFP4 conversion: same result. |
| `GGML_LF_MXFP4_CFG`, `GGML_LF_MXFP4_SPLITK` | Experiments for M4's `fp8` kernel. |

## Related, not from this repo

| variable | what it does |
|---|---|
| `GGML_LF_DFLASH_DEV=1` | rdna-boosts (F1): with a DFlash draft (`--spec-type draft-dflash`) and one sequence, the target model's features stay on the GPU. |
| `HIPBLASLT_TENSILE_LIBPATH` | hipBLASLt's own: where its kernels are.  See the build guides. |

---

# 開關

本文件由 Claude（AI）撰寫。所有開關都是環境變數，`llama-server` 啟動時讀一次。不設就是關：什麼都不設時，行為和原版 rdna-boosts 相同。在啟動 server 的 shell 裡設定（bash 寫在指令前面 `名稱=值`，PowerShell 用 `$env:名稱 = "值"`）；要關掉就取消設定。

## 主要開關

| 變數 | patch | 值 | 作用 |
|---|---|---|---|
| `GGML_LF_KQ_BLAS` | 0001 K1 | `f16`、`fp8` | 量化權重（MXFP4 除外；只有一般的矩陣乘，不含 MoE 專家）的 prefill 矩陣乘改走 hipBLASLt。`f16`：權重轉 f16，KLD 與底噪相同。`fp8`：權重與 activation 都轉 fp8，更快、誤差稍大。只處理 1024 token 以上的批次，所以 `-ub` 要 ≥ 1024。 |
| `GGML_LF_DRAFT_UB` | 0001 K1 | 數字，例如 `512` | 限制草稿模型的 ubatch（只會調小）。`-ub 2048` 時設 `512` 約省 0.7 GiB 顯存。 |
| `GGML_LF_FA_FP8` | 0002 F8 | `1` | prefill 的 flash attention 改用 fp8。只在 `-fa on` 且 KV 快取為 q8_0（`-ctk q8_0 -ctv q8_0`）時作用，批次 32 token 以上、KV 512 格以上。 |
| `GGML_LF_MXFP4_DIRECT` | 0003 M4 | `blaslt` | 只對 MXFP4 權重：prefill 時把權重精確轉成 fp8，用 0001 的 fp8 矩陣乘（不必另設 `GGML_LF_KQ_BLAS`）。批次 64 token 以上。（`fp8`、`f16` 是較舊的 kernel，稍慢。） |

decode 與草稿驗證不受影響：它們的批次都低於上面的下限。

## 需要配合的 server 參數

| 參數 | 原因 |
|---|---|
| `-ub 2048 -b 2048` | K1 只處理 1024 token 以上的批次，2048 最快。主模型的計算暫存區每個 ubatch token 約 1.1 MiB。 |
| `-fa on -ctk q8_0 -ctv q8_0` | F8 只在 flash attention＋q8_0 KV 快取上作用。 |

## 全開（基準測試用的設定）

見上方英文段的區塊：`GGML_LF_KQ_BLAS=fp8`、`GGML_LF_FA_FP8=1`、`GGML_LF_MXFP4_DIRECT=blaslt`、`GGML_LF_DRAFT_UB=512`、`GGML_LF_DFLASH_DEV=1`，加上 `-ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0`。

## 演算法選擇（平常不設）

| 變數 | 值 | 作用 |
|---|---|---|
| `GGML_LF_KQ_BLAS_ALGOS` | 不設、`def.gfx1201v1`、`K1a2.<...>`、`tune`、`0` | K1 `f16` 怎麼選 hipBLASLt 演算法。不設：內建的 gfx1201 清單（`def.gfx1201v1`；其他顯卡等於 `tune`）。`K1a2.<...>`：沿用 log 之前印出的表。`tune`：啟動時實測（約 9 秒；選擇與輸出可能每次不同）。`0`：hipBLASLt 自己的第一名。log 會印出實際使用的表 `GGML_LF_KQ_BLAS_ALGOS=K1a2.<...>`。 |
| `GGML_LF_MXFP4_ALGOS` | 不設、`heur`、`timed` | fp8 矩陣乘（K1 `fp8` 與 M4）。不設：內建的 gfx1201 fp8 清單。`heur`：hipBLASLt 的第一名。`timed`：在第一個完整批次實測（可能每次不同）。 |
| `GGML_LF_KQ_BLAS_CHUNK_MB` | 數字 | 權重分段轉換，每段幾 MB。只省約 150 MB 顯存，prefill 變慢（Windows 約 9%），輸出也會改變。不建議。 |

## 診斷用（平常不設）

| 變數 | 作用 |
|---|---|
| `GGML_LF_KQ_BLAS_LOG=1`、`GGML_LF_MXFP4_LOG=1` | 印出每個矩陣形狀選到的演算法（K1）／每個權重一行（M4）。 |
| `GGML_LF_KQ_BLAS_MIN_T`（1024）、`GGML_LF_FA_FP8_MIN_T`（32）、`GGML_LF_FA_FP8_MIN_KV`（512）、`GGML_LF_MXFP4_MIN_T`（64） | 各路徑接手的最小批次（token）或 KV 長度；括號內為預設值。 |
| `GGML_LF_KQ_BLAS_SHARE=0`、`GGML_LF_MXFP4_SHARE=0` | 每個矩陣乘都重新轉換 activation，而不是每個輸入只轉一次：結果相同、較慢。 |
| `GGML_LF_MXFP4_AMAX=row`、`GGML_LF_MXFP4_CONV=row` | activation 縮放值／MXFP4 轉換的另一種 kernel：結果相同。 |
| `GGML_LF_MXFP4_CFG`、`GGML_LF_MXFP4_SPLITK` | M4 `fp8` kernel 的實驗參數。 |

## 相關但不屬於本 repo

| 變數 | 作用 |
|---|---|
| `GGML_LF_DFLASH_DEV=1` | rdna-boosts 的 F1：使用 DFlash 草稿（`--spec-type draft-dflash`）且只有一個序列時，目標模型的特徵留在 GPU 上。 |
| `HIPBLASLT_TENSILE_LIBPATH` | hipBLASLt 自己的變數：kernel 的位置，見編譯說明。 |
