# Switches

Written with Claude (AI).  Every switch is an environment variable, read once when `llama-server` starts.  **The patches are
on by default** (the default is under each variable): to turn one off, set it to `0`.  `GGML_LF_ALL=0` makes every switch you leave unset
default to off; the build then behaves like plain rdna-boosts, apart from the two fixes (0005, 0008).  `0007` is the
exception: lossy, so off unless you ask for it.  Set the variables in the shell that starts the server (`NAME=value` before
the command in bash, `$env:NAME = "value"` in PowerShell).

(The sets in `patches/r59/`, `patches/r50/` and `patches/a55/` are the older ones: there every switch is off unless set.)

## Main switches

| variable | patch | values | what it does |
|---|---|---|---|
| `GGML_LF_KQ_BLAS`<br>(default `fp8` when every GPU is a gfx1201; otherwise off) | 0001 K1 | `fp8`, `f16`, `0` | Prefill matmuls of quantized weights (not MXFP4; dense layers, not MoE experts) through hipBLASLt.  `fp8`: weights and activations in fp8, the fastest.  `f16`: weights converted to f16, KLD at the floor.  Only batches of 1024 tokens or more, so it needs `-ub` ≥ 1024.  The default needs the built-in gfx1201 algorithm lists; on other GPUs set it explicitly (the choice is then timed at startup and can change between runs). |
| `GGML_LF_KQ_BLAS_FP8_KEEP`<br>(default `64`) | 0001 K1 | a number; `0` = off | K1 `fp8` only: per input batch, the channels with the largest activations (up to 248, rounded up to a multiple of 8) get a second fp8 term for their weight and activation rounding errors, appended to the same GEMM.  About half of the fp8 error above `f16` goes away for about 1% of prefill speed. |
| `GGML_LF_DRAFT_UB`<br>(default `512`) | 0001 K1 | a number; `0` = no cap | Caps the draft model's ubatch (only ever lowers it).  With `-ub 2048`, `512` saves about 0.7 GiB of VRAM. |
| `GGML_LF_FA_FP8`<br>(default on) | 0002 F8 | `1`, `0` | Prefill flash attention in fp8.  Runs only on RDNA4, with `-fa on` and a q8_0 KV cache (`-ctk q8_0 -ctv q8_0`), on batches of 32 tokens or more over 512 or more KV cells. |
| `GGML_LF_MXFP4_DIRECT`<br>(default `blaslt`) | 0003 M4 | `blaslt`, `0` | MXFP4 weights only: prefill converts them exactly to fp8 and runs 0001's fp8 GEMM.  Batches of 64 tokens or more.  (`fp8` and `f16` select older kernels, a little slower.) |
| `LLAMA_REASONING_GUARD`<br>(default on) | 0004 | `1`, `0` | When the model samples an end of turn inside its reasoning block, the block's end tag (`</think>`) is emitted instead and generation goes on, so the answer lands in the reply instead of the reasoning.  Nothing before that point changes; it never shortens the reasoning, and a reply cut by `max_tokens` is left as it is.  When it fires the model usually writes its answer again after the reasoning (leave room in `max_tokens`), and the log prints `emitting the end tag`.  Needs a template whose reasoning end tag ends with one token (Qwen's `</think>`); otherwise the log says it is off. |
| `GGML_CUDA_MMVK`<br>(default `2`) | 0006 mmvk | `0`-`4` | Q4_K / Q5_K / Q6_K matmuls of 1 to 8 tokens (decode and speculative verification) run a kernel that keeps the activations in registers across several weight rows.  `2`: also launches the matmuls that read the same input (gate + up, q + k + v, DeltaNet qkv + z) as one kernel; `1`: one launch each (same numbers, about 1% slower).  `1` and `2` are on only when llama-server verifies speculative drafts (a draft model, MTP or an n-gram type): one-token decode alone is 1-1.6% faster on the stock kernel.  `3` / `4` are the same, always.  Silent on a GPU it does not run on (RDNA4 only).  The server log prints `mmvk on (...)` or `mmvk off (...)`.  Not bit-identical to the stock kernel (float summation order); decode and verification give the same results as each other. |
| `GGML_CUDA_MMVK_EXT`<br>(default `1`) | 0006 mmvk | bit mask | `1`: groups may mix weight types.  `2`: IQ4_XS / IQ4_NL / MXFP4 through mmvk too (their stock kernel is already close to the bandwidth limit; normally leave it). |
| `LLAMA_SPEC_LENIENCE`<br>(default off, `1`) | 0007 | a number between 0 and 1 | **Known problem, leave unset for now** (see the [README](README.md#patches)).  **Lossy.**  With sampled drafts (llama.cpp's own `--spec-draft-sampling probabilistic`) and a request temperature above 0, a drafted token is accepted with probability min(1, p / (l·q)) instead of min(1, p / q); the rest of the verification is unchanged, so no token becomes more than 1/l times as likely as under the model.  The request field `spec_lenience` sets it per request.  Has no effect at temperature 0 or with greedy drafts.  See the [2026-10-27 report](reports/2026-10-27.md) for what `0.8663` did to speed and quality. |
| `GGML_CUDA_FA_BAND_SPLIT`<br>(default on) | 0009 | `1`, `0` | RDNA4: a draft-verification batch of 9 to 16 query rows (two slots of a unified KV cache verifying at once, or an n-gram draft longer than 7) runs its attention as two halves of at most 8 rows through the decode / verify kernel instead of the generic one.  Each row gets the result a batch of 8 rows or fewer gives (not bit-identical to the generic kernel).  Only with `-fa on`, from `GGML_CUDA_FA_BAND_SPLIT_MIN_KV` KV cells (default `8192`; `0` = any length): below that the two launches cost more than they save. |
| `LLAMA_DFLASH_ADAPTIVE`<br>(default on) | 0012 | `1`, `2`, `0` | DFlash2 drafts each sequence's own width (1 to `--spec-draft-n-max`) from running acceptance estimates; the output does not change.  `2` also logs the width changes and, per request, the verifications per width.  `0`: the fixed width.  `LLAMA_DFLASH_ADAPTIVE_COST` (`0.035`) and `LLAMA_DFLASH_ADAPTIVE_JUMP` (`0.08`) set the cost model. |
| `LLAMA_SPEC_NGRAM_SHADOW`<br>(default `24`) | 0014 | a number; `0` = off | Only with an n-gram type ahead of a model drafter (`--spec-type ngram-mod,draft-dflash`).  The n-gram drafts are held back until they have predicted this many of the model's tokens in a row; then they are verified (up to `--spec-ngram-mod-n-max`) without DFlash2, until one misses.  `0`: the n-gram drafts always go first, as upstream.  `LLAMA_SPEC_NGRAM_SHADOW_EXACT=1` cuts them to 7 tokens, so the greedy output is the one without the n-gram type. |
| `LLAMA_SPEC_ALIGN`<br>(default on) | 0015 | `1`, `0` | Hybrid / recurrent models only: sequences that verify drafts in the same batch send drafts of one length (DFlash2's adaptive width aligns up to the others, up to `--spec-draft-n-max`; a longer draft is cut to the shortest), so the verify batch is not split into ubatches. |

0001-0003 do not touch decode and draft verification: their batches are below every minimum above.  0006 and 0009 are only
for them.  0007 changes only how sampled drafts are accepted; 0012, 0014 and 0015 only what is drafted.

## Server parameters they need

| parameter | why |
|---|---|
| `-ub 2048 -b 2048` | K1 takes only batches of 1024 tokens or more; 2048 is the fastest.  The main model's compute buffer grows by about 1.1 MiB per ubatch token: lower it (e.g. 1536) if a long context does not fit. |
| `-fa on -ctk q8_0 -ctv q8_0` | F8 runs only on a q8_0 KV cache with flash attention. |
| `--spec-type ngram-mod,draft-dflash --spec-ngram-mod-n-max 15 --spec-ngram-mod-n-min 8` | Only for 0014: an n-gram type before DFlash2.  Without it 0014 does nothing. |
| `--spec-draft-sampling probabilistic` | Only for 0007.  llama.cpp's own parameter: sample the draft and verify it by rejection sampling (exact).  Faster on sampled code / math at temperature 0.7-1.0, can be slower at low temperature: measure on your own workload. |

## Turning everything off

`GGML_LF_ALL=0`: every unset switch above defaults to off.  A switch you set still wins (e.g. `GGML_LF_ALL=0 GGML_LF_FA_FP8=1`
turns on F8 alone).

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
| `GGML_CUDA_FA_STAGE_MAX_MB` | rdna-boosts: cap (MiB) of the flash-attention F16 staging copy, which only grows.  Where F8 runs (every GPU RDNA4) and the variable is unset, 0002 sets it to 1: the batches F8 does not take (9-31 tokens) then read the q8_0 cache directly, same result and speed, and the copy (about 1.2 GB at 240K context; on Windows it ends up in shared memory) is never made.  An explicit value still wins; `0` means no cap. |
| `GGML_LF_DFLASH_DEV` | rdna-boosts (F1), on by default: with a DFlash draft the target model's features stay on the GPU.  `0` turns it off. |
| `HIPBLASLT_TENSILE_LIBPATH` | hipBLASLt's own: where its kernels are.  See the build guides. |
| `HIP_VISIBLE_DEVICES` | HIP's own: which GPUs the program sees.  K1's default needs every visible GPU to be a gfx1201 (see the Windows build guide). |
| `HSA_USERPTR_FOR_PAGED_MEM` | ROCm's own, Linux.  We set it to `0`: ROCm then allocates pinned host memory (`hipHostMalloc`, the `ROCm_Host` buffers) through the driver instead of as pageable pages registered with the GPU, which the kernel may swap out or move, stopping every GPU queue of the process until the page is back.  See the [README](README.md#switches).  Not needed on Windows. |

---

# 開關

本文件由 Claude（AI）撰寫。所有開關都是環境變數，`llama-server` 啟動時讀一次。**patch 預設都是開的**（預設值寫在每個變數下方）：要關哪一個，就把它設成 `0`。`GGML_LF_ALL=0` 會讓所有沒設的開關預設改成關；這時行為和原版 rdna-boosts 相同，只差兩個修正（0005、0008）。`0007` 是例外：它有損，所以不設就是關。在啟動 server 的 shell 裡設定（bash 寫在指令前面 `名稱=值`，PowerShell 用 `$env:名稱 = "值"`）。

（`patches/r59/`、`patches/r50/`、`patches/a55/` 是舊版：那些版本不設就是關。）

## 主要開關

| 變數 | patch | 值 | 作用 |
|---|---|---|---|
| `GGML_LF_KQ_BLAS`<br>（預設：所有 GPU 都是 gfx1201 時為 `fp8`，否則關） | 0001 K1 | `fp8`、`f16`、`0` | 量化權重（MXFP4 除外；只有一般的矩陣乘，不含 MoE 專家）的 prefill 矩陣乘改走 hipBLASLt。`fp8`：權重與 activation 都轉 fp8，最快。`f16`：權重轉 f16，KLD 與底噪相同。只處理 1024 token 以上的批次，所以 `-ub` 要 ≥ 1024。預設值依賴內建的 gfx1201 演算法清單；其他顯卡請明確設定（這時會在啟動時實測挑選，結果可能每次不同）。 |
| `GGML_LF_KQ_BLAS_FP8_KEEP`<br>（預設 `64`） | 0001 K1 | 數字；`0` 為關 | 只對 K1 `fp8`：每個輸入批次挑出 activation 最大的通道（最多 248，取 8 的倍數），把這些通道的權重與 activation 的 fp8 捨入誤差，當成第二個 fp8 項接在同一次矩陣乘裡。fp8 比 `f16` 多出的誤差約減半，prefill 約慢 1%。 |
| `GGML_LF_DRAFT_UB`<br>（預設 `512`） | 0001 K1 | 數字；`0` 為不限 | 限制草稿模型的 ubatch（只會調小）。`-ub 2048` 時 `512` 約省 0.7 GiB 顯存。 |
| `GGML_LF_FA_FP8`<br>（預設開） | 0002 F8 | `1`、`0` | prefill 的 flash attention 改用 fp8。只在 RDNA4、`-fa on` 且 KV 快取為 q8_0（`-ctk q8_0 -ctv q8_0`）時作用，批次 32 token 以上、KV 512 格以上。 |
| `GGML_LF_MXFP4_DIRECT`<br>（預設 `blaslt`） | 0003 M4 | `blaslt`、`0` | 只對 MXFP4 權重：prefill 時把權重精確轉成 fp8，用 0001 的 fp8 矩陣乘。批次 64 token 以上。（`fp8`、`f16` 是較舊的 kernel，稍慢。） |
| `LLAMA_REASONING_GUARD`<br>（預設開） | 0004 | `1`、`0` | 模型在思考區裡抽到「結束回合」時，改成輸出思考區的結束標籤（`</think>`）並繼續生成，答案就會落在正文而不是思考區。在那之前的內容完全不變；不會縮短思考，被 `max_tokens` 截斷的回覆也照舊。觸發時模型通常會在思考之後把答案再寫一次（`max_tokens` 要留餘裕），log 會印 `emitting the end tag`。需要模板的思考結束標籤以單一 token 結尾（Qwen 的 `</think>`），否則 log 會說它沒開。 |
| `GGML_CUDA_MMVK`<br>（預設 `2`） | 0006 mmvk | `0`～`4` | Q4_K／Q5_K／Q6_K 在 1～8 個 token 時的矩陣乘（decode 與推測解碼的驗證）改用 mmvk：activation 留在暫存器裡，給多列權重重複使用。`2`：讀同一個輸入的矩陣乘（gate＋up、q＋k＋v、DeltaNet qkv＋z）合併成一次發射；`1`：各自發射（數值相同，約慢 1%）。`1`、`2` 只在 llama-server 驗證推測解碼草稿時開（草稿模型、MTP 或 ngram）：只有單 token decode 時原版 kernel 快 1～1.6%。`3`／`4` 相同但一律開。在不支援的 GPU 上不作用（只支援 RDNA4）。server log 會印 `mmvk on (...)` 或 `mmvk off (...)`。和原版 kernel 不逐位元相同（浮點加總順序不同）；decode 與驗證彼此一致。 |
| `GGML_CUDA_MMVK_EXT`<br>（預設 `1`） | 0006 mmvk | 位元 | `1`：合併發射可混合不同權重型別。`2`：IQ4_XS／IQ4_NL／MXFP4 也走 mmvk（這些型別的原版 kernel 已接近頻寬上限，平常不設）。 |
| `LLAMA_SPEC_LENIENCE`<br>（預設關，`1`） | 0007 | 0 到 1 之間的數字 | **已知問題，目前請不要設定**（2026-10-27：Windows build 同時有 0012、0014 與 `--spec-type ngram-mod,draft-dflash` 時，溫度 1 的長請求從一開始就以約三倍速度輸出、沒有可用的結果，原因調查中；不設它時同一個 build 正常）。**有損。** 使用抽樣草稿（llama.cpp 自己的 `--spec-draft-sampling probabilistic`）且請求溫度大於 0 時，草稿字的接受機率從 min(1, p/q) 改成 min(1, p/(l·q))；驗證的其餘部分不變，所以任何字的機率最多變成原本的 1/l 倍。請求欄位 `spec_lenience` 可逐請求設定。溫度 0 或 greedy 草稿時沒有作用。`0.8663` 對速度與品質的影響見 [2026-10-27 報告](reports/2026-10-27.md)。 |
| `GGML_CUDA_FA_BAND_SPLIT`<br>（預設開） | 0009 | `1`、`0` | RDNA4：9～16 個查詢列的草稿驗證批次（unified KV 下兩個 slot 同時驗證，或 n-gram 草稿超過 7 個 token）把注意力切成兩半、每半最多 8 列，走 decode／驗證用的 kernel，而不是一般 kernel。每一列得到的結果和 8 列以下的批次相同（和一般 kernel 不逐位元相同）。只在 `-fa on`、KV 達 `GGML_CUDA_FA_BAND_SPLIT_MIN_KV` 格（預設 `8192`；`0` 為不限長度）時作用：更短時兩次發射的成本大於省下的。 |
| `LLAMA_DFLASH_ADAPTIVE`<br>（預設開） | 0012 | `1`、`2`、`0` | DFlash2 依照持續更新的接受率估計，為每個序列各自選草稿寬度（1 到 `--spec-draft-n-max`）；輸出不變。`2` 另外記錄寬度變化，以及每個請求在各寬度的驗證次數。`0`：固定寬度。成本模型用 `LLAMA_DFLASH_ADAPTIVE_COST`（`0.035`）與 `LLAMA_DFLASH_ADAPTIVE_JUMP`（`0.08`）設定。 |
| `LLAMA_SPEC_NGRAM_SHADOW`<br>（預設 `24`） | 0014 | 數字；`0` 為關 | 只在模型草稿器前面有 n-gram 類型時作用（`--spec-type ngram-mod,draft-dflash`）。n-gram 草稿先不送出，等它連續猜中模型這麼多個 token 之後，才不經 DFlash2 直接送驗證（最多 `--spec-ngram-mod-n-max` 個），直到猜錯一次為止。`0`：n-gram 草稿一律優先，和上游相同。`LLAMA_SPEC_NGRAM_SHADOW_EXACT=1` 把它截到 7 個 token，greedy 輸出就和沒有 n-gram 類型時相同。 |
| `LLAMA_SPEC_ALIGN`<br>（預設開） | 0015 | `1`、`0` | 只對 hybrid／recurrent 模型：在同一批次裡驗證草稿的序列送出同樣長度的草稿（DFlash2 的自適應寬度往上對齊其他序列，最多到 `--spec-draft-n-max`；較長的草稿截到最短的長度），驗證批次就不會被拆成多個 ubatch。 |

0001～0003 不影響 decode 與草稿驗證：它們的批次都低於上面的下限。0006 與 0009 只作用在這兩者。0007 只改變抽樣草稿的接受方式；0012、0014、0015 只改變草稿的內容。

## 需要配合的 server 參數

| 參數 | 原因 |
|---|---|
| `-ub 2048 -b 2048` | K1 只處理 1024 token 以上的批次，2048 最快。主模型的計算暫存區每個 ubatch token 約 1.1 MiB：長上下文放不下時可以降低（例如 1536）。 |
| `-fa on -ctk q8_0 -ctv q8_0` | F8 只在 flash attention＋q8_0 KV 快取上作用。 |
| `--spec-type ngram-mod,draft-dflash --spec-ngram-mod-n-max 15 --spec-ngram-mod-n-min 8` | 只有 0014 需要：n-gram 類型放在 DFlash2 前面。沒有它，0014 不會作用。 |
| `--spec-draft-sampling probabilistic` | 只有 0007 需要。llama.cpp 自己的參數：草稿改為抽樣，以拒絕取樣驗證（無損）。溫度 0.7～1.0 的寫程式／數學變快，低溫時可能變慢：請用自己的工作負載實測。 |

## 全部關掉

`GGML_LF_ALL=0`：上面所有沒設的開關預設改成關。有設的開關仍以設定為準（例如 `GGML_LF_ALL=0 GGML_LF_FA_FP8=1` 只開 F8）。

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
| `GGML_CUDA_FA_STAGE_MAX_MB` | rdna-boosts 的變數：flash attention F16 暫存複本的上限（MiB），這個暫存區只增不減。F8 有在跑（所有 GPU 都是 RDNA4）且沒設這個變數時，0002 把它設成 1：F8 不接手的批次（9～31 token）改直接讀 q8_0 快取，結果和速度相同，也不會產生那份複本（24 萬上下文約 1.2 GB；Windows 上會落到共用記憶體）。明確設定時以設定為準；`0` 表示不設上限。 |
| `GGML_LF_DFLASH_DEV` | rdna-boosts 的 F1，預設開：使用 DFlash 草稿時，目標模型的特徵留在 GPU 上。`0` 為關。 |
| `HIPBLASLT_TENSILE_LIBPATH` | hipBLASLt 自己的變數：kernel 的位置，見編譯說明。 |
| `HIP_VISIBLE_DEVICES` | HIP 自己的變數：程式看得到哪些 GPU。K1 的預設值要求看得到的 GPU 全是 gfx1201（見 Windows 編譯說明）。 |
| `HSA_USERPTR_FOR_PAGED_MEM` | ROCm 自己的變數，Linux 用。我們設成 `0`：ROCm 的固定主機記憶體（`hipHostMalloc`、`ROCm_Host` 緩衝區）改由驅動配置，而不是登記給 GPU 的一般可分頁記憶體；後者被核心換出或搬動時，整個行程的 GPU 佇列都會停住，直到該頁回來。詳見 [README](README.md#switches)。Windows 不需要。 |
