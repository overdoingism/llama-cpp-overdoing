# Patches in detail

What each patch in [`patches/`](patches) does, its switch (with the default under it), and what it needs; then the change log.  The short table is in
the [README](README.md#patches).  Written with Claude (AI).

| patch | what | switch | type | needs |
|---|---|---|---|---|
| `0001-605-k1-v4` | prefill matmuls of K-quant weights through hipBLASLt in fp8 (or f16); fp8 corrects the largest activation channels.  v4 = v3.3, on by default | `GGML_LF_KQ_BLAS`<br>(default `fp8` when every GPU is a gfx1201; `0` = off);<br>`GGML_LF_DRAFT_UB`<br>(default `512`) shrinks the draft model's ubatch | speed/lossy | hipBLASLt |
| `0002-605-f8-v5` | prefill flash attention in fp8 (and no growing F16 staging copy).  v5 = v4 with the row sum counted from the fp8 probabilities (numerator and denominator from the same numbers); v4 = v3.3, on by default | `GGML_LF_FA_FP8`<br>(default on; `0` = off) | speed/lossy | — |
| `0003-605-m4-v3` | MXFP4 weights converted exactly to fp8 for K1's fp8 GEMM.  v3 = v2, on by default | `GGML_LF_MXFP4_DIRECT`<br>(default `blaslt`; `0` = off) | speed/lossy | 0001 |
| `0004-605-reasoning-guard-v2` | an end of turn sampled inside the reasoning block becomes the block's end tag, so the model writes its answer instead of ending with the answer left in the reasoning and an empty reply (seen with clients that send earlier reasoning back as plain text, over long contexts).  v2 = v1, on by default | `LLAMA_REASONING_GUARD`<br>(default on; `0` = off) | guard | — |
| `0005-605-backend-sampling-reserve-v1` | with `--backend-sampling`, the compute buffers are reserved once per sequence instead of being freed and allocated again for every request (upstream llama.cpp behavior; about 0.25 s per request on a 27B model at `-ub 2048`) | —<br>(always on) | fix | — |
| `0006-605-mmvk-v3` | Q4_K / Q5_K / Q6_K matmuls of 1 to 8 tokens (decode and speculative verification) keep the activations in registers across several weight rows, and the matmuls that read the same input go out as one launch; not bit-identical to the stock kernel (float summation order), KLD below the run-to-run floor.  Only when llama-server verifies drafts (draft model, MTP or n-gram): one-token decode alone is faster on the stock kernel.  v3 = v2, on by default | `GGML_CUDA_MMVK`<br>(default `2`; `0` = off, `4` = also without drafts) | speed/diff | 0001, 0003 (context) |
| `0007-605-spec-lenience-v1` | with sampled drafts (llama.cpp's own `--spec-draft-sampling probabilistic`, which covers DFlash2 drafts from this base on) a drafted token is accepted with probability min(1, p / (l·q)) instead of min(1, p / q): more drafted tokens kept, and no token more than 1/l times as likely as under the model.  Only at a request temperature above 0 | `LLAMA_SPEC_LENIENCE` or the request field `spec_lenience`<br>(default off; e.g. `0.8663`) | speed/lossy | 0004 (context) |
| `0008-605-windows-moe-cache-close-v1` | the Windows build of r143 and later: two `close()` calls in `moe-expert-cache.cu` that the Windows build does not declare go through a helper (rdna-boosts issue #231); nothing changes on Linux.  rdna-boosts has the same fix since r147: leave it out from r147 on | —<br>(always on) | fix | — |
| `0009-605-fa-band-split-v1` | on RDNA4, a draft-verification batch of 9 to 16 query rows (two slots of a unified KV cache verifying at once, or an n-gram draft longer than 7) runs its attention as two halves of at most 8 rows through the decode / verify kernel instead of the generic one, which re-reads the KV cache several times: at 110K cells such a step takes 30-34% less time, and two slots decoding at once at 50K each went from 13-19% below one slot alone to 33-38% above it.  Only from 8192 KV cells (below, two launches cost more than they save).  Each row gets the same result as in a batch of 8 rows or fewer; not bit-identical to the stock generic kernel | `GGML_CUDA_FA_BAND_SPLIT`<br>(default on; `0` = off);<br>`GGML_CUDA_FA_BAND_SPLIT_MIN_KV`<br>(default `8192`) | speed/diff | 0002 (context) |
| `0012-605-dflash-adaptive-width-v1` | DFlash2 drafts w tokens per sequence, w following running per-position acceptance estimates to the width with the most tokens per unit of step cost (0.035 of a step per verified row, plus the RDNA4 band's tile step from 5 rows at long context).  The output does not change.  One request at a time: general workloads +0.1 %, low-acceptance long text +5..+11 % | `LLAMA_DFLASH_ADAPTIVE`<br>(default on; `0` = fixed width, `2` = also log the widths) | speed | — |
| `0014-605-ngram-shadow-v1` | with `--spec-type ngram-mod,draft-dflash`, the n-gram drafts are held back and compared with what the model writes; after 24 correct tokens in a row the sequence is copying and gets n-gram drafts (up to 15 tokens) without DFlash2, until one misses.  Whole-file rewrites +38..+68 %, other requests within ±1 %.  Not bit-identical while copying (16-row verification), unless `LLAMA_SPEC_NGRAM_SHADOW_EXACT=1` | `LLAMA_SPEC_NGRAM_SHADOW`<br>(default `24`; `0` = n-gram first, as upstream) | speed/diff | an n-gram type before a model drafter in `--spec-type` |
| `0015-605-spec-align-v1` | on hybrid / recurrent models (Qwen3.x) a verify batch whose sequences differ in length is split and runs the weights twice; slots verifying together now send drafts of one length (0012 widens the narrower DFlash2 drafts, a copying n-gram draft is cut to the others' length).  Two slots at once: 0012 no longer loses 16-23 %, and a slot next to a copying one no longer loses 47 % | `LLAMA_SPEC_ALIGN`<br>(default on; `0` = off) | speed | 0012, 0014 (context) |

Type: `speed/lossy` = faster, with error above the run-to-run floor (or, for 0007, a bounded change of the output distribution); `speed/diff` = faster, output not bit-identical to the stock code but within the floor; `guard` = changes behavior only in a failure case; `fix` = a bug fix; `speed/trade-off` = faster on some workloads and slower on others, so off unless you ask for it.

## Change log

**2026-10-27 (late night), `0007`:** with `LLAMA_SPEC_LENIENCE=0.8663` on a Windows build that also has `0012` and `0014` (and `--spec-type ngram-mod,draft-dflash`), a long sampled request (temperature 1) ran at about three times the usual speed from the start and gave no usable output. Until this is understood, leave `LLAMA_SPEC_LENIENCE` unset (the default) and do not send `spec_lenience`. Without it the same build runs normally.

**2026-10-27 (later), documents only:** the patches also apply on rdna-boosts `v15-6050b467c-r152` (not built or measured
there), except `0008`, whose fix rdna-boosts has since r147.  New Linux note on ROCm's `HSA_USERPTR_FOR_PAGED_MEM=0`
([README](README.md#switches)).  Defaults are now shown under each switch; the Chinese half of SWITCHES.md lists every
switch again (0009, 0012, 0014, 0015 were missing there).

**2026-10-27 (night):** new `0012` (DFlash2 adaptive verify width), `0014` (shadow n-gram) and `0015` (one draft length
per verify batch), all on by default; `0014` acts only with an n-gram type before a model drafter in `--spec-type`.
Without `0015`, two slots decoding at once lost 16-23 % with `0012`, and a slot next to a copying one lost 47 % with
`0014`.  See the [report](reports/2026-10-27.md#0012-0014-0015-draft-length).

**2026-10-27 (evening):** `0002` F8 v5 (replaces v4).  F8 added the fp32 probabilities into the row sum but multiplied V by
their fp8 conversions; a probability below the smallest fp8 subnormal (about 2^-18 of the row's maximum) dropped out of
the numerator while staying in the denominator, so the attention output came out slightly small, more so the longer the
context.  v5 sums the fp8 values themselves (the principle of Barron et al., "A Fix for Softmax and Cross-Entropy",
arXiv 2610.31642).  On a 170K-token reasoning conversation F8's own KLD drops from 0.000396 to 0.000307 (floor 0.00021);
with all defaults the change is below the floor; prefill speed is unchanged.  v4 users: apply
[`patches/increments/0002-605-f8-v4-to-v5.patch`](patches/increments/0002-605-f8-v4-to-v5.patch) (it touches only
`lf-fa-fp8.cu`, which no other patch changes), or start again from a clean tree with the new set.  See the
[report](reports/2026-10-27.md#0002-v5-the-row-sum-of-f8).

**2026-10-27 (later):** a new set for rdna-boosts `v15-6050b467c-r144` (upstream base `6050b467c`); the r59 set moved to
[`patches/r59/`](patches/r59).  **The patches are now on by default** (`0` turns one off, `GGML_LF_ALL=0` turns off every
switch left unset; [SWITCHES.md](SWITCHES.md)); 0007 stays off.  `0007` is new: llama.cpp now samples DFlash2 drafts itself
(`--spec-draft-sampling probabilistic`), so the old 0007 is gone and 0007 is the lenience of their verification (lossy,
off unless `LLAMA_SPEC_LENIENCE` is set).  New `0008` for the Windows build.  New `0009`: draft-verification batches of 9-16 rows (two slots
verifying at once) through the decode / verify attention kernel, on by default.  Speed and KLD on r144: see the
[report](reports/2026-10-27.md).

**2026-10-27:** `0007` v3 (replaces v2), a fix for `--spec-draft-p-min` above 0.  v2 decided whether a drafted token stays in the draft from that token's own probability, so which tokens were drafted depended on which were drawn, and the output no longer followed the model's distribution: in the distribution test (Python prompt, temperature 1, p-min 0.5, 500 seeds) the most frequent token at a position rose from 80% to 96% of the samples, and 9 of 36 tests gave p < 0.01.  v3 stops at the probability of the position's most likely candidate, read before the draw, and keeps a draft shorter than `--spec-draft-n-min`; the same test passes (0 of 36 below 0.05).  With the default p-min 0 nothing changes, so v2 users need to update only if they set it (`git apply -R` the v2 file, then apply v3).  Also measured since: at Qwen3.8-27B's recommended thinking-mode sampling (temperature 1.0, top-p 0.95, top-k 20), 10 recorded agent turns ran 8.7% faster with `0007` (one seed, on a later rdna-boosts base with our other patches).

**2026-10-24:** new `0007`: DFlash2 probabilistic drafting, **a trade-off, off by default** (`--spec-draft-sampling probabilistic`).  A sampled request's draft is sampled too and verified by rejection sampling: same output distribution, different tokens.  Code / math at temperature 0.7-1.0 got faster (+11% to +31% on r59, +5% to +17% on r115); low-temperature agent turns got slower (−3.9% at 0.3).  Measure on your own workload.  See the [report](reports/2026-10-24.md).

**2026-10-19:** `0001` K1 v3.3: the fp8 path's activation pass is cheaper (the largest channels chosen by a pairwise merge, and the per-channel maximum taken by the GLU kernels that write the activation).  The output is bit-identical to v3.2; prefill +2.1% / +3.3% / +2.5% at 8K / 32K / 64K (production setup).  No new switch.  See the [report](reports/2026-10-19.md).

**2026-10-18:** new `0006` (mmvk), for decode with speculative drafts: Qwen3.8-27B K-quant models take 4.5-6% less time per step (standard benchmark Q5 decode +6%), draft acceptance unchanged, KLD below the floor.  Off unless `GGML_CUDA_MMVK` is set; `2` turns it on only when llama-server verifies drafts.  See the [report](reports/2026-10-18.md).

**2026-10-15 (later):** new `0005`, a fix for `--backend-sampling`: llama.cpp re-reserved the compute graph on every request, freeing and allocating all compute buffers again.  Same output; see the [report](reports/2026-10-15.md#backend-sampling-fix-0005).

**2026-10-15:** support for rdna-boosts r51 and later (tested r59; on r59 the outputs equal r50's).  K1 v3.2 converts the weights to fp8 in one step: same output as v3, about 12% faster prefill.  New `0004`: the reasoning guard (off unless set).  When it fires, the model writes its answer a second time after the reasoning: leave room in `max_tokens`.

**2026-10-14:** `0002` F8 v3.3 fixes prompts prefilled together in one batch (`--parallel` ≥ 2): v3.1 and v3.2 zeroed the other sequences' attention in the F8 layers.  One sequence gives the same output as v3.2.  The `a55` file still has the bug: use `--parallel 1` with it.
