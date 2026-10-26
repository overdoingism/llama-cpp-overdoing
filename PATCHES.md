# Patches in detail

What each patch in [`patches/`](patches) does, how to turn it on, and what it needs; then the change log.  The short table is in
the [README](README.md#patches).  Written with Claude (AI).

| patch | what | turn on | type | needs |
|---|---|---|---|---|
| `0001-097-k1-v3.3` | prefill matmuls of K-quant weights through hipBLASLt (f16, or fp8; fp8 corrects the largest activation channels; v3.3: a cheaper fp8 activation pass, same output as v3.2) | `GGML_LF_KQ_BLAS=f16` (or `fp8`); `GGML_LF_DRAFT_UB=512` shrinks the draft model's ubatch | speed/lossy | hipBLASLt |
| `0002-097-f8-v3.3` | prefill flash attention in fp8 (and no growing F16 staging copy) | `GGML_LF_FA_FP8=1` | speed/lossy | — |
| `0003-097-m4-v2` | MXFP4 weights converted exactly to fp8 for K1's fp8 GEMM | `GGML_LF_MXFP4_DIRECT=blaslt` | speed/lossy | 0001 |
| `0004-097-reasoning-guard-v1` | an end of turn sampled inside the reasoning block becomes the block's end tag, so the model writes its answer instead of ending with the answer left in the reasoning and an empty reply (seen with clients that send earlier reasoning back as plain text, over long contexts) | `LLAMA_REASONING_GUARD=1` | guard | — |
| `0005-097-backend-sampling-reserve-v1` | with `--backend-sampling`, the compute buffers are reserved once per sequence instead of being freed and allocated again for every request (upstream llama.cpp behavior; about 0.25 s per request on a 27B model at `-ub 2048`) | always on | fix | — |
| `0006-097-mmvk-v2` | Q4_K / Q5_K / Q6_K matmuls of 1 to 8 tokens (decode and speculative verification) keep the activations in registers across several weight rows, and the matmuls that read the same input go out as one launch; not bit-identical to the stock kernel (float summation order), KLD below the run-to-run floor.  On only when llama-server verifies drafts (draft model, MTP or n-gram): one-token decode alone is faster on the stock kernel | `GGML_CUDA_MMVK=2` (`4`: also without drafts) | speed/diff | 0001, 0003 (context) |
| `0007-097-dflash2-probabilistic-v3` | with a DFlash2 draft (`--spec-type draft-dflash`) and a request temperature above 0, the draft path is sampled from its candidates (first cut by the request's top-k / top-p / min-p) and the target verifies it by rejection sampling, as llama.cpp already does for draft-simple and MTP.  The output follows the same distribution but is not the same tokens as with greedy drafting; greedy requests are untouched.  **A trade-off**: faster on sampled code / math at temperature 0.7-1.0, slower on low-temperature agent turns (see the 2026-10-24 report) | server parameter `--spec-draft-sampling probabilistic` | speed/trade-off | — |

Type: `speed/lossy` = faster, with error above the run-to-run floor; `speed/diff` = faster, output not bit-identical to the stock code but within the floor; `guard` = changes behavior only in a failure case; `fix` = a bug fix; `speed/trade-off` = faster on some workloads and slower on others, so off unless you ask for it.

## Change log

**2026-10-27:** `0007` v3 (replaces v2), a fix for `--spec-draft-p-min` above 0.  v2 decided whether a drafted token stays in the draft from that token's own probability, so which tokens were drafted depended on which were drawn, and the output no longer followed the model's distribution: in the distribution test (Python prompt, temperature 1, p-min 0.5, 500 seeds) the most frequent token at a position rose from 80% to 96% of the samples, and 9 of 36 tests gave p < 0.01.  v3 stops at the probability of the position's most likely candidate, read before the draw, and keeps a draft shorter than `--spec-draft-n-min`; the same test passes (0 of 36 below 0.05).  With the default p-min 0 nothing changes, so v2 users need to update only if they set it (`git apply -R` the v2 file, then apply v3).  Also measured since: at Qwen3.8-27B's recommended thinking-mode sampling (temperature 1.0, top-p 0.95, top-k 20), 10 recorded agent turns ran 8.7% faster with `0007` (one seed, on a later rdna-boosts base with our other patches).

**2026-10-24:** new `0007`: DFlash2 probabilistic drafting, **a trade-off, off by default** (`--spec-draft-sampling probabilistic`).  A sampled request's draft is sampled too and verified by rejection sampling: same output distribution, different tokens.  Code / math at temperature 0.7-1.0 got faster (+11% to +31% on r59, +5% to +17% on r115); low-temperature agent turns got slower (−3.9% at 0.3).  Measure on your own workload.  See the [report](reports/2026-10-24.md).

**2026-10-19:** `0001` K1 v3.3: the fp8 path's activation pass is cheaper (the largest channels chosen by a pairwise merge, and the per-channel maximum taken by the GLU kernels that write the activation).  The output is bit-identical to v3.2; prefill +2.1% / +3.3% / +2.5% at 8K / 32K / 64K (production setup).  No new switch.  See the [report](reports/2026-10-19.md).

**2026-10-18:** new `0006` (mmvk), for decode with speculative drafts: Qwen3.8-27B K-quant models take 4.5-6% less time per step (standard benchmark Q5 decode +6%), draft acceptance unchanged, KLD below the floor.  Off unless `GGML_CUDA_MMVK` is set; `2` turns it on only when llama-server verifies drafts.  See the [report](reports/2026-10-18.md).

**2026-10-15 (later):** new `0005`, a fix for `--backend-sampling`: llama.cpp re-reserved the compute graph on every request, freeing and allocating all compute buffers again.  Same output; see the [report](reports/2026-10-15.md#backend-sampling-fix-0005).

**2026-10-15:** support for rdna-boosts r51 and later (tested r59; on r59 the outputs equal r50's).  K1 v3.2 converts the weights to fp8 in one step: same output as v3, about 12% faster prefill.  New `0004`: the reasoning guard (off unless set).  When it fires, the model writes its answer a second time after the reasoning: leave room in `max_tokens`.

**2026-10-14:** `0002` F8 v3.3 fixes prompts prefilled together in one batch (`--parallel` ≥ 2): v3.1 and v3.2 zeroed the other sequences' attention in the F8 layers.  One sequence gives the same output as v3.2.  The `a55` file still has the bug: use `--parallel 1` with it.
