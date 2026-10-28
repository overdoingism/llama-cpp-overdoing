# Patches in detail

What each patch in [`patches/`](patches) does, its switch (with the default under it), and what it needs; then the change log.
[`patches/r144/`](patches/r144) holds `0015` v1 (for r144 to r153); on Windows use r147 or later (r143 to r146 do not compile
there, rdna-boosts issue #231).  The short table is in
the [README](README.md#patches).  Written with Claude (AI).

| patch | what | switch | type | needs |
|---|---|---|---|---|
| `0001-605-k1-v6` | prefill matmuls of K-quant weights through hipBLASLt in fp8 (or f16); fp8 corrects the largest activation channels.  v6 = v4 also taking batches of 384..1023 tokens on weights of 20M elements or more (a prompt's last batch, short prompts, the draft's batches), and converting Q4_K / Q5_K / IQ4_XS weights without the LDS round trip (same bytes, faster) | `GGML_LF_KQ_BLAS`<br>(default `fp8` when every GPU is a gfx1201; `0` = off);<br>`GGML_LF_DRAFT_UB`<br>(default `512`) shrinks the draft model's ubatch | speed/lossy | hipBLASLt |
| `0002-605-f8-v5` | prefill flash attention in fp8 (and no growing F16 staging copy).  v5 = v4 with the row sum counted from the fp8 probabilities (numerator and denominator from the same numbers); v4 = v3.3, on by default | `GGML_LF_FA_FP8`<br>(default on; `0` = off) | speed/lossy | — |
| `0003-605-m4-v3` | MXFP4 weights converted exactly to fp8 for K1's fp8 GEMM.  v3 = v2, on by default | `GGML_LF_MXFP4_DIRECT`<br>(default `blaslt`; `0` = off) | speed/lossy | 0001 |
| `0004-605-reasoning-guard-v2` | an end of turn sampled inside the reasoning block becomes the block's end tag, so the model writes its answer instead of ending with the answer left in the reasoning and an empty reply (seen with clients that send earlier reasoning back as plain text, over long contexts).  v2 = v1, on by default | `LLAMA_REASONING_GUARD`<br>(default on; `0` = off) | guard | — |
| `0005-605-backend-sampling-reserve-v1` | with `--backend-sampling`, the compute buffers are reserved once per sequence instead of being freed and allocated again for every request (upstream llama.cpp behavior; about 0.25 s per request on a 27B model at `-ub 2048`) | —<br>(always on) | fix | — |
| `0006-605-mmvk-v3` | Q4_K / Q5_K / Q6_K matmuls of 1 to 8 tokens (decode and speculative verification) keep the activations in registers across several weight rows, and the matmuls that read the same input go out as one launch; not bit-identical to the stock kernel (float summation order), KLD below the run-to-run floor.  Only when llama-server verifies drafts (draft model, MTP or n-gram): one-token decode alone is faster on the stock kernel.  v3 = v2, on by default | `GGML_CUDA_MMVK`<br>(default `2`; `0` = off, `4` = also without drafts) | speed/diff | 0001, 0003 (context) |
| `0007-605-spec-lenience-v1` | with sampled drafts (llama.cpp's own `--spec-draft-sampling probabilistic`, which covers DFlash2 drafts from this base on) a drafted token is accepted with probability min(1, p / (l·q)) instead of min(1, p / q): more drafted tokens kept, and no token more than 1/l times as likely as under the model.  Only at a request temperature above 0 | `LLAMA_SPEC_LENIENCE` or the request field `spec_lenience`<br>(default off; e.g. `0.8663`) | speed/lossy | 0004 (context) |
| `0008-605-glu-inplace-v1` | the graph allocator may give a GLU (SwiGLU, GEGLU, ...) the memory of its gate when nothing else reads the gate, so an FFN's gate, up and GLU output are no longer all live at once; the K1 GLU kernel of `0001` drops its `restrict` qualifiers for that.  At `-ub 2048` the compute buffer is 150 MiB smaller on Qwen3.8-27B (one 136 MiB tensor plus the HIP margin) and 70 MiB on Qwen3.6-35B-A3B.  Same output: greedy text identical with it on and off on Linux and Windows (Qwen3.8-27B Q5_K and UD-Q4_K_XL, Qwen3.6-35B-A3B; K1 on and off; DFlash2, MTP and no draft; 1, 2 and 4 slots; `-ub 512` and `2048`; the CPU backend); prefill speed within ±0.4 % of the build without it | `GGML_GLU_INPLACE`<br>(default on; `0` = off) | memory | 0001 |
| `0009-605-fa-band-split-v1` | on RDNA4, a draft-verification batch of 9 to 16 query rows (two slots of a unified KV cache verifying at once, or an n-gram draft longer than 7) runs its attention as two halves of at most 8 rows through the decode / verify kernel instead of the generic one, which re-reads the KV cache several times: at 110K cells such a step takes 30-34% less time, and two slots decoding at once at 50K each went from 13-19% below one slot alone to 33-38% above it.  Only from 8192 KV cells (below, two launches cost more than they save).  Each row gets the same result as in a batch of 8 rows or fewer; not bit-identical to the stock generic kernel | `GGML_CUDA_FA_BAND_SPLIT`<br>(default on; `0` = off);<br>`GGML_CUDA_FA_BAND_SPLIT_MIN_KV`<br>(default `8192`) | speed/diff | 0002 (context) |
| `0012-605-dflash-adaptive-width-v1` | DFlash2 drafts w tokens per sequence, w following running per-position acceptance estimates to the width with the most tokens per unit of step cost (0.035 of a step per verified row, plus the RDNA4 band's tile step from 5 rows at long context).  The output does not change.  One request at a time: general workloads +0.1 %, low-acceptance long text +5..+11 % | `LLAMA_DFLASH_ADAPTIVE`<br>(default on; `0` = fixed width, `2` = also log the widths) | speed | — |
| `0013-605-gdn-verify-anchor-v3` | Gated DeltaNet draft verification on qwen35 / qwen35moe models.  v1 (not published): the verify kernel reads the state and writes the per-token snapshots with non-temporal hints, the same values (0.5-0.6% less time per step).  v2 adds the anchored rollback (opt-in in v2; v3 = v2 with it on by default): each cell keeps two state planes (after its last batch, and the one that batch started from) and the batch's per-token replay factors instead of 1 + n_rs_seq full planes, and a partly accepted draft is replayed from the start state when the cell is next read.  Same output; Qwen3.8-27B: 1.03 GiB less VRAM with 2 slots (0.51 with 1), 0.3-0.4% less time per step, two slots at once +2-4%.  CPU and CUDA / ROCm only (a recurrent layer elsewhere turns it off; a warning when `LLAMA_RS_ANCHOR` is set, an info line when not).  The mechanism follows radiance's GDN verify (Apache-2.0); the code is new.  And a fix, always on: from 3 slots on, a cell moved out of the way of a later ubatch of the same decode kept only plane 0 of its state, so a rollback of that sequence read another sequence's snapshot (upstream llama.cpp too; with the anchor the rollback was refused and the server aborted).  Moved cells now keep their snapshots; with 1 or 2 slots no cell moves this way | `LLAMA_RS_ANCHOR`<br>(default on; `0` = off);<br>`GGML_CUDA_GDN_NT`<br>(default on; `0` = off) | speed, fix | — |
| `0014-605-ngram-shadow-v1` | with `--spec-type ngram-mod,draft-dflash`, the n-gram drafts are held back and compared with what the model writes; after 24 correct tokens in a row the sequence is copying and gets n-gram drafts (up to 15 tokens) without DFlash2, until one misses.  Whole-file rewrites +38..+68 %, other requests within ±1 %.  Not bit-identical while copying (16-row verification), unless `LLAMA_SPEC_NGRAM_SHADOW_EXACT=1` | `LLAMA_SPEC_NGRAM_SHADOW`<br>(default `24`; `0` = n-gram first, as upstream) | speed/diff | an n-gram type before a model drafter in `--spec-type` |
| `0015-605-spec-align-v2` | on hybrid / recurrent models (Qwen3.x) a verify batch whose sequences differ in length is split and runs the weights twice; slots verifying together now send drafts of one length (0012 widens the narrower DFlash2 drafts, a copying n-gram draft is cut to the others' length).  Two slots at once: 0012 no longer loses 16-23 %, and a slot next to a copying one no longer loses 47 %.  v2 = v1 for r154 and later: it reads the release's own `LLAMA_SPEC_ALIGN` (rdna-boosts aligns `draft-mtp-adaptive` with it), and the cut to the shortest draft has its own switch | `LLAMA_SPEC_ALIGN`<br>(default on; `0` = off, ours and the release's);<br>`LLAMA_SPEC_ALIGN_CUT`<br>(default on; `0` = no cut, as the release) | speed | 0012, 0014 (context); r154 or later |

Type: `memory` = less VRAM, same output; `speed/lossy` = faster, with error above the run-to-run floor (or, for 0007, a bounded change of the output distribution); `speed/diff` = faster, output not bit-identical to the stock code but within the floor; `guard` = changes behavior only in a failure case; `fix` = a bug fix; `speed/trade-off` = faster on some workloads and slower on others, so off unless you ask for it.

## Change log

**2026-10-28 (night, later):** new `0008` (the number of the withdrawn Windows fix, reused): a GLU may take its gate's
memory, 150 MiB less compute buffer at `-ub 2048` on Qwen3.8-27B, same output.  It needs `0001` (it edits K1's GLU kernel).

**2026-10-28 (night):** `0008` withdrawn (the Windows build fix for r143 to r146): on Windows use r147 or later.

**2026-10-28 (evening):** `0013` v3: the anchored rollback is on by default (`LLAMA_RS_ANCHOR=0` turns it off; left
unset, `GGML_LF_ALL=0` turns it off too); otherwise the same as v2.  v2 users can apply
[`patches/increments/0013-605-gdn-verify-anchor-v2-to-v3.patch`](patches/increments/0013-605-gdn-verify-anchor-v2-to-v3.patch).

**2026-10-28 (later):** new `0013` v2 (v1 was not published): non-temporal hints in the GDN verify kernel, the opt-in
anchored rollback (`LLAMA_RS_ANCHOR=1`: same output, about 1 GiB less VRAM with 2 slots), and a fix for 3 or more slots (a
moved cell kept only plane 0 of its rollback snapshots).  `GGML_LF_ALL=0` leaves the fix on: with 3 or more slots the
output can then differ from the plain release, where the release reads another sequence's snapshot.

**2026-10-28:** for rdna-boosts r156: `0015` v2 shares r154's own `LLAMA_SPEC_ALIGN`, and the cut to the shortest draft
gets its own switch, `LLAMA_SPEC_ALIGN_CUT` (a copying slot next to a Chinese-text slot, both at once: 109.4 t/s with the
cut, 102.9 without); the other files are unchanged.  `0015` v1 and `0008` moved to [`patches/r144/`](patches/r144) for
r144 to r153.  On r156 `GGML_LF_ALL=0` gives the same output as the plain release.

**2026-10-27:** a new set for rdna-boosts `v15-6050b467c-r144` (upstream base `6050b467c`); the r59 set moved to
[`patches/r59/`](patches/r59).  **The patches are now on by default** (`0` turns one off, `GGML_LF_ALL=0` every switch left
unset; [SWITCHES.md](SWITCHES.md)).  New: `0007`, the lenience of sampled-draft verification (llama.cpp now samples DFlash2
drafts itself, which the old 0007 added; lossy, off unless `LLAMA_SPEC_LENIENCE` is set), `0008` for the Windows build,
`0009` (9-16 row verification through the decode / verify kernel), `0012` (DFlash2 adaptive verify width), `0014` (shadow
n-gram) and `0015` (one draft length per verify batch).  `0002` v5 counts F8's row sum from the fp8 probabilities: F8's own
KLD on a 170K-token conversation 0.000396 to 0.000307 (floor 0.00021), same speed; v4 users can apply
[`patches/increments/0002-605-f8-v4-to-v5.patch`](patches/increments/0002-605-f8-v4-to-v5.patch).  In `patches/r59/`,
`0007` v3 fixes `--spec-draft-p-min` above 0 (v2 skewed the output distribution).  See the [report](reports/2026-10-27.md).

**2026-10-24:** new `0007`: DFlash2 probabilistic drafting, **a trade-off, off by default** (`--spec-draft-sampling probabilistic`).  A sampled request's draft is sampled too and verified by rejection sampling: same output distribution, different tokens.  Code / math at temperature 0.7-1.0 got faster (+11% to +31% on r59, +5% to +17% on r115); low-temperature agent turns got slower (−3.9% at 0.3).  Measure on your own workload.  See the [report](reports/2026-10-24.md).

**2026-10-19:** `0001` K1 v3.3: the fp8 path's activation pass is cheaper (the largest channels chosen by a pairwise merge, and the per-channel maximum taken by the GLU kernels that write the activation).  The output is bit-identical to v3.2; prefill +2.1% / +3.3% / +2.5% at 8K / 32K / 64K (production setup).  No new switch.  See the [report](reports/2026-10-19.md).

**2026-10-18:** new `0006` (mmvk), for decode with speculative drafts: Qwen3.8-27B K-quant models take 4.5-6% less time per step (standard benchmark Q5 decode +6%), draft acceptance unchanged, KLD below the floor.  Off unless `GGML_CUDA_MMVK` is set; `2` turns it on only when llama-server verifies drafts.  See the [report](reports/2026-10-18.md).

**2026-10-15 (later):** new `0005`, a fix for `--backend-sampling`: llama.cpp re-reserved the compute graph on every request, freeing and allocating all compute buffers again.  Same output; see the [report](reports/2026-10-15.md#backend-sampling-fix-0005).

**2026-10-15:** support for rdna-boosts r51 and later (tested r59; on r59 the outputs equal r50's).  K1 v3.2 converts the weights to fp8 in one step: same output as v3, about 12% faster prefill.  New `0004`: the reasoning guard (off unless set).  When it fires, the model writes its answer a second time after the reasoning: leave room in `max_tokens`.

**2026-10-14:** `0002` F8 v3.3 fixes prompts prefilled together in one batch (`--parallel` ≥ 2): v3.1 and v3.2 zeroed the other sequences' attention in the F8 layers.  One sequence gives the same output as v3.2.  The `a55` file still has the bug: use `--parallel 1` with it.
