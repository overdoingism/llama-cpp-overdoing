# llama-cpp-overdoing

Extra patches for [llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) on AMD RDNA4 (gfx1201, Radeon AI PRO R9700),
for the changes that are **not** part of rdna-boosts: mostly lossy prefill speed-ups, a faster decode / draft-verification
matmul, a guard against replies that end inside the reasoning, an opt-in lenience for sampled drafts, faster wide draft
verification, and fixes of upstream llama.cpp code that we keep here.
Other bit-identical work goes to rdna-boosts itself.
Patches only; MIT.  Written with Claude (AI).

## Patches

File names: `<serial>-<upstream base>-<name>-v<version>.patch`.  The files in `patches/` are for rdna-boosts
`v15-6050b467c-r156` (upstream base `6050b467c`); `patches/r144/` has what r144 to r153 need besides them
([Applicable versions](#applicable-versions)).  Older sets: `patches/r59/` for `v12-097cf33d9-r51` to `r59`, `patches/r50/` for
`v16-097cf33d9-r50`, `patches/a55/` for `v16-a55e952b8-rN`.

| patch | what | switch | type |
|---|---|---|---|
| `0001-605-k1-v4` | K-quant prefill matmuls via hipBLASLt | `GGML_LF_KQ_BLAS`<br>(default on, fp8) | speed/<br>lossy |
| `0002-605-f8-v5` | fp8 prefill attention | `GGML_LF_FA_FP8`<br>(default on) | speed/<br>lossy |
| `0003-605-m4-v3` | MXFP4 weights to fp8 for `0001` | `GGML_LF_MXFP4_DIRECT`<br>(default on) | speed/<br>lossy |
| `0004-605-reasoning-guard-v2` | no reply left inside the reasoning | `LLAMA_REASONING_GUARD`<br>(default on) | guard |
| `0005-605-backend-sampling-reserve-v1` | no buffer reallocation per request | —<br>(always on) | fix |
| `0006-605-mmvk-v3` | faster 1-8 token K-quant matmuls | `GGML_CUDA_MMVK`<br>(default on) | speed/<br>diff |
| `0007-605-spec-lenience-v1` | keep more sampled-draft tokens | `LLAMA_SPEC_LENIENCE`<br>(default off) | speed/<br>lossy |
| `0009-605-fa-band-split-v1` | faster 9-16 row draft verification | `GGML_CUDA_FA_BAND_SPLIT`<br>(default on) | speed/<br>diff |
| `0012-605-dflash-adaptive-width-v1` | DFlash2 verify width from the acceptance | `LLAMA_DFLASH_ADAPTIVE`<br>(default on) | speed |
| `0014-605-ngram-shadow-v1` | n-gram drafts only while the text is being copied | `LLAMA_SPEC_NGRAM_SHADOW`<br>(default on, with `ngram-mod`) | speed/<br>diff |
| `0015-605-spec-align-v2` | slots verifying together send drafts of one length | `LLAMA_SPEC_ALIGN`,<br>`LLAMA_SPEC_ALIGN_CUT`<br>(default on) | speed |

<sub>0010, 0011, 0013 (Temporarily)</sub>

What each patch does, what it needs, the types, and the change log: [PATCHES.md](PATCHES.md).

All are on by default except `0007`; `GGML_LF_ALL=0` turns off every switch left unset ([SWITCHES.md](SWITCHES.md)).
Prefill speed-ups measured on Linux, R9700, Qwen3.8-27B UD-Q4_K_XL, DFlash2 draft, 8K / 32K prompt, on `v16-a55e952b8-r8` (the
patch code is the same since); KLD measured again on r144 ([2026-10-27 report](reports/2026-10-27.md)):

| | prefill | KLD vs official UD-Q5_K_M on r144 (zh / en) |
|---|---|---|
| reference (no patch) | — | 0.0018 / 0.0006 (run-to-run floor) |
| K1 f16 | +14% / +15% | 0.0018 / 0.0005 (≈ floor) |
| K1 fp8 (the default) | +16% / +22% on top of f16 | 0.0026 / 0.0007 |
| F8 (v5) | +3% / +17% | 0.0018 / 0.0005 |
| M4 (MXFP4 models only) | +55% at 8K, +44% at 110K | +0.002 / +0.0006 over the MXFP4 weights themselves |
| mmvk (`0006`) | — (decode: 4.5-6% less time per step with drafts) | 0.0015 / 0.0003 (below the floor) |
| **all defaults** | | **0.0025 / 0.0009** |

K1, F8 and M4 leave decode unchanged; `0006` changes decode only (for its KLD the decode path runs it at one token, `GGML_CUDA_MMVK=4`).  `fp8` also raises the perplexity of the prompt's own logits (every position scored on the prefill path) by
2-3% for Chinese text and 0.1-0.3% for English; generation reads the prompt through the KV cache and sees far less (the KLD column).
For prompt-scoring workloads set `GGML_LF_KQ_BLAS=f16`.  Details: [2026-10-12 report](reports/2026-10-12.md).

F8 v5 counts the row sum from the fp8 probabilities themselves: on a 170K-token reasoning conversation F8's own KLD drops by
a fifth (0.000396 to 0.000307, floor 0.00021); with all defaults the change is below the floor, at the same prefill speed
([report](reports/2026-10-27.md#0002-v5-the-row-sum-of-f8)).  On the table's 48K test v5 is within the floor of v4 (F8 alone
0.0019 / 0.0007 with v4; all defaults 0.0024 / 0.0008 with v4).

`0009` changes only draft-verification batches of 9-16 rows (two slots verifying at once, or a long n-gram draft), which
neither the KLD run nor the standard benchmark has; each of those rows then gets the result a batch of 8 rows or fewer gives.
Two slots decoding at once at 50K each: 13-19% slower than one after the other without it, 33-38% faster with it.

## Switches

Every switch, its values, and the server parameters it needs: [SWITCHES.md](SWITCHES.md).

On Linux we also set ROCm's own `HSA_USERPTR_FOR_PAGED_MEM=0`.  By default ROCm allocates pinned host memory
(`hipHostMalloc`, llama.cpp's `ROCm_Host` buffers) as ordinary pageable pages registered with the GPU; when the kernel swaps out
or moves one of them (memory pressure, compaction), the driver stops all GPU queues of the process until the page is
back (5-440 ms at a time, often several in a row).  With `0` those buffers are allocated by the driver instead and cannot be
swapped or moved.  Under the same memory pressure: 23 stalls (1.16 s in total over 19 requests) became none, with the
same output and speed ([report](reports/2026-10-27.md#linux-gpu-stalls-under-memory-pressure)).  The cost: those
buffers (about 0.7 GB with every layer on the GPU) stay in RAM; with large weights left in host memory (partial offload
without mmap) they count against the driver's limit.  Not needed on Windows.

## Applicable versions

Tested on rdna-boosts `v15-6050b467c-r156`: the patches in `patches/` apply in number order on the plain release, build, and
give the History numbers.  `0015` v2 needs r154 or later (it shares the release's own `LLAMA_SPEC_ALIGN`).  For r144 to r153,
apply the same files except `0015` (its `git apply --check` fails there), then `patches/r144/0015-605-spec-align-v1.patch`;
on Windows, r143 to r146 also need `patches/r144/0008-605-windows-moe-cache-close-v1.patch` (rdna-boosts has its fix since r147).
On another release, check with `git apply --check` first.  A release on a new upstream base (not `6050b467c`) needs a new
patch set.

## History

Speed on the standard benchmark (Qwen3.8-27B, 32K-token prompt, R9700, Linux), tokens/s.  `original`: plain rdna-boosts.
Each dated row: this repo's patches of that date on it, all on; the date links to the report (setup and KLD).

| date | build | Q5 prefill | Q5 decode | MXFP4 prefill | MXFP4 decode |
|---|---|---|---|---|---|
| original | `v16-a55e952b8-r8` | 983 | 47.4 | 1151 | 54.7 |
| [2026&#8209;10&#8209;07](reports/2026-10-07.md) | + K1&nbsp;v2, F8&nbsp;v3.1, M4&nbsp;v2 | **1958** | 47.5 | **2393** | **63.0** |
| [2026&#8209;10&#8209;12](reports/2026-10-12.md) | + K1&nbsp;v3, F8&nbsp;v3.2, M4&nbsp;v2 | **1954** | 49.0 | **2388** | **62.9** |
| original | `v16-097cf33d9-r50` | 1124 | 50.8 | 1344 | 60.8 |
| [2026&#8209;10&#8209;13](reports/2026-10-13.md) | + K1&nbsp;v3, F8&nbsp;v3.2, M4&nbsp;v2 | **1959** | 50.6 | **2397** | **66.2** |
| original | `v12-097cf33d9-r59` | 1125 | 51.1 | 1335 | 60.0 |
| [2026&#8209;10&#8209;15](reports/2026-10-15.md) | + K1&nbsp;v3.2, F8&nbsp;v3.3, M4&nbsp;v2 | **2197** | 50.4 | **2365** | **66.2** |
| [2026&#8209;10&#8209;18](reports/2026-10-18.md) | + K1&nbsp;v3.2, F8&nbsp;v3.3, M4&nbsp;v2, mmvk&nbsp;v2 | **2207** | **53.8** | **2389** | **67.2** |
| original | `v15-6050b467c-r144` | 1154 | 50.7 | 1381 | 63.5 |
| [2026&#8209;10&#8209;27](reports/2026-10-27.md) | + K1&nbsp;v4, F8&nbsp;v4, M4&nbsp;v3, mmvk&nbsp;v3 (the defaults) | **2293** | **56.1** | **2449** | **65.4** |
| original | `v15-6050b467c-r156` | 1155 | 52.8 | 1384 | 57.9 |
| [2026&#8209;10&#8209;28](PATCHES.md#change-log) | + K1&nbsp;v4, F8&nbsp;v5, M4&nbsp;v3, mmvk&nbsp;v3 (the defaults) | **2300** | **58.8** | **2467** | **61.0** |

Q5 = official UD-Q5_K_M; MXFP4 = FreedomAISVR MXFP4.  From r144 to r156 the time per decode step did not change (MXFP4:
43.35 / 43.21 ms with the patches); the decode figures moved with the draft acceptance, as the generated text changed.

## Build

- [Linux](BUILD-linux.md)
- [Windows](BUILD-windows.md)
- Then: [switches](SWITCHES.md)

## Thanks

To stew675 and the rdna-boosts contributors, and to the llama.cpp / ggml community.

## Note

Issues and pull requests may not get an answer here.  For problems, try a frontier cloud model first.

本 repo 不一定會回應 issue 與 PR。遇到問題，建議先請雲端前沿模型協助處理。
