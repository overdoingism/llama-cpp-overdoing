# llama-cpp-overdoing

Extra patches for [llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) on AMD RDNA4 (gfx1201, Radeon AI PRO R9700),
for the changes that are **not** part of rdna-boosts: mostly lossy prefill speed-ups, a faster decode / draft-verification
matmul, an opt-in guard against replies that end inside the reasoning, an opt-in sampled DFlash2 draft, and fixes of upstream
llama.cpp code that we keep here.
Other bit-identical work goes to rdna-boosts itself.
Patches only; MIT.  Written with Claude (AI).

## Patches

File names: `<serial>-<upstream base>-<name>-v<version>.patch`.  The files in `patches/` are for rdna-boosts
`v12-097cf33d9-r51` and later (tested on r59); `patches/r50/` holds the set for `v16-097cf33d9-r50`, and `patches/a55/` the set for
the previous base (`v16-a55e952b8-rN`).

| patch | what | switch | default | type |
|---|---|---|---|---|
| `0001-097-k1-v3.3` | K-quant prefill matmuls via hipBLASLt | `GGML_LF_KQ_BLAS` | off | speed/<br>lossy |
| `0002-097-f8-v3.3` | fp8 prefill attention | `GGML_LF_FA_FP8` | off | speed/<br>lossy |
| `0003-097-m4-v2` | MXFP4 weights to fp8 for `0001` | `GGML_LF_MXFP4_DIRECT` | off | speed/<br>lossy |
| `0004-097-reasoning-guard-v1` | no reply left inside the reasoning | `LLAMA_REASONING_GUARD` | off | guard |
| `0005-097-backend-sampling-reserve-v1` | no buffer reallocation per request | — | on | fix |
| `0006-097-mmvk-v2` | faster 1-8 token K-quant matmuls | `GGML_CUDA_MMVK` | off | speed/<br>diff |
| `0007-097-dflash2-probabilistic-v3` | sampled DFlash2 drafts | `--spec-draft-sampling` | off | speed/<br>trade-off |

What each patch does, what it needs, the types, and the change log: [PATCHES.md](PATCHES.md).

All are off unless the variable (for `0007`, the server parameter) is set.  Measured on Linux, R9700, Qwen3.8-27B UD-Q4_K_XL, DFlash2 draft, 8K / 32K prompt, on
`v16-a55e952b8-r8` (the patch code is the same on r50; speeds on r50 are in History, the quality check on r50 is in its report):

| | prefill | KLD vs official UD-Q5_K_M (zh / en) |
|---|---|---|
| reference (no patch) | — | 0.0018 / 0.0006 (run-to-run floor) |
| K1 f16 | +14% / +15% | ≈ floor |
| K1 fp8 (on top of f16) | +16% / +22% | 0.0023 / 0.0008 |
| F8 | +3% / +17% | 0.0020 / 0.0007 |
| M4 (MXFP4 models only) | +55% at 8K, +44% at 110K | +≈0.001 over the MXFP4 weights themselves |
| mmvk v2 (`0006`; measured on r59, against plain r59) | — (decode: 4.5-6% less time per step with drafts) | 0.0015 / 0.0003 (below the floor) |

K1, F8 and M4 leave decode unchanged; `0006` changes decode only (for its KLD the decode path runs it at one token, `GGML_CUDA_MMVK=4`).  `fp8` also raises the perplexity of the prompt's own logits (every position scored on the prefill path) by
2-3% for Chinese text and 0.1-0.3% for English; generation reads the prompt through the KV cache and sees far less (the KLD column).
For prompt-scoring workloads use `f16`.  Details: [2026-10-12 report](reports/2026-10-12.md).

## Switches

Every switch, its values, and the server parameters it needs: [SWITCHES.md](SWITCHES.md).

## Applicable versions

Tested on rdna-boosts `v12-097cf33d9-r59`: the patches in `patches/` apply in number order on the plain release, build, and give
the History numbers.  On another release, check with `git apply --check` first.  A release on a new upstream base (not
`097cf33d9`) needs a new patch set.

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

Q5 = official UD-Q5_K_M; MXFP4 = FreedomAISVR MXFP4.

## Build

- [Linux](BUILD-linux.md)
- [Windows](BUILD-windows.md)
- Then: [switches](SWITCHES.md)

## Thanks

To stew675 and the rdna-boosts contributors, and to the llama.cpp / ggml community.

## Note

Issues and pull requests may not get an answer here.  For problems, try a frontier cloud model first.

本 repo 不一定會回應 issue 與 PR。遇到問題，建議先請雲端前沿模型協助處理。
