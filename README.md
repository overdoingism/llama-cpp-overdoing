# llama-cpp-overdoing

Extra patches for [llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) on AMD RDNA4 (gfx1201, Radeon AI PRO R9700),
for the changes that are **not** part of rdna-boosts: mostly lossy prefill speed-ups.  Bit-identical work goes to rdna-boosts itself.
Patches only; MIT.  Written with Claude (AI).

## Patches

File names: `<serial>-<upstream base>-<name>-v<version>.patch`.  Current base: `097` (rdna-boosts `v16-097cf33d9-rN`); the `a55`
files are the same patches for the previous base (`v16-a55e952b8-rN`).

| patch | what | turn on | type | needs |
|---|---|---|---|---|
| `0001-097-k1-v3` | prefill matmuls of K-quant weights through hipBLASLt (f16, or fp8; fp8 corrects the largest activation channels) | `GGML_LF_KQ_BLAS=f16` (or `fp8`); `GGML_LF_DRAFT_UB=512` shrinks the draft model's ubatch | lossy | hipBLASLt |
| `0002-097-f8-v3.2` | prefill flash attention in fp8 (and no growing F16 staging copy) | `GGML_LF_FA_FP8=1` | lossy | — |
| `0003-097-m4-v2` | MXFP4 weights converted exactly to fp8 for K1's fp8 GEMM | `GGML_LF_MXFP4_DIRECT=blaslt` | lossy | 0001 |

All are off unless the variable is set.  Measured on Linux, R9700, Qwen3.8-27B UD-Q4_K_XL, DFlash2 draft, 8K / 32K prompt, on
`v16-a55e952b8-r8` (the patch code is the same on r50; speeds on r50 are in History, KLD was not measured again):

| | prefill | KLD vs official UD-Q5_K_M (zh / en) |
|---|---|---|
| reference (no patch) | — | 0.0018 / 0.0006 (run-to-run floor) |
| K1 f16 | +14% / +15% | ≈ floor |
| K1 fp8 (on top of f16) | +16% / +22% | 0.0023 / 0.0008 |
| F8 | +3% / +17% | 0.0020 / 0.0007 |
| M4 (MXFP4 models only) | +55% at 8K, +44% at 110K | +≈0.001 over the MXFP4 weights themselves |

Decode is unchanged.  `fp8` also raises the perplexity of the prompt's own logits (every position scored on the prefill path) by
2-3% for Chinese text and 0.1-0.3% for English; generation reads the prompt through the KV cache and sees far less (the KLD column).
For prompt-scoring workloads use `f16`.  Details: [2026-10-12 report](reports/2026-10-12.md).

## Switches

Every switch, its values, and the server parameters it needs: [SWITCHES.md](SWITCHES.md).

## Applicable versions

- **`097` files, tested:** rdna-boosts `v16-097cf33d9-r50`.  They apply in number order on the plain release, build, and give the
  History numbers.  Other `v16-097cf33d9-rN` releases share the base and should work: check with `git apply --check` first.
- **`a55` files:** `v16-a55e952b8-r8` tested (the Patches table and the first History rows; that build also had bit-identical
  patches since sent to rdna-boosts).  On r45 and later, `0001` has one conflicting hunk in
  `ggml/src/ggml-cuda/ggml-cuda.cu` (start of graph compute: keep both rdna-boosts' `bf16_src1_cache_clear()` line and K1's
  `graph_begin` line).
- A new upstream base (a different `v16-<hash>`) needs a new patch set.

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
