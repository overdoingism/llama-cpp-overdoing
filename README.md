# llama-cpp-overdoing

Extra patches for [llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) on AMD RDNA4 (gfx1201, Radeon AI PRO R9700),
for the changes that are **not** part of rdna-boosts: mostly lossy prefill speed-ups.  Bit-identical work goes to rdna-boosts itself.
Patches only; MIT.  Written with Claude (AI).

## Patches

File names: `<serial>-<upstream base>-<name>-v<version>.patch`.

| patch | what | turn on | lossy | needs |
|---|---|---|---|---|
| `0001-a55-k1-v2` | prefill matmuls of K-quant weights through hipBLASLt (f16, or fp8) | `GGML_LF_KQ_BLAS=f16` (or `fp8`); `GGML_LF_DRAFT_UB=512` shrinks the draft model's ubatch | yes | hipBLASLt |
| `0002-a55-f8-v2` | prefill flash attention in fp8 | `GGML_LF_FA_FP8=1` | yes | — |
| `0003-a55-m4-v2` | MXFP4 weights converted exactly to fp8 for K1's fp8 GEMM | `GGML_LF_MXFP4_DIRECT=blaslt` | yes | 0001 |

All are off unless the variable is set.  Measured on Linux, R9700, Qwen3.8-27B UD-Q4_K_XL, DFlash2 draft, 8K / 32K prompt:

| | prefill | KLD vs official UD-Q5_K_M (de / zh / en) |
|---|---|---|
| reference (no patch) | — | 0.0011 / 0.0018 / 0.0006 (run-to-run floor) |
| K1 f16 | +14% / +15% | ≈ floor |
| K1 fp8 (on top of f16) | +16% / +22% | 0.0017 / 0.0027 / 0.0010 |
| F8 | +3% / +17% | — |
| M4 (MXFP4 models only) | +55% at 8K, +44% at 110K | +≈0.001 over the MXFP4 weights themselves |

Decode is unchanged.

## Versions

- **Tested:** rdna-boosts `v16-a55e952b8-r8`: these three apply and build on the plain release; the numbers above come from a build
  that also had two bit-identical patches since sent to rdna-boosts.
- **Apply cleanly:** `v16-a55e952b8-r10`, `v16-a55e952b8-r15` (not built or run).  Other `v16-a55e952b8-rN` releases share the base and
  should work: check with `git apply --check` first.  A new upstream base (a different `v16-<hash>`) needs a new patch set.

## Build

- [Linux](BUILD-linux.md)
- [Windows](BUILD-windows.md)

## Thanks

To stew675 and the rdna-boosts contributors, and to the llama.cpp / ggml community.

## Note

Issues and pull requests may not get an answer here.  For problems, try a frontier cloud model first.

本 repo 不一定會回應 issue 與 PR。遇到問題，建議先請雲端前沿模型協助處理。
