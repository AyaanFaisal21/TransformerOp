# Where a training step's time goes

Source: `python -m benchmarks.profile_step` (PyTorch profiler, 20 steps after warmup),
RTX 2060 SUPER, B=64, T=256, 10.8M-param model. Total GPU time 6.77 s / 20 steps
= ~338 ms/step (consistent with the 362 ms benchmark).

This document is the Phase 3 target list. Every kernel must point at a line here.

## Attribution (by GPU "self" time, operation level)

| Category | Op(s) | % of GPU time | What it is |
|---|---|---|---|
| **Linear matmuls** | `mm` + `addmm` | **~60%** | Q/K/V projections, FFN (2 per block), output proj, lm_head |
| **Attention matmuls** | `bmm` | ~7.5% | the batched Q·Kᵀ and weights·V |
| Elementwise add | `add_` | ~7.7% | residuals, bias adds, optimizer update |
| Memory copies | `copy_` + Memcpy DtoD | ~5.3% | per-head concat, transposes |
| Elementwise mul | `mul` | ~3.9% | the 1/√d attention scaling, etc. |
| Causal mask | `masked_fill_` | ~2.3% | blanking the upper triangle |
| Dropout | `native_dropout` | ~2.1% | |
| Remainder | softmax, LayerNorm, AdamW internals | ~11% | |

Underlying CUDA kernels for all the matmuls are cuBLAS `volta_sgemm_*` (the card is
Turing, which the profiler labels "volta").

## Headline finding

**~2/3 of training time is matrix multiplication** (`mm` + `addmm` + `bmm` ≈ 67%).
This model is matmul-bound. The earlier assumption "attention is the hot path" is
only partly true at this size — with T=256 the FFN/Linear matmuls dominate, and
attention's own matmuls are ~7.5%. (At larger T the T² attention term grows and
would eventually overtake them.)

## What this means for Phase 3 — honest target ranking

1. **The biggest category is the least beatable.** Those matmuls already run on
   cuBLAS at ~50% of card peak. A hand-written matmul will not beat cuBLAS — so the
   matmul-kernel ladder (naive → tiled → vectorized) stays a *learning* exercise,
   not a training-speed win. Do not expect to reclaim the 60%.

2. **Attention is fragmented — that's the real fusion win.** Attention is currently
   spread across `bmm` + `mul` + `masked_fill_` + softmax + dropout + several
   `copy_`/Memcpy, each a *separate kernel* that writes the big (B, heads, T, T)
   matrix to memory and reads it back. A fused attention kernel (mini-FlashAttention)
   collapses these into one pass that never materializes the T×T matrix. The win is
   **memory traffic, not FLOPs** — it targets the ~10-15% spread across those rows,
   plus matches what torch SDPA already does (the 3.6× from baseline.md).

3. **Free PyTorch-level win:** the model runs attention as a Python loop over 6
   separate `Head`s, inflating kernel-launch count and the `copy_`/concat cost.
   Batching the heads into one set of matmuls cuts launches and copies before any
   CUDA is written.

4. **Generation (KV cache) is not in this profile** — it's an inference-time problem
   (the 54 tok/s in baseline.md), separate from training-step cost. Still the biggest
   user-visible win, addressed independently.

## Bottom line

Training is dominated by cuBLAS matmuls we can't beat by hand. The hand-written-kernel
value is concentrated in **attention fusion** (memory-traffic reduction) and the
**KV cache** (generation), not in the raw matmul math. Phase 3 is scoped accordingly.
