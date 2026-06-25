# Custom kernel results

Median of 50 runs after warmup, RTX 2060 SUPER, fp32. Each kernel is correctness-
checked (`torch.allclose`, max_err shown) *before* timing. Built via
`cmd /c "kernels\winbuild.bat -m kernels.<module>"`.

## Softmax (`kernels/softmax_ext.cu`)

Row-wise softmax over the last dim. Two designs, benchmarked against `F.softmax`:

- **v1** — one block per row, shared-memory tree reduction.
- **v2** — one warp (32 threads) per row, warp-shuffle reductions, 8 rows/block.

- **adaptive** — dispatch by shape at runtime: warp kernel if `cols <= 512`, else block kernel.

| shape (rows × cols) | v1 (block) | v2 (warp) | adaptive | vs torch (adaptive) |
|---|---|---|---|---|
| 98304 × 256 (**attention shape**) | 0.19× | 1.00× | warp | **1.10×** |
| 4096 × 1024 | 0.77× | 0.64× | block | 0.74× |
| 4096 × 4096 | 0.89× | 0.63× | block | 0.83× |

**Finding: optimal strategy is shape-dependent.** Warp-per-row wins on short-and-many
rows (few columns → 32 lanes is enough parallelism, shuffle reductions have ~zero
overhead vs. shared-mem + `__syncthreads`). It loses on wide rows, where capping at 32
threads/row serializes too much — there the block-per-row design's higher per-row
parallelism wins. The launcher dispatches by shape (the same idea torch/cuBLAS use:
a zoo of specialized kernels + a heuristic dispatcher), so we get the best of both.

The 512 threshold is a hand-picked midpoint between the measured crossover (warp wins
at 256, block at 1024); pinning it exactly, and per-GPU, is what autotuning does via a
sweep. Irrelevant for our fixed workload (T=256, firmly in the warp regime).

**Honest takeaway:** we *match* torch's tuned softmax on the target shape — expected
(it's already well-fused). The real value is the warp-shuffle + stable-softmax craft,
which transfers directly to the fused-attention kernel, where fusion lets a custom
kernel do what calling separate torch ops cannot.

## Fused causal attention (`kernels/attn_ext.cu`)

`softmax(causal(QKᵀ/√hs)) V` without materializing the (T,T) score matrix, via the
FlashAttention online-softmax recurrence (running max + running sum + rescaled
accumulator). Shape B=64, nh=6, T=256, hs=64. Correctness vs `F.scaled_dot_product_attention`.

Reference bars: SDPA (torch FlashAttention) ~1.1 ms, naive materialized (cuBLAS bmm) ~4.6 ms.
All versions correct (max_err ~1e-6 vs SDPA).

| version | design | time | vs v1 | bottleneck found |
|---|---|---|---|---|
| **v1** | one thread per query row | **7.78 ms** | — | acc[64] spills to local mem; but ~100% occupancy |
| v2 | warp/row, dims split across lanes | 8.96 ms | 0.87× | warp reduction *per key* (dot too small to split) |
| v3 | warp/row, lane=key, 1 warp/block | 30.4 ms | 0.26× | ~6% occupancy (16 KB shared starves the SM) |
| v4 | v3 + 8 queries/block sharing K/V tile | 11.3 ms | 0.69× | 32-way shared-memory bank conflict in the dot |
| v5 | v4 + K stored transposed (conflict-free) | 8.36 ms | 0.93× | ~50% occupancy + per-tile barriers; still trails v1 |

**Conclusion (honest).** The simplest kernel, v1, stays fastest — its 100% occupancy and
embarrassingly-parallel simplicity beat the sophisticated tiled versions even after fixing
their reduction overhead (v2), occupancy collapse (v3→v4), and bank conflicts (v4→v5). And
**every hand-written version loses to cuBLAS (naive, 4.6 ms) and SDPA (1.1 ms).** At this
model's scale (T=256, hs=64) beating those by hand is not realistic — they embody years of
tuning. The deliverable here is the *journey*: a correct fused kernel built 5 ways, with each
version's slowdown profiled and root-caused (reduction cost → occupancy → bank conflicts →
barriers) — the real bottleneck hierarchy of GPU optimization. The production-grade win comes
from using the right tool (swap the model's naive attention for SDPA) rather than out-coding NVIDIA.

**v1 is correct but slow** — slower than both SDPA (FlashAttention) and the naive
materialized path (which leans on cuBLAS bmm). Causes, in order of impact:
1. one thread per row → no intra-row parallelism; each thread serially walks up to
   T keys doing 64-dim dot products.
2. `acc[64]` per thread spills to local memory → heavy local-memory traffic.
3. no shared-memory tiling → K/V reread from global memory every row.
4. causal load imbalance across warps (row 0 does 1 key, row 255 does 256).

Optimization runway (real FlashAttention): warp/block per query tile, shared-memory
K/V tiles reused across queries, accumulator in registers across warp lanes,
coalesced loads. Matching SDPA is the hard part; v1 establishes correctness first.

**Honest takeaway:** we *match*, not beat, torch's tuned softmax on the target shape —
expected (it's already a good fused kernel). The real value here is the warp-shuffle +
stable-softmax craft, which transfers directly to the fused-attention kernel, where
fusion lets a custom kernel do what calling separate torch ops cannot.
