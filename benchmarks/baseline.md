# Pre-optimization baseline

Numbers from `python -m benchmarks.bench`, RTX 2060 SUPER (Turing sm_75), torch 2.12.0+cu126.
Model: 10.8M params (6 layer, 6 head, n_embd 384, block_size 256). All timings are
median of 50 runs after warmup, GPU-synchronized.

## Model level (B=64, T=256 → 16,384 tokens/step)

| Op | Time | Throughput |
|---|---|---|
| forward | 118.5 ms | 138,270 tok/s |
| train step (fwd+bwd+opt) | 361.8 ms | 45,283 tok/s |
| generate (no KV cache) | 3,711 ms / 200 tok | 54 tok/s |

Note: the 45K tok/s train-step figure supersedes the ~15K reported by the training
loop — that number was deflated because the loop's timer included the periodic
`estimate_loss` eval. This is exactly why a dedicated harness exists.

## Op level (at model shapes: B=64, T=256, C=384, heads=6)

| Op | Time | Notes |
|---|---|---|
| matmul, FFN up-proj (B·T, C)·(C, 4C) | 5.28 ms | 3,658 GFLOP/s (~50% of card FP32 peak) |
| softmax (B, nh, T, T) | 0.90 ms | |
| attention, naive materialized | 6.09 ms | our model's math (and the model is even slower — see below) |
| attention, fused (torch SDPA) | 1.67 ms | **3.6× faster than naive** |

## Phase 3 targets, ranked

1. **Generation: 54 tok/s** is the most dramatic gap (vs 138K tok/s of raw forward
   throughput). Cause: every generated token reprocesses the whole growing context.
   Fix: **KV cache**. Biggest user-visible win.
2. **Attention: 3.6× headroom** (naive 6.09 → fused 1.67 ms). Fix: a hand-written
   **fused causal attention** kernel (mini-FlashAttention). The capstone.
3. **Free PyTorch-level win first:** the model runs attention as a Python loop over 6
   separate `Head` modules (6 small matmuls). Batching the heads into one matmul
   (nanoGPT-style) should help before any CUDA is written.
4. **matmul** is already at ~50% of peak via cuBLAS. Beating cuBLAS is very hard; the
   matmul kernel ladder (naive → tiled → vectorized) is a *learning* exercise that
   climbs toward it, not a realistic win. Be honest about this in results.
