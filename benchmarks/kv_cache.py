"""KV cache: correctness + when it actually helps.

Naive generate reprocesses the whole context every token (O(n^2) compute). The KV
cache caches each layer's K,V and computes only the new token (O(n)). But the win
only shows when generation is COMPUTE-bound. At small scale it's OVERHEAD-bound
(per-step Python + launch cost dominates), so the cache is ~par. Bigger model /
longer context -> compute dominates -> the cache wins. This measures both.

    python -m benchmarks.kv_cache
"""

import time

import torch

from model.gpt import GPT, GPTConfig

DEVICE = "cuda"


def time_gen(cfg, n):
    model = GPT(cfg).to(DEVICE).eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    model.generate(idx, 8, use_cache=False)   # warmup both paths
    model.generate(idx, 8, use_cache=True)

    def timed(use_cache):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model.generate(idx, n, use_cache=use_cache)
        torch.cuda.synchronize(); return time.perf_counter() - t0

    return timed(False), timed(True)


def correctness():
    cfg = GPTConfig(vocab_size=65, block_size=256)
    model = GPT(cfg).to(DEVICE).eval()
    torch.manual_seed(0)
    seq = torch.randint(cfg.vocab_size, (1, 64), device=DEVICE)
    with torch.no_grad():
        full, _ = model(seq)
        kv = model._empty_cache(1, DEVICE)
        inc = torch.stack([model(seq[:, t:t + 1], kv_caches=kv, cache_pos=t)[0][:, -1, :]
                           for t in range(seq.shape[1])], dim=1)
    print(f"correctness: allclose={torch.allclose(full, inc, atol=1e-4)}  "
          f"max_err={(full - inc).abs().max():.2e}\n")


if __name__ == "__main__":
    correctness()
    regimes = [
        ("real model  (384d,  6L, ctx 256)", GPTConfig(vocab_size=65, block_size=256), 200),
        ("bigger      (768d, 12L, ctx 512)",
         GPTConfig(vocab_size=65, block_size=512, n_embd=768, n_layer=12, n_head=12), 400),
    ]
    for name, cfg, n in regimes:
        tn, ty = time_gen(cfg, n)
        print(f"{name}  gen {n}:")
        print(f"    no cache  {tn*1e3:8.0f} ms  ({n/tn:6.0f} tok/s)")
        print(f"    KV cache  {ty*1e3:8.0f} ms  ({n/ty:6.0f} tok/s)   speedup {tn/ty:.2f}x")
