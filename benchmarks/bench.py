"""Benchmark + timing harness for TransformerOp.

Honest GPU timing: warmup first, then torch.cuda.synchronize() around every
measurement (GPU calls are async -- without sync you time the *launch*, not the
*work*), median over many runs. Everything in Phase 3 reuses benchmark().

Usage:
    python -m benchmarks.bench
    python -m benchmarks.bench --batch 32     # if batch 64 OOMs
"""

import argparse
import statistics
import time

import torch
from torch.nn import functional as F

from model.gpt import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def benchmark(fn, warmup=10, iters=50):
    """Median wall-clock seconds per call of fn(), GPU-synchronized."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def fmt(secs):
    return f"{secs * 1e3:8.3f} ms"


def bench_model(cfg, batch):
    model = GPT(cfg).to(DEVICE)
    x = torch.randint(cfg.vocab_size, (batch, cfg.block_size), device=DEVICE)
    y = torch.randint(cfg.vocab_size, (batch, cfg.block_size), device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tokens = batch * cfg.block_size
    params = sum(p.numel() for p in model.parameters())

    def fwd():
        with torch.no_grad():
            model(x, y)

    def step():
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    t_fwd = benchmark(fwd)
    model.train()
    t_step = benchmark(step)

    print(f"\n=== full model (B={batch}, T={cfg.block_size}, {params/1e6:.1f}M params) ===")
    print(f"forward       {fmt(t_fwd)}   {tokens / t_fwd:>12,.0f} tok/s")
    print(f"train step    {fmt(t_step)}   {tokens / t_step:>12,.0f} tok/s")
    return model


def bench_generate(model, cfg):
    idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    n = 200
    t = benchmark(lambda: model.generate(idx, n), warmup=2, iters=5)
    print(f"generate      {fmt(t)}   {n / t:>12,.0f} tok/s  (sequential, no KV cache)")


def bench_ops(cfg, batch):
    B, T, C, nh = batch, cfg.block_size, cfg.n_embd, cfg.n_head
    hs = C // nh
    print(f"\n=== ops at model shapes (B={B}, T={T}, C={C}, heads={nh}) ===")

    # the FFN's big up-projection matmul: (B*T, C) @ (C, 4C)
    a = torch.randn(B * T, C, device=DEVICE)
    w = torch.randn(C, 4 * C, device=DEVICE)
    t_mm = benchmark(lambda: a @ w)
    flops = 2 * (B * T) * C * (4 * C)
    print(f"matmul (FFN up-proj)         {fmt(t_mm)}   {flops / t_mm / 1e9:>8,.0f} GFLOP/s")

    # softmax over the attention rows
    s = torch.randn(B, nh, T, T, device=DEVICE)
    t_sm = benchmark(lambda: F.softmax(s, dim=-1))
    print(f"softmax (B,nh,T,T)           {fmt(t_sm)}")

    # attention: our naive materialized path vs torch's fused kernel
    q = torch.randn(B, nh, T, hs, device=DEVICE)
    k = torch.randn(B, nh, T, hs, device=DEVICE)
    v = torch.randn(B, nh, T, hs, device=DEVICE)
    mask = torch.tril(torch.ones(T, T, device=DEVICE)) == 0

    def naive_attn():
        wei = (q @ k.transpose(-2, -1)) * hs ** -0.5
        wei = wei.masked_fill(mask, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

    t_naive = benchmark(naive_attn)
    t_sdpa = benchmark(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    print(f"attention naive (our path)   {fmt(t_naive)}")
    print(f"attention fused (torch SDPA) {fmt(t_sdpa)}   {t_naive / t_sdpa:>6.1f}x faster")
    print("  ^ the Phase 3 prize: a hand-written fused kernel aims to close this gap.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()
    if DEVICE == "cpu":
        print("WARNING: CUDA unavailable; CPU timings are not meaningful for this project.")
    print(f"device={DEVICE} {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else ''}")
    cfg = GPTConfig(vocab_size=65, block_size=256)
    model = bench_model(cfg, args.batch)
    bench_generate(model, cfg)
    bench_ops(cfg, args.batch)


if __name__ == "__main__":
    main()
