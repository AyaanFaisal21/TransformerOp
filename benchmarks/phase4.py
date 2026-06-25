"""Phase 4: end-to-end effect of swapping the model's naive attention for SDPA.

The kernel study (benchmarks/kernels.md) showed hand-written attention can't beat
torch's fused SDPA. The production win is to *use* SDPA in the model. This measures
the full training-step speedup, and checks both paths produce the same output.

    python -m benchmarks.phase4
"""

import copy

import torch

from benchmarks.bench import benchmark
from model.gpt import GPT, GPTConfig

DEVICE = "cuda"
B, T = 64, 256


def train_step_time(use_sdpa):
    cfg = GPTConfig(vocab_size=65, block_size=T, use_sdpa=use_sdpa)
    model = GPT(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(cfg.vocab_size, (B, T), device=DEVICE)
    y = torch.randint(cfg.vocab_size, (B, T), device=DEVICE)

    def step():
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return benchmark(step)


def correctness():
    # same weights, both paths, eval mode (no dropout) -> outputs must match
    torch.manual_seed(0)
    naive = GPT(GPTConfig(vocab_size=65, block_size=T, use_sdpa=False)).to(DEVICE).eval()
    sdpa = GPT(GPTConfig(vocab_size=65, block_size=T, use_sdpa=True)).to(DEVICE).eval()
    sdpa.load_state_dict(naive.state_dict())
    idx = torch.randint(65, (4, 64), device=DEVICE)
    with torch.no_grad():
        a, _ = naive(idx)
        b, _ = sdpa(idx)
    print(f"paths agree: allclose={torch.allclose(a, b, atol=1e-4)}  max_err={(a-b).abs().max():.2e}")


if __name__ == "__main__":
    correctness()
    t_naive = train_step_time(False)
    t_sdpa = train_step_time(True)
    print(f"\ntrain step (B={B}, T={T}):")
    print(f"  naive attention  {t_naive*1e3:7.2f} ms   {B*T/t_naive:>10,.0f} tok/s")
    print(f"  SDPA attention   {t_sdpa*1e3:7.2f} ms   {B*T/t_sdpa:>10,.0f} tok/s")
    print(f"  speedup: {t_naive/t_sdpa:.2f}x")
