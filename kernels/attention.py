"""Build, correctness-check, and benchmark the fused causal attention kernel.

    cmd /c "kernels\winbuild.bat -m kernels.attention"

Correctness gate: match F.scaled_dot_product_attention (the same FlashAttention
torch uses) on the model's real shape. Then benchmark vs SDPA (the bar) and vs
the naive materialized path (our model's pre-fusion attention).
"""

from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

from benchmarks.bench import benchmark

ext = load(
    name="attn_ext",
    sources=[str(Path(__file__).parent / "attn_ext.cu")],
    verbose=False,
)

B, nh, T, hs = 64, 6, 256, 64
scale = hs ** -0.5
q = torch.randn(B, nh, T, hs, device="cuda")
k = torch.randn(B, nh, T, hs, device="cuda")
v = torch.randn(B, nh, T, hs, device="cuda")

mine = ext.attn(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
max_err = (mine - ref).abs().max().item()
print(f"correctness: allclose={torch.allclose(mine, ref, atol=1e-3)}  max_err={max_err:.2e}")

mask = torch.tril(torch.ones(T, T, device="cuda")) == 0


def naive():
    w = (q @ k.transpose(-2, -1)) * scale
    w = w.masked_fill(mask, float("-inf"))
    w = F.softmax(w, dim=-1)
    return w @ v


t_mine = benchmark(lambda: ext.attn(q, k, v))
t_sdpa = benchmark(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
t_naive = benchmark(naive)
print(f"\nmine   {t_mine*1e3:7.3f} ms")
print(f"SDPA   {t_sdpa*1e3:7.3f} ms   (mine is {t_sdpa/t_mine:.2f}x of SDPA)")
print(f"naive  {t_naive*1e3:7.3f} ms   (mine is {t_naive/t_mine:.2f}x of naive)")
