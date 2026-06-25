"""Build, correctness-check, and benchmark the custom softmax kernel.

    cmd /c "kernels\winbuild.bat -m kernels.softmax"

Correctness FIRST (allclose vs F.softmax), then speed -- a fast wrong kernel is
worthless. Reuses benchmark() from benchmarks.bench (warmup + sync + median).
"""

from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

from benchmarks.bench import benchmark

ext = load(
    name="softmax_ext",
    sources=[str(Path(__file__).parent / "softmax_ext.cu")],
    verbose=False,
)


def check_and_bench(shape):
    x = torch.randn(*shape, device="cuda")
    mine = ext.softmax(x)
    ref = F.softmax(x, dim=-1)
    ok = torch.allclose(mine, ref, atol=1e-5)
    max_err = (mine - ref).abs().max().item()
    t_mine = benchmark(lambda: ext.softmax(x))
    t_ref = benchmark(lambda: F.softmax(x, dim=-1))
    print(f"{str(shape):>18}  allclose={str(ok):5}  max_err={max_err:.1e}  "
          f"mine {t_mine*1e3:6.3f} ms | torch {t_ref*1e3:6.3f} ms | torch/mine {t_ref/t_mine:4.2f}x")


if __name__ == "__main__":
    print("rows x cols (softmax over cols)")
    check_and_bench((64 * 6 * 256, 256))   # the model's attention softmax, flattened
    check_and_bench((4096, 1024))
    check_and_bench((4096, 4096))
