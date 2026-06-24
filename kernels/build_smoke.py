"""Compile + load the smoke-test CUDA extension and check it against torch.

JIT path: torch.utils.cpp_extension.load() invokes nvcc + MSVC under the hood and
returns an importable module. If this prints OK, the Phase 3 build toolchain works.

    python -m kernels.build_smoke
"""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ext = load(
    name="smoke_ext",
    sources=[str(Path(__file__).parent / "smoke_ext.cu")],
    verbose=True,
)

a = torch.randn(100_000, device="cuda")
b = torch.randn(100_000, device="cuda")
out = ext.add(a, b)

assert torch.allclose(out, a + b), "kernel result disagrees with torch"
print("\nextension build + run OK -- Phase 3 toolchain works")
