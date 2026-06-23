# Kernels (Cuda Portion - Phase 3)

Custom CUDA kernels, compiled as PyTorch C++ extensions (`torch.utils.cpp_extension`), each replacing a hot path identified in `benchmarks/profile.md`.

Planned progression (each step benchmarked before moving on):

1. **Softmax** — row-wise, online max/sum, warp shuffle reductions. Smallest surface area; the place to learn the extension build, launch configs, and correct GPU timing.
2. **Matmul** — naive → shared-memory tiled → vectorized (`float4`) loads. The classic ladder; the benchmark table at each rung *is* the deliverable.
3. **Fused causal attention** — QK^T → mask → softmax → V in one kernel without materializing the (T, T) attention matrix (mini-FlashAttention).

Requirements (beyond Phase 1): CUDA Toolkit 12.x (`nvcc`), MSVC via Visual Studio Build Tools (C++ workload). Target arch: `sm_75`.

Correctness gate before any benchmark: `torch.allclose` against the reference op at fp32, plus gradcheck where a backward pass is implemented.
