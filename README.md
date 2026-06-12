# TransformerOp

Train a small GPT from scratch, then make it fast: profile the training loop, hand-write CUDA kernels for the hot paths, and benchmark them honestly against stock PyTorch ops.

Two phases, each justifying the other: the from-scratch model is the *real workload* the kernels are measured against — not synthetic shapes.

## Roadmap

### Phase 1 — Model from scratch (PyTorch, no `nn.Transformer`)
- [ ] Char-level tokenizer
- [ ] Bigram baseline (establishes the loss floor to beat)
- [ ] Single-head causal self-attention
- [ ] Multi-head attention + feedforward + residual blocks → full GPT
- [ ] Training loop with train/val eval, checkpointing, loss curves
- [ ] Generated samples in this README

### Phase 2 — Measure
- [ ] Profile one training step (PyTorch profiler → Nsight Systems/Compute)
- [ ] Document where step time actually goes (`benchmarks/profile.md`)

### Phase 3 — Custom CUDA kernels (compiled as PyTorch C++ extensions)
- [ ] Softmax (warp shuffles, online normalization)
- [ ] Matmul progression: naive → shared-memory tiling → vectorized loads, benchmarked at each step
- [ ] Fused causal attention (mini-FlashAttention)
- [ ] Each kernel: benchmark table vs. the `torch` equivalent **at this model's actual shapes** — including the cases where the stock op wins

### Phase 4 — Close the loop
- [ ] Swap custom kernels into the training run
- [ ] End-to-end step time, before vs. after

## Results

*(populated as phases complete — loss curves, profiles, kernel benchmark tables)*

## Layout

```
data/        dataset download + prepared binaries
model/       tokenizer, bigram baseline, GPT
kernels/     CUDA C++ kernels + extension bindings (Phase 3)
benchmarks/  profiles and benchmark tables (Phases 2–4)
train.py     training entry point
sample.py    generate text from a checkpoint
```

## Setup

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python data\get_data.py
```

Phase 3 additionally requires the CUDA Toolkit 12.x (`nvcc`) and MSVC (Visual Studio Build Tools, C++ workload).

## Hardware

NVIDIA RTX 2060 (8 GB, Turing sm_75), Windows 10. All benchmarks in this repo were run on this card.
