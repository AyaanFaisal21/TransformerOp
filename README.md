# TransformerOp

Train a small GPT from scratch, then make it fast: profile the training loop, hand-write CUDA kernels for the hot paths, and benchmark them honestly against stock PyTorch ops.

Two phases, each justifying the other: the from-scratch model is the *real workload* the kernels are measured against — not synthetic shapes.

## Roadmap

### Phase 1 — Model from scratch (PyTorch, no `nn.Transformer`) ✅
- [x] Char-level tokenizer
- [x] Bigram baseline (establishes the loss floor to beat)
- [x] Single-head causal self-attention
- [x] Multi-head attention + feedforward + residual blocks → full GPT
- [x] Training loop with train/val eval, checkpointing, loss curves
- [x] Generated samples in this README

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

### Phase 1 — model quality

Trained on Tiny Shakespeare (1.0M train / 0.11M val characters, vocab 65), 5000 iters, batch 64 × block 256, AdamW.

| Model | Params | Val loss | Bits/char | What it captures |
|---|---|---|---|---|
| Bigram baseline | 4,225 | 2.49 | 3.59 | adjacent character pairs only |
| **GPT** (6 layer, 6 head, 384 dim) | 10.8M | **1.50** | **2.16** | spelling, names, dialogue structure |

The 2.49 → 1.50 gap is the measured value of attention over a one-character-context model. For reference, Shannon's estimate of the entropy of English is ~1 bit/char — the GPT (2.16) is in the right neighborhood; the bigram is not.

**Honesty notes.** The GPT learns *form*, not *meaning* — output is orthographically and structurally Shakespearean but semantically incoherent (expected at 10M params on 1MB of text). Training shows mild overfitting: val loss bottomed at ~1.478 near iter 3500, then drifted up to 1.50 by iter 5000 while train loss kept falling to 0.997 (early stopping ~3500 would help).

Sample (GPT, 500 tokens, untrimmed):

```
DUKE VINCENTIO:
Base name, good Catesby; as I am hurry
Surprised under him.

ISABELLA:
He would have a foul time to do it.
```

### Phase 2+ — performance

Pre-optimization training throughput: **~15,000 tok/s** on an RTX 2060 SUPER (crude in-loop timing; Phase 2 re-measures with warmup + `cuda.synchronize`). This is the baseline the custom kernels are measured against.

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

NVIDIA RTX 2060 SUPER (8 GB, Turing sm_75), Windows 10. All benchmarks in this repo were run on this card.
