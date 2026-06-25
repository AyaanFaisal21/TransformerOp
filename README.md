# TransformerOp

A GPT built from scratch (tokenizer → attention → training loop), then a **measured**
optimization study: profile it, hand-write CUDA kernels, and benchmark every change
honestly against PyTorch — keeping only the wins that survive *end-to-end*.

**Headline finding:** at a small model's scale the bottleneck is *overhead* (kernel
launches, occupancy), not *compute* (FLOPs). Textbook compute-reductions (fused
attention, KV cache) are masked; the real wins came from changes that hit the
*measured* bottleneck (batching heads, CUDA graphs).

## Roadmap

- [x] **Phase 1 — Model from scratch** (no `nn.Transformer`): tokenizer, bigram baseline, multi-head attention, GPT, training loop, samples
- [x] **Phase 2 — Measure**: benchmark harness (warmup + `cuda.synchronize` + median), training-step profile, regression tests
- [x] **Phase 3 — Custom CUDA kernels**: softmax ✅, fused causal attention (5 versions) ✅; *matmul ladder skipped* — its tiling / bank-conflict lessons were already covered inside the attention kernels, and a plain matmul only loses to cuBLAS
- [x] **Phase 4 — Close the loop**: swap the attention path, measure the end-to-end training step
- [x] **Phase 5 — Overhead minimization** (bonus): static KV cache + CUDA-graph decode → **4.2× faster generation**

## Results

**Model quality** — Tiny Shakespeare, 5000 iters, B=64 × T=256:

| Model | Params | Val loss | Bits/char |
|---|---|---|---|
| Bigram baseline | 4.2K | 2.49 | 3.59 |
| **GPT** (6L, 6H, 384d) | 10.8M | **1.50** | **2.16** |

The 2.49 → 1.50 gap is the measured value of attention. (Shannon's estimate for English
is ≈1 bit/char; the GPT at 2.16 is close, the bigram isn't.) Output is Shakespearean in
*form*, not meaning — expected at this scale.

**Optimization scorecard** — full detail in [`benchmarks/kernels.md`](benchmarks/kernels.md):

| Change | Measured | Verdict |
|---|---|---|
| Batch attention heads (PyTorch) | gen **4.7×**, train 1.12× | ✅ functional win |
| Custom softmax kernel | = torch at the attention shape | ✅ matched (learning) |
| Fused attention, v1–v5 from scratch | < cuBLAS / SDPA | ❌ honest loss at this scale |
| KV cache | 0.96× (small) / **2.21×** (bigger model) | ⚪ regime-dependent |
| SDPA swap, end-to-end | 0.95× | ⚪ op-level 3.6× didn't survive |
| KV cache vs static cache + **CUDA-graph decode** | **4.2× generation** | ✅ biggest win — overhead fix on the overhead-bound path |

**Three lessons the numbers taught:**
1. **Regime decides.** Compute-reductions help only when compute is the bottleneck — here it isn't, so the KV cache and SDPA fusion barely moved the needle, while overhead fixes (batched heads, CUDA graphs) did.
2. **cuBLAS/SDPA are unbeatable by hand at this scale.** The value was learning the bottleneck hierarchy — reduction cost → occupancy → bank conflicts — measured at every rung of the 5-version attention kernel.
3. **Believe only end-to-end measurements.** SDPA was 3.6× faster on the attention *op* but 0.95× on the full training *step* (attention is only ~10% of it). An op-level win that doesn't survive is ideology, not optimization.

## Run

```powershell
# setup
py -3.10 -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python data\get_data.py

# train + sample
.venv\Scripts\python train.py --model gpt
.venv\Scripts\python sample.py --model gpt --tokens 500

# measure
.venv\Scripts\python -m pytest tests/ -q       # regression tests
.venv\Scripts\python -m benchmarks.bench       # forward / train / generate + op timings
.venv\Scripts\python -m benchmarks.kv_cache       # KV cache regime study
.venv\Scripts\python -m benchmarks.overhead       # CUDA graph overhead study (forward)
.venv\Scripts\python -m benchmarks.graphed_decode # static cache + graphed decode (4.2x gen)

# custom CUDA kernels (need CUDA Toolkit 12.x + MSVC; winbuild.bat sets up the env)
cmd /c "kernels\winbuild.bat -m kernels.softmax"
cmd /c "kernels\winbuild.bat -m kernels.attention"
```

## Layout

```
model/       tokenizer, bigram, GPT
kernels/     CUDA C++ kernels + extension build (winbuild.bat)
benchmarks/  profile.md, kernels.md + the benchmark scripts
train.py / sample.py
```

## Hardware

RTX 2060 SUPER (8 GB, Turing sm_75), Windows 10, torch 2.12 + CUDA 12.6. All numbers from this card.
