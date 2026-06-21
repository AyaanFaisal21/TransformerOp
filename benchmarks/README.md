# Benchmarks

Phase 2 onwards artifacts live here.

- `profile.md` — where a training step's time actually goes (PyTorch profiler, then Nsight). Written *before* any kernel work; every Phase 3 kernel must point at a line in this document to justify its existence.
- Per-kernel benchmark tables: custom kernel vs. stock `torch` op, at the model's real shapes (B=64, T=256, C=384, 6 heads) and a sweep around them. Report median of N runs after warmup, with `torch.cuda.synchronize()` around timing. Include the configurations where the stock op wins.
- End-to-end: training step time before/after kernel swap (Phase 4).

Hardware for all numbers: RTX 2060 8 GB (Turing sm_75), Windows 10.
