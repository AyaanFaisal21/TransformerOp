"""Overhead-targeted optimization for a small (overhead-bound) model: CUDA Graphs.

A CUDA graph records a whole sequence of kernel launches once and replays it with
a single launch, eliminating per-launch CPU/Python overhead. (torch.compile needs
Triton, unavailable on this Windows box; CUDA graphs don't.) Prediction: the win is
largest at small batch (GPU starved, overhead dominates) and shrinks as a bigger
batch fills the machine with real work.

    python -m benchmarks.overhead
"""

import torch

from benchmarks.bench import benchmark
from model.gpt import GPT, GPTConfig

DEVICE = "cuda"


def capture_forward(model, x):
    """Warm up, then capture model(x) forward into a replayable CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                model(x)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        static_logits, _ = model(x)
    return g, static_logits


def main():
    cfg = GPTConfig(vocab_size=65, block_size=256)
    torch.manual_seed(0)
    model = GPT(cfg).to(DEVICE).eval()

    print("forward (T=256): eager vs CUDA-graph replay")
    for B in (1, 8, 64):
        x = torch.randint(65, (B, 256), device=DEVICE)

        def eager_f():
            with torch.no_grad():
                model(x)

        te = benchmark(eager_f, warmup=20, iters=50)

        g, static_logits = capture_forward(model, x)
        g.replay(); torch.cuda.synchronize()
        with torch.no_grad():
            ref = model(x)[0]
        ok = torch.allclose(static_logits, ref, atol=1e-3)
        tg = benchmark(lambda: g.replay(), warmup=20, iters=50)

        tok = B * 256
        print(f"  B={B:3d}: eager {te*1e3:7.2f} ms ({tok/te:>10,.0f} tok/s) | "
              f"graph {tg*1e3:7.2f} ms ({tok/tg:>10,.0f} tok/s) | {te/tg:.2f}x | correct={ok}")


if __name__ == "__main__":
    main()
