"""Profile one training step and attribute the time, op by op.

The benchmark (bench.py) times ops in isolation. This answers the harder
question: in a *real* training step, where do the milliseconds actually go?
That attribution -- not a guess -- decides what Phase 3 optimizes.

Usage:
    python -m benchmarks.profile_step
    python -m benchmarks.profile_step --batch 32     # if 64 OOMs
"""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from model.gpt import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()

    cfg = GPTConfig(vocab_size=65, block_size=256)
    model = GPT(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(cfg.vocab_size, (args.batch, cfg.block_size), device=DEVICE)
    y = torch.randint(cfg.vocab_size, (args.batch, cfg.block_size), device=DEVICE)

    def step():
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # warm up (first steps include lazy CUDA init / autotuning -- not representative)
    for _ in range(5):
        step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if DEVICE == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(args.steps):
            step()
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    sort_key = "self_cuda_time_total" if DEVICE == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


if __name__ == "__main__":
    main()
