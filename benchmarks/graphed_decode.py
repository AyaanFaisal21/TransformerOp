"""Phase 5 capstone: static KV cache + CUDA-graph decode, end-to-end on generation.

Generation is overhead-bound (hundreds of tiny launches per token). A CUDA graph
needs fixed shapes, so we make every decode step fixed-shape: a STATIC cache
preallocated to block_size, attention over the full cache (masking slots > pos),
and the position passed as a tensor (not a Python int). Then we capture one step
and replay it per token -- one launch instead of hundreds.

    python -m benchmarks.graphed_decode
"""

import time

import torch

from model.gpt import GPT, GPTConfig

DEVICE = "cuda"


def main():
    cfg = GPTConfig(vocab_size=65, block_size=256)
    model = GPT(cfg).to(DEVICE).eval()

    # --- correctness: static decode_step token-by-token must match a full forward ---
    torch.manual_seed(0)
    seq = torch.randint(cfg.vocab_size, (1, 64), device=DEVICE)
    with torch.no_grad():
        full, _ = model(seq)
        kv = model._empty_cache(1, DEVICE)
        key_pos = torch.arange(cfg.block_size, device=DEVICE)
        tok_buf = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
        pos_buf = torch.zeros(1, dtype=torch.long, device=DEVICE)
        steps = []
        for t in range(seq.shape[1]):
            tok_buf.copy_(seq[:, t:t + 1]); pos_buf.fill_(t)
            steps.append(model.decode_step(tok_buf, pos_buf, kv, key_pos))
        inc = torch.stack(steps, dim=1)
    print(f"correctness: allclose={torch.allclose(full, inc, atol=1e-4)} "
          f"max_err={(full - inc).abs().max():.2e}\n")

    # --- speedup: 200-token generation, three paths ---
    n = 200
    idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    for kw in (dict(use_cache=False), dict(use_cache=True), dict(use_graph=True)):
        model.generate(idx, 10, **kw)   # warmup

    def timed(**kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model.generate(idx, n, **kw)
        torch.cuda.synchronize(); return time.perf_counter() - t0

    t_no = timed(use_cache=False)
    t_kv = timed(use_cache=True)
    t_g = timed(use_graph=True)
    print(f"generate {n} tokens:")
    print(f"  no cache        {t_no*1e3:8.1f} ms   {n/t_no:7.1f} tok/s")
    print(f"  KV cache        {t_kv*1e3:8.1f} ms   {n/t_kv:7.1f} tok/s")
    print(f"  static + graph  {t_g*1e3:8.1f} ms   {n/t_g:7.1f} tok/s   ({t_kv/t_g:.2f}x vs KV, {t_no/t_g:.2f}x vs no-cache)")


if __name__ == "__main__":
    main()
