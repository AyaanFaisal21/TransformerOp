"""Training entry point.

Usage:
    python train.py --model bigram     # plumbing check + loss floor
    python train.py --model gpt        # the real thing

Wired-up infrastructure: arg parsing, data loading, device selection,
checkpointing. The learning-critical pieces (batching, eval, the loop itself)
are TODOs.
"""

import argparse
import time
from pathlib import Path

import torch

from model.bigram import BigramLM
from model.gpt import GPT, GPTConfig
from model.tokenizer import CharTokenizer

# ---- hyperparameters (tuned for an 8 GB RTX 2060; shrink batch_size first if OOM)
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
# Per-model: the bigram's smooth loss surface tolerates a far higher rate than
# AdamW on a deep transformer, which diverges if pushed.
LEARNING_RATE = {"bigram": 1e-2, "gpt": 3e-4}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA = Path(__file__).parent / "data" / "shakespeare.txt"
CKPT_DIR = Path(__file__).parent / "checkpoints"


def load_data() -> tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    """Returns (train_ids, val_ids, tokenizer) with a 90/10 split."""
    text = DATA.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], tok


def get_batch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input, target) sequences.

    y[b, t] is the "next char" after x[b, t] -- every position is a
    training example, which is what makes one batch worth B*T of them.
    """
    offsets = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in offsets])
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in offsets])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data) -> dict[str, float]:
    """Average loss over EVAL_ITERS batches for each split.

    Single-batch loss is too noisy to compare runs -- that's why we average.
    """
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bigram", "gpt"], default="gpt")
    args = parser.parse_args()

    train_data, val_data, tok = load_data()
    print(f"device={DEVICE}, vocab_size={tok.vocab_size}, "
          f"train={len(train_data):,} tokens, val={len(val_data):,} tokens")

    if args.model == "bigram":
        model = BigramLM(tok.vocab_size)
    else:
        cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE)
        model = GPT(cfg)
    model = model.to(DEVICE)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE[args.model])

    t0 = time.time()
    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            dt = time.time() - t0
            # tokens seen since the last eval == iters * tokens-per-batch
            iters_done = EVAL_INTERVAL if it else 1
            tok_per_sec = iters_done * BATCH_SIZE * BLOCK_SIZE / dt if dt else 0
            print(f"iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {tok_per_sec:,.0f} tok/s")
            t0 = time.time()

        x, y = get_batch(train_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.model}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
