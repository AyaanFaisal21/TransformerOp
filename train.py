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
LEARNING_RATE = 3e-4
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

    TODO(you):
      - draw BATCH_SIZE random start offsets in [0, len(data) - BLOCK_SIZE)
      - x = the BLOCK_SIZE-length slices at those offsets   (B, T)
      - y = the same slices shifted right by one            (B, T)
        (y[b, t] is the "next char" after x[b, t] -- that's the whole
        supervision signal; every position is a training example)
      - move both to DEVICE and return
    """
    raise NotImplementedError


@torch.no_grad()
def estimate_loss(model, train_data, val_data) -> dict[str, float]:
    """Average loss over EVAL_ITERS batches for each split.

    TODO(you): model.eval(), loop over both splits, model.train() after.
    Single-batch loss is too noisy to compare runs -- that's why we average.
    """
    raise NotImplementedError


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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # TODO(you): the training loop.
    #   for it in range(MAX_ITERS):
    #     - every EVAL_INTERVAL iters: estimate_loss, print, and time the
    #       interval (tokens/sec is your Phase 2 baseline number -- log it
    #       from day one)
    #     - get_batch -> forward -> loss
    #     - optimizer.zero_grad(set_to_none=True); loss.backward(); step()
    raise NotImplementedError

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.model}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
