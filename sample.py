"""Generate text from a trained checkpoint.

Usage:
    python sample.py --model gpt --tokens 500
"""

import argparse
from pathlib import Path

import torch

from model.bigram import BigramLM
from model.gpt import GPT, GPTConfig
from model.tokenizer import CharTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA = Path(__file__).parent / "data" / "shakespeare.txt"
CKPT_DIR = Path(__file__).parent / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bigram", "gpt"], default="gpt")
    parser.add_argument("--tokens", type=int, default=500)
    args = parser.parse_args()

    tok = CharTokenizer(DATA.read_text(encoding="utf-8"))
    if args.model == "bigram":
        model = BigramLM(tok.vocab_size)
    else:
        model = GPT(GPTConfig(vocab_size=tok.vocab_size))
    model.load_state_dict(torch.load(CKPT_DIR / f"{args.model}.pt", map_location=DEVICE))
    model.to(DEVICE).eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # start from newline/first id
    out = model.generate(context, max_new_tokens=args.tokens)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
