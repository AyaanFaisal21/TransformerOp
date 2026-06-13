"""Bigram baseline: predict the next character from the current one alone.

This is the dumbest possible language model -- a single embedding lookup where
row i is the logit distribution over next-characters given character i. It
exists to (a) get the data/train/sample plumbing working before attention
enters the picture, and (b) establish the loss floor the transformer must
beat. On Tiny Shakespeare it converges around val loss ~2.5; the GPT should
land near ~1.5.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.next_char_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) int ids. Returns (logits, loss)."""
        logits = self.next_char_logits(idx)  # (B, T, C): one grid row per position
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressive sampling: extend idx (B, T) by max_new_tokens."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                        # (B, C): last position only
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_id), dim=1)             # (B, T+1)
        return idx
