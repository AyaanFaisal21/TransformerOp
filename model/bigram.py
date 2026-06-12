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


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO(you): one nn.Embedding(vocab_size, vocab_size).
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) int ids. Returns (logits, loss).

        TODO(you):
          - logits = embedding lookup on idx          -> (B, T, vocab_size)
          - if targets is None: loss = None
          - else: cross_entropy. F.cross_entropy wants (N, C) vs (N,), so
            reshape logits to (B*T, C) and targets to (B*T,).
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressive sampling: extend idx (B, T) by max_new_tokens.

        TODO(you), per step:
          - forward pass, take logits at the last time step  -> (B, C)
          - softmax -> probabilities, torch.multinomial to sample
          - append the sampled id to idx and repeat
        This same loop is reused unchanged by the GPT (with a context crop).
        """
        raise NotImplementedError
