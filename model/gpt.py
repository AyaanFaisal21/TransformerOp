"""A small GPT, written by hand -- no nn.Transformer, no F.scaled_dot_product_attention.

The explicit attention math here is the point: in Phase 3 these exact lines
(matmul -> mask -> softmax -> matmul) are what get profiled and replaced with
custom CUDA kernels. Keep them naive and readable.

Shape conventions used throughout:
  B = batch size, T = sequence length (<= block_size), C = n_embd, hs = head size
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256   # max context length
    vocab_size: int = 65    # set from tokenizer at runtime
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384       # must be divisible by n_head
    dropout: float = 0.2


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # tril is fixed, not learned -> register_buffer (moves to GPU with the
        # model, but the optimizer never touches it).
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> out: (B, T, hs)
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # the all-pairs score grid, in one matmul; scale keeps softmax sane
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5     # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # blank the future
        wei = F.softmax(wei, dim=-1)                            # each row sums to 1
        wei = self.dropout(wei)

        out = wei @ v                                           # (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """n_head Heads in parallel, concatenated, then projected back to n_embd."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # TODO(you): nn.ModuleList of Heads, output projection
        # nn.Linear(n_embd, n_embd), dropout.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(you): concat head outputs on the channel dim, project, dropout.
        raise NotImplementedError


class FeedForward(nn.Module):
    """Position-wise MLP: n_embd -> 4*n_embd -> GELU -> n_embd -> dropout."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # TODO(you)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Block(nn.Module):
    """Transformer block: pre-norm residual attention + MLP.

    x = x + attn(ln1(x)); x = x + ffwd(ln2(x))
    Pre-norm (LayerNorm *inside* the residual branch) is what lets deep stacks
    train without warmup tricks -- the residual stream stays an identity path.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # TODO(you): ln1, ln2 (nn.LayerNorm), MultiHeadAttention, FeedForward
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # TODO(you):
        #   - token embedding: nn.Embedding(vocab_size, n_embd)
        #   - position embedding: nn.Embedding(block_size, n_embd)
        #   - nn.Sequential of n_layer Blocks
        #   - final LayerNorm, then lm_head: nn.Linear(n_embd, vocab_size)
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # TODO(you):
        #   tok_emb (B, T, C) + pos_emb (T, C) -> blocks -> ln -> lm_head
        #   loss: same cross-entropy reshape as the bigram model.
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # TODO(you): same loop as BigramLM.generate, but crop the context to
        # the last block_size tokens before each forward pass (the position
        # embedding table has no rows past block_size).
        raise NotImplementedError
