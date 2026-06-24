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
    """One head of causal self-attention -- the readable, single-head reference.

    The model uses the batched `MultiHeadAttention` below (all heads in a few large
    ops). This class stays as the 1:1-with-the-math version and is exercised by
    tests/test_model.py.
    """

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
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)  # Q, K, V in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hs = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=-1)                # each (B, T, C), one matmul
        # split channels into heads, move the head dim next to batch -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) * hs ** -0.5          # (B, nh, T, T), all heads at once
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v                                         # (B, nh, T, hs)

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # reassemble heads -> (B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise MLP: n_embd -> 4*n_embd -> GELU -> n_embd -> dropout."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),  # widen
            nn.GELU(),                               # the nonlinearity that makes depth count
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),  # project back
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: pre-norm residual attention ++ MLP.

    x = x + attn(ln1(x)); x = x + ffwd(ln2(x))
    Pre-norm (LayerNorm *inside* the residual branch) is what lets deep stacks
    train without warmup tricks -- the residual stream stays an identity path.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the `x +` is the residual: each sub-layer proposes an *edit* to x
        x = x + self.attn(self.ln1(x))   # communicate across positions
        x = x + self.ffwd(self.ln2(x))   # then compute per position
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        tok = self.token_emb(idx)                                  # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))     # (T, C), broadcast over B
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                   # (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]   # crop: pos_emb has no rows past block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
