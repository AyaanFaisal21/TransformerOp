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
    use_sdpa: bool = True   # True: fused F.scaled_dot_product_attention; False: naive materialized


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
        self.use_sdpa = cfg.use_sdpa
        self.dropout_p = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)  # Q, K, V in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor, cache=None, cache_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        hs = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=-1)                # each (B, T, C), one matmul
        # split channels into heads, move the head dim next to batch -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        if cache is not None:
            # incremental decode: write this step's K/V into the cache, attend over all so far
            ck, cv = cache                                    # preallocated (B, nh, block_size, hs)
            ck[:, :, cache_pos:cache_pos + T] = k
            cv[:, :, cache_pos:cache_pos + T] = v
            k = ck[:, :, :cache_pos + T]                      # all keys seen so far
            v = cv[:, :, :cache_pos + T]
            # T==1 (decode step): the new query attends to ALL cached keys -> no mask.
            # T>1 only happens priming the seed at cache_pos==0 -> standard causal.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        elif self.use_sdpa:
            # fused FlashAttention kernel: scale + causal mask + softmax + @V, one call
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0)
        else:
            # naive materialized path (the (B, nh, T, T) matrix hits memory)
            wei = (q @ k.transpose(-2, -1)) * hs ** -0.5      # (B, nh, T, T), all heads at once
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.attn_dropout(wei)
            out = wei @ v                                     # (B, nh, T, hs)

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # reassemble heads -> (B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out

    def decode_step(self, x, kc, vc, pos, key_pos):
        """One fixed-shape decode step, built for CUDA-graph capture.

        x: (B,1,C). kc/vc: STATIC caches (B, nh, block_size, hs) preallocated to the full
        context. pos: a (1,) long *tensor*; key_pos: arange(block_size).

        Two decisions make this graph-capturable: (1) we always attend over the *full*
        block_size (fixed shape every step, masking slots > pos), and (2) the position is
        a tensor, not a Python int -- so the single recorded graph is valid at every step.
        K/V are written with index_copy_ (a tensor index) instead of a Python slice.
        """
        B, _, C = x.shape
        hs = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, 1, self.n_head, hs).transpose(1, 2)
        k = k.view(B, 1, self.n_head, hs).transpose(1, 2)
        v = v.view(B, 1, self.n_head, hs).transpose(1, 2)
        kc.index_copy_(2, pos, k)                            # write this token's K at `pos`
        vc.index_copy_(2, pos, v)
        att = (q @ kc.transpose(-2, -1)) * hs ** -0.5        # (B, nh, 1, block_size)
        att = att.masked_fill(key_pos > pos, float("-inf"))  # keep only positions 0..pos
        att = F.softmax(att, dim=-1)
        out = att @ vc                                       # (B, nh, 1, hs)
        out = out.transpose(1, 2).contiguous().view(B, 1, C)
        return self.proj(out)


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

    def forward(self, x: torch.Tensor, cache=None, cache_pos: int = 0) -> torch.Tensor:
        # the `x +` is the residual: each sub-layer proposes an *edit* to x
        x = x + self.attn(self.ln1(x), cache, cache_pos)   # communicate across positions
        x = x + self.ffwd(self.ln2(x))                     # then compute per position
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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                kv_caches=None, cache_pos: int = 0):
        B, T = idx.shape
        tok = self.token_emb(idx)                                          # (B, T, C)
        pos = self.pos_emb(torch.arange(cache_pos, cache_pos + T, device=idx.device))
        x = tok + pos
        for i, block in enumerate(self.blocks):
            x = block(x, kv_caches[i] if kv_caches is not None else None, cache_pos)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                           # (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def _empty_cache(self, B, device):
        hs = self.cfg.n_embd // self.cfg.n_head
        return [[torch.zeros(B, self.cfg.n_head, self.cfg.block_size, hs, device=device),
                 torch.zeros(B, self.cfg.n_head, self.cfg.block_size, hs, device=device)]
                for _ in range(self.cfg.n_layer)]

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, use_cache: bool = True) -> torch.Tensor:
        if not use_cache:
            # naive: re-run the full forward on the whole context every token
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.block_size:]
                logits, _ = self(idx_cond)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
            return idx

        # KV-cached: compute each new token once; attend over cached K/V
        kv = self._empty_cache(idx.shape[0], idx.device)
        seed = idx[:, -self.cfg.block_size:]
        logits, _ = self(seed, kv_caches=kv, cache_pos=0)   # prime the cache with the seed
        pos = seed.shape[1]
        for _ in range(max_new_tokens):
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if pos >= self.cfg.block_size:
                break   # cache full; sliding-window eviction not implemented (context limit)
            logits, _ = self(next_id, kv_caches=kv, cache_pos=pos)
            pos += 1
        return idx
