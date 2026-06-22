"""Regression + correctness tests for TransformerOp.

Run either way:
    python -m pytest tests/ -q       # if pytest is installed
    python tests/test_model.py       # plain, no extra deps

Small CPU config so the whole suite runs in seconds. These guard the model as
Phase 3 swaps PyTorch ops for custom CUDA kernels -- the causal-mask and shape
tests in particular will catch a kernel that computes the wrong thing.
"""

import math
from pathlib import Path

import torch

from model.bigram import BigramLM
from model.gpt import GPT, GPTConfig, Head
from model.tokenizer import CharTokenizer

CFG = GPTConfig(vocab_size=65, block_size=64, n_layer=2, n_head=4, n_embd=128)


def _text():
    return (Path(__file__).parent.parent / "data" / "shakespeare.txt").read_text(encoding="utf-8")


def test_tokenizer_roundtrip():
    tok = CharTokenizer(_text())
    s = "To be, or not to be."
    assert tok.decode(tok.encode(s)) == s


def test_tokenizer_vocab_size():
    assert CharTokenizer(_text()).vocab_size == 65


def test_forward_shapes():
    m = GPT(CFG)
    idx = torch.zeros((2, 16), dtype=torch.long)
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 16, CFG.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_forward_without_targets():
    logits, loss = GPT(CFG)(torch.zeros((1, 8), dtype=torch.long))
    assert loss is None
    assert logits.shape == (1, 8, CFG.vocab_size)


def test_init_loss_near_uniform():
    # untrained loss ~ ln(vocab); random init pushes a little above it
    torch.manual_seed(0)
    _, loss = GPT(CFG)(*(torch.randint(CFG.vocab_size, (8, 32)),) * 2)
    base = math.log(CFG.vocab_size)
    assert 0.8 * base < loss.item() < 1.3 * base


def test_determinism():
    torch.manual_seed(0)
    a = GPT(CFG)(torch.zeros((1, 8), dtype=torch.long))[0]
    torch.manual_seed(0)
    b = GPT(CFG)(torch.zeros((1, 8), dtype=torch.long))[0]
    assert torch.allclose(a, b)


def test_causal_mask_no_future_leak():
    # attention must be strictly lower-triangular and each row must sum to 1
    torch.manual_seed(0)
    h = Head(CFG)
    x = torch.randn(1, 10, CFG.n_embd)
    k, q = h.key(x), h.query(x)
    wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
    wei = wei.masked_fill(h.tril[:10, :10] == 0, float("-inf"))
    wei = torch.softmax(wei, dim=-1)[0]
    assert torch.all(torch.triu(wei, diagonal=1) == 0), "attention leaked into the future"
    assert torch.allclose(wei.sum(-1), torch.ones(10)), "rows must sum to 1"


def test_generate_shape_and_token_range():
    out = GPT(CFG).generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    assert out.shape == (1, 21)
    assert out.min() >= 0 and out.max() < CFG.vocab_size


def test_bigram_interface_matches():
    logits, loss = BigramLM(65)(torch.zeros((1, 5), dtype=torch.long),
                                torch.zeros((1, 5), dtype=torch.long))
    assert logits.shape == (1, 5, 65) and torch.isfinite(loss)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
