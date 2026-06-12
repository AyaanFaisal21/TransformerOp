"""Character-level tokenizer.

Every distinct character in the training text gets an integer id. Vocab for
Tiny Shakespeare is ~65 symbols. Crude compared to BPE, but it keeps Phase 1
focused on the model -- and the round-trip property below makes it easy to
verify.

Invariant to test: decode(encode(s)) == s for any s drawn from the corpus.
"""


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    # Quick self-test once implemented:
    from pathlib import Path

    text = (Path(__file__).parent.parent / "data" / "shakespeare.txt").read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    sample = text[:200]
    assert tok.decode(tok.encode(sample)) == sample, "round-trip failed"
    print(f"vocab_size={tok.vocab_size}, round-trip OK")