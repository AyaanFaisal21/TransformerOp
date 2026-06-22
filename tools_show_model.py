from model.gpt import GPT, GPTConfig

m = GPT(GPTConfig(vocab_size=65, block_size=256))
print(m)
print("\n--- params by top-level module ---")
for n, mod in m.named_children():
    print(f"{n:12s} {sum(p.numel() for p in mod.parameters()):>12,}")
print(f"{'TOTAL':12s} {sum(p.numel() for p in m.parameters()):>12,}")
