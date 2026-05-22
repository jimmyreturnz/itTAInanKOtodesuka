import torch
from taiko.model.decoder import TaikoDecoder
from taiko.model.model import build_model

model = build_model(vocab_size=4000)

mel = torch.randn(2, 128, 800)
tokens = torch.randint(0, 4000, (2, 128))
cond = torch.randint(0, 100, (2, 5))
mask = torch.ones(2, 128, dtype=torch.bool)

loss = model(mel, tokens, cond, mask)

print(loss)