"""
test_model.py

Sanity check for Phase 2 model components.
No real data needed — just checks shapes and forward pass.

Run from project root:
    python test_model.py
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, ".")


def separator(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B          = 2      # batch size
    T_audio    = 819    # mel frames (8192ms / 10ms)
    T_seq      = 512    # token sequence length
    N_COND     = 5      # conditioning tokens

    # ------------------------------------------------------------------ #
    separator("AudioEncoder")
    # ------------------------------------------------------------------ #
    from taiko.model.audio_encoder import AudioEncoder

    encoder = AudioEncoder(n_mels=128, d_model=512).to(device)
    mel     = torch.randn(B, 128, T_audio).to(device)

    ctx = encoder(mel)
    print(f"Input  mel:     {tuple(mel.shape)}")
    print(f"Output context: {tuple(ctx.shape)}   (expected: [{B}, ~205, 512])")
    assert ctx.shape[0] == B
    assert ctx.shape[2] == 512
    print("AudioEncoder ✓")


    # ------------------------------------------------------------------ #
    separator("TaikoDecoder")
    # ------------------------------------------------------------------ #
    from taiko.model.decoder import TaikoDecoder

    VOCAB = 3512
    decoder   = TaikoDecoder(vocab_size=VOCAB, d_model=512, nhead=8, num_layers=6).to(device)
    token_ids = torch.randint(0, VOCAB, (B, T_seq)).to(device)
    cond_ids  = torch.randint(0, 100,   (B, N_COND)).to(device)
    pad_mask  = torch.zeros(B, T_seq, dtype=torch.bool).to(device)  # no padding

    logits = decoder(token_ids, ctx, cond_ids, pad_mask)
    print(f"Input  tokens:  {tuple(token_ids.shape)}")
    print(f"Output logits:  {tuple(logits.shape)}   (expected: [{B}, {T_seq}, {VOCAB}])")
    assert logits.shape == (B, T_seq, VOCAB)
    print("TaikoDecoder ✓")


    # ------------------------------------------------------------------ #
    separator("Full TaikoMapper — training forward pass")
    # ------------------------------------------------------------------ #
    from taiko.model.model import TaikoMapper, TaikoModelConfig
    from taiko.data.tokenizer import TaikoVocabulary

    vocab  = TaikoVocabulary()
    config = TaikoModelConfig(vocab_size=len(vocab))
    model  = TaikoMapper(config).to(device)

    params = model.count_parameters()
    print(f"Parameters: encoder={params['encoder']}  decoder={params['decoder']}  total={params['total']}")

    mel        = torch.randn(B, 128, T_audio).to(device)
    token_ids  = torch.randint(4, len(vocab), (B, T_seq)).to(device)  # skip special tokens
    token_ids[:, 0]  = vocab.SOS_ID
    token_ids[:, -1] = vocab.EOS_ID
    cond_ids   = torch.randint(0, 50, (B, N_COND)).to(device)
    token_mask = torch.ones(B, T_seq, dtype=torch.bool).to(device)

    loss = model(mel, token_ids, cond_ids, token_mask)
    print(f"Training loss: {loss.item():.4f}   (expected: ~{torch.log(torch.tensor(len(vocab))).item():.2f} for random init)")
    assert loss.item() > 0
    print("Training forward pass ✓")

    # Check loss backpropagates
    loss.backward()
    print("Backward pass ✓")


    # ------------------------------------------------------------------ #
    separator("Autoregressive generation (short)")
    # ------------------------------------------------------------------ #
    model.eval()
    mel_single  = torch.randn(1, 128, T_audio).to(device)
    cond_single = torch.randint(0, 50, (1, N_COND)).to(device)

    tokens = model.generate(
        mel_single,
        cond_single,
        max_new_tokens=32,
        temperature=1.0,
        top_p=0.92,
        cfg_scale=1.5,
    )
    print(f"Generated {len(tokens)} tokens: {tokens[:10]}...")
    assert tokens[0] == vocab.SOS_ID
    print("Generation ✓")


    # ------------------------------------------------------------------ #
    separator("VRAM usage")
    # ------------------------------------------------------------------ #
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved  = torch.cuda.memory_reserved(device)  / 1024**3
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print(f"(Training with batch_size=8 + fp16 will use ~4-6 GB)")

    separator("ALL CHECKS PASSED — Model is ready for Phase 3")


if __name__ == "__main__":
    main()
