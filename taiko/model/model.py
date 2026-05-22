"""
taiko/model/model.py

Full TaikoMapper model: AudioEncoder + TaikoDecoder.

Forward pass (training):
    mel        → AudioEncoder → audio_context
    cond_ids   → prepended to decoder input
    token_ids  → TaikoDecoder (teacher forcing) → logits
    loss       = cross_entropy(logits, targets, ignore PAD + cond prefix)

Forward pass (inference):
    mel        → AudioEncoder → audio_context
    cond_ids   + SOS → TaikoDecoder (autoregressive) → token sequence
             → OsuTaikoSerializer → .osu file
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from taiko.model.audio_encoder import AudioEncoder
from taiko.model.decoder import TaikoDecoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TaikoModelConfig:
    # Vocabulary
    vocab_size:      int   = 3_512    # ~3500 TIME tokens + note/special/cond tokens
    pad_token_id:    int   = 0
    sos_token_id:    int   = 1
    eos_token_id:    int   = 2
    n_cond_tokens:   int   = 5        # number of conditioning tokens prepended

    # Audio encoder
    n_mels:          int   = 128
    encoder_d_model: int   = 256

    # Decoder
    d_model:         int   = 256
    nhead:           int   = 4
    num_layers:      int   = 4
    dim_feedforward: int   = 1024
    dropout:         float = 0.1
    max_seq_len:     int   = 512

    # Classifier-free guidance (CFG)
    cfg_dropout:     float = 0.1      # prob of dropping conditioning during training


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TaikoMapper(nn.Module):
    """
    End-to-end taiko beatmap generator.

    Parameters (approx):
        AudioEncoder:  ~18M  (4 transformer layers + CNN)
        TaikoDecoder:  ~85M  (6 transformer layers, d=512)
        Total:         ~103M
    """

    def __init__(self, config: TaikoModelConfig):
        super().__init__()
        self.config = config

        self.audio_encoder = AudioEncoder(
            n_mels=config.n_mels,
            d_model=config.encoder_d_model,
            dropout=config.dropout,
        )

        self.decoder = TaikoDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len + config.n_cond_tokens,
        )

        # Project encoder output to decoder dimension if they differ
        if config.encoder_d_model != config.d_model:
            self.enc_proj = nn.Linear(config.encoder_d_model, config.d_model)
        else:
            self.enc_proj = nn.Identity()

    def encode_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, 128, T] → context: [B, T//4, d_model]"""
        ctx = self.audio_encoder(mel)
        return self.enc_proj(ctx)

    def forward(
        self,
        mel: torch.Tensor,              # [B, 128, T_audio]
        token_ids: torch.Tensor,        # [B, T_seq]  input (shifted right)
        cond_ids: torch.Tensor,         # [B, N_cond]
        token_mask: torch.Tensor,       # [B, T_seq]  True=real, False=PAD
        cfg_dropout: bool = True,       # apply CFG dropout during training
    ) -> torch.Tensor:
        """
        Training forward pass.
        Returns cross-entropy loss (scalar).
        """
        B = mel.shape[0]

        # Classifier-free guidance: randomly drop conditioning
        if cfg_dropout and self.training:
            drop_mask = torch.rand(B, device=mel.device) < self.config.cfg_dropout
            # Replace conditioning with UNK_COND tokens (last cond token in vocab)
            unk_cond = torch.full_like(cond_ids, cond_ids.max())  # rough UNK proxy
            cond_ids = torch.where(drop_mask.unsqueeze(1), unk_cond, cond_ids)

        # Encode audio
        audio_context = self.encode_audio(mel)  # [B, T//4, D]

        # Decoder input: all tokens except last (teacher forcing)
        dec_input  = token_ids[:, :-1]           # [B, T-1]
        dec_target = token_ids[:, 1:]            # [B, T-1]  (shifted left)
        dec_mask   = token_mask[:, 1:]           # [B, T-1]  real token mask

        # PAD mask for attention: True = ignore
        key_pad_mask = ~token_mask[:, :-1]       # [B, T-1]

        # Decode
        logits = self.decoder(
            token_ids=dec_input,
            audio_context=audio_context,
            cond_ids=cond_ids,
            key_padding_mask=key_pad_mask,
        )  # [B, T-1, vocab_size]

        # Loss: cross-entropy only on real (non-PAD) tokens
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            dec_target.reshape(-1),
            ignore_index=self.config.pad_token_id,
            reduction="none",
        )
        # Mask and mean
        loss = (loss * dec_mask.reshape(-1).float()).sum() / dec_mask.float().sum().clamp(min=1)
        return loss

    @torch.inference_mode()
    def generate(
        self,
        mel: torch.Tensor,              # [1, 128, T]
        cond_ids: torch.Tensor,         # [1, N_cond]
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.92,
        cfg_scale: float = 1.5,         # >1 = stronger conditioning
        eos_token_id: Optional[int] = None,
    ) -> list[int]:
        """
        Autoregressive generation with top-p sampling and classifier-free guidance.

        Returns list of token IDs (without conditioning prefix).
        """
        eos_id = eos_token_id or self.config.eos_token_id
        device = mel.device

        # Encode audio — once, reuse for every step
        audio_context = self.encode_audio(mel)  # [1, T//4, D]

        # For CFG: also encode with null conditioning
        if cfg_scale != 1.0:
            null_cond = torch.full_like(cond_ids, cond_ids.max())
            # Batch both: [2, T//4, D]
            audio_ctx_pair = audio_context.expand(2, -1, -1)
            cond_pair      = torch.cat([cond_ids, null_cond], dim=0)
        else:
            audio_ctx_pair = audio_context
            cond_pair      = cond_ids

        # Start with SOS
        generated = [self.config.sos_token_id]

        for _ in range(max_new_tokens):
            seq = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)

            if cfg_scale != 1.0:
                seq_pair = seq.expand(2, -1)
                logits = self.decoder(
                    token_ids=seq_pair,
                    audio_context=audio_ctx_pair,
                    cond_ids=cond_pair,
                )  # [2, T, vocab_size]
                # CFG: interpolate between conditioned and unconditioned
                cond_logits, uncond_logits = logits[0, -1], logits[1, -1]
                logits_final = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            else:
                logits = self.decoder(
                    token_ids=seq,
                    audio_context=audio_context,
                    cond_ids=cond_ids,
                )  # [1, T, vocab_size]
                logits_final = logits[0, -1]   # [vocab_size]

            # Temperature + top-p sampling
            next_token = _sample_top_p(logits_final, temperature, top_p)
            generated.append(next_token)

            if next_token == eos_id:
                break

        return generated  # includes SOS + EOS

    def count_parameters(self) -> dict:
        enc_params = sum(p.numel() for p in self.audio_encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total      = enc_params + dec_params
        return {
            "encoder":  f"{enc_params/1e6:.1f}M",
            "decoder":  f"{dec_params/1e6:.1f}M",
            "total":    f"{total/1e6:.1f}M",
        }


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Top-p (nucleus) sampling."""
    logits = logits / max(temperature, 1e-5)
    probs  = F.softmax(logits, dim=-1)

    # Sort descending
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens beyond top_p
    remove = cumsum - sorted_probs > top_p
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum()

    # Sample
    next_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx[next_idx].item()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(vocab_size: int, **kwargs) -> TaikoMapper:
    config = TaikoModelConfig(vocab_size=vocab_size, **kwargs)
    return TaikoMapper(config)
