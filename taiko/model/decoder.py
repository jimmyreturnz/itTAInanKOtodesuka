"""
taiko/model/decoder.py

Autoregressive transformer decoder.
Generates taiko note tokens one at a time, attending to audio context.

Architecture:
    - Token embedding + positional encoding
    - Conditioning tokens prepended (not predicted, no loss)
    - N layers of:
        - Masked self-attention (causal — can't see future tokens)
        - Cross-attention to audio encoder output
        - Feed-forward
    - Linear head → vocab logits

During training:
    Input:  [SOS, t1, t2, ..., t_{n-1}]   (teacher forcing)
    Target: [t1,  t2, ..., t_{n-1}, EOS]

During inference:
    Autoregressively sample one token at a time.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TokenEmbedding(nn.Module):
    """Standard learned token embedding with scaling."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale  = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * self.scale


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class DecoderLayer(nn.Module):
    """
    Single transformer decoder layer with pre-norm (more stable than post-norm).

    Components:
        1. Causal self-attention (masked)
        2. Cross-attention to audio context
        3. Feed-forward network
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention (causal)
        self.self_attn    = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1        = nn.LayerNorm(d_model)

        # Cross-attention to audio
        self.cross_attn   = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2        = nn.LayerNorm(d_model)

        # Feed-forward
        self.ff           = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3        = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,               # [B, T_seq, D]
        audio_context: torch.Tensor,   # [B, T_audio, D]
        causal_mask: torch.Tensor,     # [T_seq, T_seq] additive mask
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T_seq] bool
    ) -> torch.Tensor:

        # 1. Causal self-attention (pre-norm)
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )
        x = residual + x

        # 2. Cross-attention to audio (pre-norm)
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(
            query=x,
            key=audio_context,
            value=audio_context,
        )
        x = residual + x

        # 3. Feed-forward (pre-norm)
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + x

        return x


class TaikoDecoder(nn.Module):
    """
    Autoregressive decoder that generates taiko token sequences.

    Input:
        token_ids:     [B, T_seq]        — input sequence (SOS + tokens, shifted right)
        audio_context: [B, T_audio, D]   — from AudioEncoder
        cond_ids:      [B, N_cond]       — conditioning token IDs, prepended

    Output:
        logits: [B, T_seq, vocab_size]   — predictions for each position
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size

        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.pos_enc     = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embedding and lm_head weights (reduces params, improves training)
        self.lm_head.weight = self.token_embed.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        token_ids: torch.Tensor,         # [B, T_seq]
        audio_context: torch.Tensor,     # [B, T_audio, D]
        cond_ids: Optional[torch.Tensor] = None,  # [B, N_cond]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T_seq] True=ignore
    ) -> torch.Tensor:
        """
        Returns logits: [B, T_seq, vocab_size]
        (If conditioning is prepended, logits[:, :N_cond, :] are ignored in loss)
        """
        B, T = token_ids.shape

        # Prepend conditioning tokens if provided
        if cond_ids is not None:
            full_ids = torch.cat([cond_ids, token_ids], dim=1)  # [B, N_cond + T]
        else:
            full_ids = token_ids

        T_full = full_ids.shape[1]

        # Embed tokens
        x = self.token_embed(full_ids)   # [B, T_full, D]
        x = self.pos_enc(x)

        # Build causal mask: upper triangle = -inf
        causal_mask = torch.triu(
            torch.full((T_full, T_full), float("-inf"), device=x.device),
            diagonal=1,
        )

        # Extend key_padding_mask for conditioning prefix (never masked)
        if key_padding_mask is not None and cond_ids is not None:
            N_cond = cond_ids.shape[1]
            cond_mask = torch.zeros(B, N_cond, dtype=torch.bool, device=x.device)
            # key_padding_mask: True = ignore (PAD). Flip for prefix.
            key_padding_mask = torch.cat([cond_mask, key_padding_mask], dim=1)

        # Decoder layers
        for layer in self.layers:
            x = layer(x, audio_context, causal_mask, key_padding_mask)

        x = self.norm(x)
        logits = self.lm_head(x)   # [B, T_full, vocab_size]

        # Return only the token part (strip conditioning prefix from logits)
        if cond_ids is not None:
            N_cond = cond_ids.shape[1]
            logits = logits[:, N_cond:, :]   # [B, T_seq, vocab_size]

        return logits

    def get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((size, size), float("-inf"), device=device),
            diagonal=1,
        )
