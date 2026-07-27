"""Dynamic-word ASL classifier: Conv1D temporal stem + Transformer encoder
over a fixed-length sequence of normalized hand+pose landmarks, with masked
attention pooling. A real temporal deep-learning model (as opposed to voting
over per-frame predictions), matched to how winning solutions on Kaggle's
isolated-sign-language competitions approach this problem.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AttentionPool(nn.Module):
    """Learned attention pooling over the time dimension, respecting a
    padding mask so padded frames never influence the pooled result."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) with True = valid frame.
        scores = self.query(x).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (x * weights).sum(dim=1)  # (B, D)


class WordClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 192,
        num_layers: int = 3,
        num_heads: int = 6,
        ff_dim: int = 384,
        dropout: float = 0.2,
        max_len: int = 96,
    ):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttentionPool(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim), mask: (B, T) True = valid frame.
        x = self.conv_stem(x.transpose(1, 2)).transpose(1, 2)  # (B, T, d_model)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=~mask)
        pooled = self.pool(x, mask)
        return self.head(pooled)
