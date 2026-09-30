"""Compact T-Prime-style Transformer plugin for the RFML Toolkit.

The toolkit calls build_model with the number of classes, input length, and
input channels inferred from the selected training dataset.
"""

import math

import torch
import torch.nn as nn


MODEL_NAME = "Tiny T-Prime"


class TinyTPrime(nn.Module):
    """Small Transformer encoder for 1-D real or I/Q RF sequences."""

    def __init__(self, num_classes, input_size, in_channels, embed_dim=64,
                 num_heads=4, num_layers=2, patch_size=16, dropout=0.1):
        super().__init__()
        # Strided convolution converts consecutive RF samples into tokens.
        self.patch_embed = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_tokens = max(1, math.ceil(input_size / patch_size))
        self.position = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, in_channels, input_size)
        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = tokens + self.position[:, :tokens.size(1)]
        features = self.norm(self.encoder(tokens).mean(dim=1))
        return self.classifier(features)  # unnormalized logits


def build_model(*, num_classes, input_size, in_channels, **kwargs):
    """Required RFML Toolkit plugin factory."""
    return TinyTPrime(
        num_classes=num_classes,
        input_size=input_size,
        in_channels=in_channels,
        embed_dim=int(kwargs.get("embed_dim", 64)),
        num_heads=int(kwargs.get("num_heads", 4)),
        num_layers=int(kwargs.get("num_layers", 2)),
        patch_size=int(kwargs.get("patch_size", 16)),
        dropout=float(kwargs.get("dropout", 0.1)),
    )
