# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRA(nn.Module):
    def __init__(self, base_layer, rank, alpha=1):
        """
        Implements Low-Rank Adaptation (LoRA).

        Args:
            base_layer (nn.Module): The original layer (e.g., Linear or Conv2d).
            rank (int): The rank for the LoRA adaptation.
            alpha (float): Scaling factor for LoRA. Default is 1.
        """
        super(LoRA, self).__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Create the down and up projection layers based on base_layer type
        if isinstance(base_layer, nn.Linear):
            self.down = nn.Linear(base_layer.in_features, rank, bias=False)
            self.up = nn.Linear(rank, base_layer.out_features, bias=False)
        elif isinstance(base_layer, nn.Conv2d):
            self.down = nn.Conv2d(
                base_layer.in_channels, rank, kernel_size=1, bias=False
            )
            self.up = nn.Conv2d(
                rank, base_layer.out_channels, kernel_size=1, bias=False
            )
        else:
            raise ValueError("LoRA currently supports only Linear or Conv2d layers.")

        # Initialize the LoRA weights
        nn.init.zeros_(self.up.weight)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))

    def forward(self, x):
        # Combine the original base layer output with the LoRA output
        return self.base_layer(x) + self.up(self.down(x)) * (self.alpha / self.rank)
