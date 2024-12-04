# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import os
import sys

import torch.nn as nn

# Add the directory containing lora_custom.py to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from lora_custom import LoRA

# Example usage of LoRA
base_layer = nn.Linear(10, 10)
lora_layer = LoRA(base_layer, rank=4, alpha=1)

print(lora_layer)
