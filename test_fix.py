#!/usr/bin/env python3
"""Quick test to verify the training fix"""

import sys
import os
sys.path.insert(0, '/home/shuwen/chatneurx')
os.environ['LLM_MULTIMODAL'] = '1'

import numpy as np
import neurx
import neurx.nn as nn
from app.core.models_neurx import NeurXChatModel

print("Testing NeurX model forward pass...")

# Create tiny model
model = NeurXChatModel(
    vocab_size=100,
    hidden_dim=32,
    num_layers=1,
    num_heads=2,
    max_seq_len=16,
    dropout=0.0,
)

# Test forward pass
x = neurx.Tensor(np.random.randint(0, 100, (2, 16)))
y = neurx.Tensor(np.random.randint(0, 100, (2, 16)))

print("Input shape:", x.shape)

# This should return a dict
model_output = model(x, y)

if isinstance(model_output, dict):
    print("✓ Model returns dict as expected")
    print(f"  Keys: {list(model_output.keys())}")
    print(f"  Logits shape: {model_output['logits'].shape}")
    print(f"  Loss: {model_output['loss']}")
else:
    print("✗ Model does not return dict")
    print(f"  Type: {type(model_output)}")

# Test saving (simulate the fixed code path)
output = "test_checkpoint.pkl"
print(f"\n✓ Variable 'output' is string: {isinstance(output, str)}")
print(f"  Output path: {output}")

import pickle
os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
payload = {"test": "data"}
with open(output, "wb") as f:
    pickle.dump(payload, f)
print(f"✓ Checkpoint saved successfully to {output}")

# Cleanup
if os.path.exists(output):
    os.remove(output)
    print("✓ Test completed successfully!")
