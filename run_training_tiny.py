#!/usr/bin/env python
"""Run training with tiny model."""
import sys
import subprocess

# Run with tiny model parameters
cmd = [
    sys.executable,
    "tests/mnist_diffusion/train_mnist_fast.py",
    "--embed_dim", "128",
    "--num_layers", "2",
    "--num_heads", "4",
    "--batch_size", "4",
    "--epochs", "50",
    "--num_train", "2000",
    "--num_val", "500",
    "--output_dir", "outputs/mnist_diffusion_fast",
]

print("Running command:", " ".join(cmd))
subprocess.run(cmd)
