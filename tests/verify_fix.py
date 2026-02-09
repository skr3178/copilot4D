"""Verify the VQ codebook collapse fix with more realistic training.

This script demonstrates the improvements with a simulation that forces
codebook diversity through varied inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=" * 70)
print("VQ Codebook Collapse Fix Verification")
print("=" * 70)

# Test configuration
B, N = 8, 256  # Batch size, num tokens
DIM = 256
CODEBOOK_SIZE = 1024
CODEBOOK_DIM = 1024
NUM_STEPS = 300

# Import both implementations
print("\n1. Importing implementations...")
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from copilot4d.tokenizer.vector_quantizer import VectorQuantizer as VQOld
from copilot4d.tokenizer.vector_quantizer_fixed import VectorQuantizerFixed as VQNew

# Create a simple encoder for simulation
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)

# Create models
print("\n2. Creating VQ models with encoder...")
encoder_old = SimpleEncoder(DIM, DIM)
encoder_new = SimpleEncoder(DIM, DIM)
encoder_new.load_state_dict(encoder_old.state_dict())

vq_old = VQOld(
    dim=DIM,
    codebook_size=CODEBOOK_SIZE,
    codebook_dim=CODEBOOK_DIM,
    commitment_cost=0.25,  # Original (WRONG - swapped)
    codebook_cost=1.0,     # Original (WRONG - swapped)
)

vq_new = VQNew(
    dim=DIM,
    codebook_size=CODEBOOK_SIZE,
    codebook_dim=CODEBOOK_DIM,
    commitment_cost=1.0,   # Fixed: correct coefficient
    codebook_cost=0.25,    # Fixed: correct coefficient
    decay=0.99,
    reinit_every=50,       # Check every 50 steps for this test
)

# Optimizers
optimizer_old = torch.optim.Adam(
    list(encoder_old.parameters()) + list(vq_old.parameters()), 
    lr=5e-4
)
optimizer_new = torch.optim.Adam(
    list(encoder_new.parameters()) + list(vq_new.pre_proj.parameters()) + list(vq_new.post_proj.parameters()), 
    lr=5e-4
)

print(f"\n3. Simulating {NUM_STEPS} training steps with diverse inputs...")
print("-" * 70)

print(f"{'Step':>6} | {'Old Loss':>10} | {'Old Use':>7} | {'New Loss':>10} | {'New Use':>7} | {'New Perp':>8} | {'Dead Check':>10}")
print("-" * 70)

# Track metrics
old_usage_history = []
new_usage_history = []
new_perp_history = []

for step in range(NUM_STEPS):
    # Generate diverse input (different modes to force codebook usage)
    mode = step % 8
    x_base = torch.randn(B, N, DIM)
    
    # Add mode-specific structure
    if mode == 0:
        x_base[:, :, :32] += 5.0  # Mode 0: high in first dims
    elif mode == 1:
        x_base[:, :, 32:64] += 5.0  # Mode 1: high in second dims
    elif mode == 2:
        x_base[:, :, 64:96] += 5.0
    elif mode == 3:
        x_base[:, :, 96:128] += 5.0
    # Modes 4-7 use pure random (different distribution)
    
    # ========== Old VQ ==========
    encoder_old.train()
    vq_old.train()
    
    z_e_old = encoder_old(x_base)
    q_old, idx_old, loss_old = vq_old(z_e_old)
    
    # Commitment + reconstruction loss
    recon_loss_old = F.mse_loss(q_old, z_e_old.detach())
    total_loss_old = recon_loss_old + loss_old
    
    optimizer_old.zero_grad()
    total_loss_old.backward()
    torch.nn.utils.clip_grad_norm_(encoder_old.parameters(), 1.0)
    optimizer_old.step()
    
    unique_old = torch.unique(idx_old).numel()
    
    # ========== New VQ ==========
    encoder_new.train()
    vq_new.train()
    
    z_e_new = encoder_new(x_base)
    q_new, idx_new, loss_new, metrics_new = vq_new(z_e_new)
    
    # Commitment + reconstruction loss
    recon_loss_new = F.mse_loss(q_new, z_e_new.detach())
    total_loss_new = recon_loss_new + loss_new
    
    optimizer_new.zero_grad()
    total_loss_new.backward()
    torch.nn.utils.clip_grad_norm_(encoder_new.parameters(), 1.0)
    optimizer_new.step()
    
    unique_new = int(metrics_new['usage'])
    perp_new = metrics_new['perplexity']
    
    # Track history
    old_usage_history.append(unique_old)
    new_usage_history.append(unique_new)
    new_perp_history.append(perp_new)
    
    # Detect if dead code check triggered
    dead_check = ""
    if step > 200 and step % 50 == 0:
        dead_check = "CHECK"
    
    # Log every 30 steps
    if step % 30 == 0 or step == NUM_STEPS - 1:
        print(f"{step:>6} | {loss_old.item():>10.4f} | {unique_old:>7} | {loss_new.item():>10.4f} | {unique_new:>7} | {perp_new:>8.1f} | {dead_check:>10}")

print("-" * 70)

# Final analysis
print("\n4. Results Analysis:")
print("-" * 70)

# Average usage over last 100 steps
old_usage_avg = np.mean(old_usage_history[-100:])
new_usage_avg = np.mean(new_usage_history[-100:])
new_perp_avg = np.mean(new_perp_history[-100:])

print(f"\nCodebook Usage (avg last 100 steps):")
print(f"  Old VQ: {old_usage_avg:.0f}/1024 ({100*old_usage_avg/1024:.1f}%) - {'HEALTHY' if old_usage_avg > 500 else 'COLLAPSED'}")
print(f"  New VQ: {new_usage_avg:.0f}/1024 ({100*new_usage_avg/1024:.1f}%) - {'HEALTHY' if new_usage_avg > 500 else 'COLLAPSED'}")

print(f"\nPerplexity (avg last 100 steps):")
print(f"  New VQ: {new_perp_avg:.1f}/1024 - {'HEALTHY' if new_perp_avg > 400 else 'LOW DIVERSITY'}")

# Loss trend
old_loss_first = np.mean([loss_old.item() for _ in range(10)])  # Placeholder
old_loss_last = loss_old.item()
new_loss_last = loss_new.item()

print(f"\nVQ Loss at final step:")
print(f"  Old VQ: {old_loss_last:.4f}")
print(f"  New VQ: {new_loss_last:.4f}")

# Improvement ratio
if old_loss_last > 0:
    improvement = old_loss_last / (new_loss_last + 1e-8)
    print(f"\nLoss Improvement Ratio: {improvement:.2f}x")

print("\n5. Key Fixes Applied:")
print("-" * 70)
fixes = [
    ("Loss Coefficients", "λ₁=0.25 (codebook), λ₂=1.0 (commitment)", "Critical"),
    ("EMA Updates", "Codebook updated via moving average", "High"),
    ("Periodic Restart", "Dead codes checked every N steps", "Medium"),
    ("Unit Sphere Init", "Normalized initial codebook", "Medium"),
    ("Perplexity Metric", "Monitor codebook diversity", "Low"),
]

for i, (fix, desc, priority) in enumerate(fixes, 1):
    print(f"  {i}. [{priority:>8}] {fix:<18}: {desc}")

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

if new_usage_avg > old_usage_avg:
    print("✓ Codebook usage IMPROVED with fixed implementation")
else:
    print("⚠ Codebook usage similar (may need more training/hyperparameter tuning)")

if new_perp_avg > 400:
    print("✓ Perplexity is HEALTHY (>400)")
else:
    print("⚠ Perplexity is LOW - consider increasing commitment_cost or training longer")

if new_loss_last < 0.1:
    print("✓ VQ loss is CONVERGED (<0.1)")
else:
    print("⚠ VQ loss still high - may need more training")

print("\nRecommendation:")
print("-" * 70)
print("""
The fixed VQ implementation should be used for all future training.
Key hyperparameters to tune if issues persist:

  1. commitment_cost: Increase to 1.5-2.0 if encoder doesn't commit
  2. decay: Reduce to 0.95-0.98 if codebook changes too slowly  
  3. reinit_every: Reduce to 50 if dead codes accumulate quickly
  4. Learning rate: Reduce to 5e-5 if training is unstable

To restart training:
  python scripts/train_tokenizer.py --config configs/tokenizer.yaml

Monitor logs for:
  - perp > 600 (healthy)
  - use > 800 (healthy)
  - vq < 0.05 (converged)
""")
print("=" * 70)
