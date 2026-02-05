#!/usr/bin/env python3
"""Plot training loss curves from training.log"""

import re
import matplotlib.pyplot as plt
import numpy as np

# Read the log file
log_file = '/media/skr/storage/self_driving/CoPilot4D/training.log'
with open(log_file, 'r') as f:
    lines = f.readlines()

# Parse data
steps = []
losses = []
depth_losses = []
vq_losses = []
lrs = []

# Pattern to match: loss=8.1180, depth=7.6850, vq=0.0009, lr=0.000001
pattern = r'loss=([\d.]+), depth=([\d.]+), vq=([\d.]+), lr=([\d.e+-]+)'

for line in lines:
    match = re.search(pattern, line)
    if match:
        # Extract step number from the line
        step_match = re.search(r'(\d+)/\d+', line)
        if step_match:
            step = int(step_match.group(1))
            loss = float(match.group(1))
            depth = float(match.group(2))
            vq = float(match.group(3))
            lr = float(match.group(4))
            
            steps.append(step)
            losses.append(loss)
            depth_losses.append(depth)
            vq_losses.append(vq)
            lrs.append(lr)

# Convert to numpy arrays
steps = np.array(steps)
losses = np.array(losses)
depth_losses = np.array(depth_losses)
vq_losses = np.array(vq_losses)
lrs = np.array(lrs)

print(f"Parsed {len(steps)} training steps")
print(f"Step range: {steps.min()} - {steps.max()}")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Total Loss
ax1 = axes[0, 0]
ax1.plot(steps, losses, 'b-', linewidth=0.8, alpha=0.8)
ax1.set_xlabel('Step')
ax1.set_ylabel('Total Loss')
ax1.set_title('Total Loss vs Step')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)

# Plot 2: Depth Loss
ax2 = axes[0, 1]
ax2.plot(steps, depth_losses, 'g-', linewidth=0.8, alpha=0.8)
ax2.set_xlabel('Step')
ax2.set_ylabel('Depth Loss')
ax2.set_title('Depth Loss vs Step')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(left=0)

# Plot 3: VQ Loss
ax3 = axes[1, 0]
ax3.plot(steps, vq_losses, 'r-', linewidth=0.8, alpha=0.8)
ax3.set_xlabel('Step')
ax3.set_ylabel('VQ Loss')
ax3.set_title('VQ Loss vs Step')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(left=0)

# Plot 4: Learning Rate
ax4 = axes[1, 1]
ax4.plot(steps, lrs, 'm-', linewidth=0.8, alpha=0.8)
ax4.set_xlabel('Step')
ax4.set_ylabel('Learning Rate')
ax4.set_title('Learning Rate vs Step')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(left=0)

plt.tight_layout()
plt.savefig('/media/skr/storage/self_driving/CoPilot4D/loss_curves.png', dpi=150, bbox_inches='tight')
print("Saved loss_curves.png")

# Create combined plot
fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(steps, losses, 'b-', linewidth=1, label='Total Loss', alpha=0.9)
ax.plot(steps, depth_losses, 'g-', linewidth=1, label='Depth Loss', alpha=0.9)
ax.plot(steps, vq_losses, 'r-', linewidth=1, label='VQ Loss', alpha=0.9)
ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Loss Curves', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
plt.tight_layout()
plt.savefig('/media/skr/storage/self_driving/CoPilot4D/loss_combined.png', dpi=150, bbox_inches='tight')
print("Saved loss_combined.png")

# Print summary statistics
print("\n=== Training Summary ===")
print(f"Total steps: {len(steps)}")
print(f"\nTotal Loss:")
print(f"  Initial: {losses[0]:.4f}")
print(f"  Final: {losses[-1]:.4f}")
print(f"  Min: {losses.min():.4f} at step {steps[losses.argmin()]}")
print(f"\nDepth Loss:")
print(f"  Initial: {depth_losses[0]:.4f}")
print(f"  Final: {depth_losses[-1]:.4f}")
print(f"  Min: {depth_losses.min():.4f} at step {steps[depth_losses.argmin()]}")
print(f"\nVQ Loss:")
print(f"  Initial: {vq_losses[0]:.4f}")
print(f"  Final: {vq_losses[-1]:.4f}")
print(f"  Min: {vq_losses.min():.4f} at step {steps[vq_losses.argmin()]}")
print(f"\nLearning Rate:")
print(f"  Initial: {lrs[0]:.6f}")
print(f"  Final: {lrs[-1]:.6f}")
