#!/usr/bin/env python3
"""Plot advanced training metrics including perplexity and codebook usage"""

import re
import matplotlib.pyplot as plt
import numpy as np

# Read the log file
log_file = '/media/skr/storage/self_driving/CoPilot4D/outputs/training_scaled.log'
with open(log_file, 'r') as f:
    lines = f.readlines()

# Parse training data
steps = []
losses = []
depth_losses = []
vq_losses = []
lrs = []

# Parse perplexity data
perplexity_steps = []
perplexities = []

# Parse codebook collapse events
collapse_steps = []
collapse_counts = []

# Parse re-initialization events
reinit_steps = []

# Pattern to match: loss=8.1180, depth=7.6850, vq=0.0009, lr=0.000001
pattern = r'loss=([\d.]+), depth=([\d.]+), vq=([\d.]+), lr=([\d.e+-]+)'
perplexity_pattern = r'VQ perplexity: ([\d.]+)'
collapse_pattern = r'Codebook collapse detected: (\d+) dead codes'
reinit_pattern = r'Re-initializing ENTIRE codebook'

for i, line in enumerate(lines):
    # Parse training metrics
    match = re.search(pattern, line)
    if match:
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
    
    # Parse perplexity
    p_match = re.search(perplexity_pattern, line)
    if p_match:
        # Find the closest step before this line
        for j in range(i-1, max(0, i-50), -1):
            step_match = re.search(r'(\d+)/\d+', lines[j])
            if step_match:
                perplexity_steps.append(int(step_match.group(1)))
                perplexities.append(float(p_match.group(1)))
                break
    
    # Parse collapse events
    c_match = re.search(collapse_pattern, line)
    if c_match:
        for j in range(i-1, max(0, i-50), -1):
            step_match = re.search(r'(\d+)/\d+', lines[j])
            if step_match:
                collapse_steps.append(int(step_match.group(1)))
                collapse_counts.append(int(c_match.group(1)))
                break
    
    # Parse re-initialization events
    if re.search(reinit_pattern, line):
        for j in range(i-1, max(0, i-50), -1):
            step_match = re.search(r'(\d+)/\d+', lines[j])
            if step_match:
                reinit_steps.append(int(step_match.group(1)))
                break

# Convert to numpy arrays
steps = np.array(steps)
losses = np.array(losses)
depth_losses = np.array(depth_losses)
vq_losses = np.array(vq_losses)
lrs = np.array(lrs)
perplexity_steps = np.array(perplexity_steps)
perplexities = np.array(perplexities)

print(f"Parsed {len(steps)} training steps")
print(f"Parsed {len(perplexities)} perplexity values")
print(f"Found {len(collapse_steps)} collapse events")
print(f"Found {len(reinit_steps)} re-initialization events")

# Create comprehensive figure
fig, axes = plt.subplots(3, 2, figsize=(14, 14))

# Plot 1: Total Loss with collapse events
ax1 = axes[0, 0]
ax1.plot(steps, losses, 'b-', linewidth=0.8, alpha=0.8, label='Total Loss')
for cs in collapse_steps:
    ax1.axvline(x=cs, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_xlabel('Step')
ax1.set_ylabel('Total Loss')
ax1.set_title('Total Loss vs Step (red lines = collapse events)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)

# Plot 2: VQ Loss with re-initialization events
ax2 = axes[0, 1]
ax2.plot(steps, vq_losses, 'r-', linewidth=0.8, alpha=0.8, label='VQ Loss')
for rs in reinit_steps:
    ax2.axvline(x=rs, color='g', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_xlabel('Step')
ax2.set_ylabel('VQ Loss')
ax2.set_title('VQ Loss vs Step (green lines = codebook re-init)')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(left=0)

# Plot 3: Perplexity
ax3 = axes[1, 0]
if len(perplexities) > 0:
    ax3.plot(perplexity_steps, perplexities, 'g-o', linewidth=1.5, markersize=4, label='VQ Perplexity')
    ax3.axhline(y=1024, color='r', linestyle='--', alpha=0.7, label='Codebook Size (1024)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Perplexity')
    ax3.set_title('VQ Perplexity (higher = better codebook utilization)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    print(f"\nPerplexity Stats:")
    print(f"  Initial: {perplexities[0]:.1f}")
    print(f"  Final: {perplexities[-1]:.1f}")
    print(f"  Max: {perplexities.max():.1f} at step {perplexity_steps[perplexities.argmax()]}")
    print(f"  Min: {perplexities.min():.1f} at step {perplexity_steps[perplexities.argmin()]}")
else:
    ax3.text(0.5, 0.5, 'No perplexity data found', ha='center', va='center', transform=ax3.transAxes)

# Plot 4: Codebook Collapse Events
ax4 = axes[1, 1]
if len(collapse_counts) > 0:
    ax4.bar(range(len(collapse_counts)), collapse_counts, color='orange', alpha=0.7)
    ax4.set_xlabel('Collapse Event #')
    ax4.set_ylabel('Dead Codes Count')
    ax4.set_title('Codebook Collapse Events (dead codes per event)')
    ax4.grid(True, alpha=0.3, axis='y')
    print(f"\nCollapse Events: {len(collapse_counts)}")
    print(f"  Average dead codes: {np.mean(collapse_counts):.0f}")
    print(f"  Max dead codes: {np.max(collapse_counts)}")
else:
    ax4.text(0.5, 0.5, 'No collapse events found', ha='center', va='center', transform=ax4.transAxes)

# Plot 5: Combined Loss with Perplexity overlay
ax5 = axes[2, 0]
ax5_twin = ax5.twinx()
ax5.plot(steps, losses, 'b-', linewidth=0.8, alpha=0.7, label='Total Loss')
ax5.plot(steps, depth_losses, 'g-', linewidth=0.8, alpha=0.7, label='Depth Loss')
ax5.plot(steps, vq_losses, 'r-', linewidth=0.8, alpha=0.7, label='VQ Loss')
if len(perplexities) > 0:
    ax5_twin.plot(perplexity_steps, perplexities, 'm-s', linewidth=1.5, markersize=3, label='Perplexity', alpha=0.8)
    ax5_twin.set_ylabel('Perplexity', color='m')
    ax5_twin.tick_params(axis='y', labelcolor='m')
ax5.set_xlabel('Step')
ax5.set_ylabel('Loss')
ax5.set_title('All Losses + Perplexity Overlay')
ax5.legend(loc='upper right')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(left=0)

# Plot 6: Learning Rate
ax6 = axes[2, 1]
ax6.plot(steps, lrs, 'm-', linewidth=0.8, alpha=0.8)
ax6.set_xlabel('Step')
ax6.set_ylabel('Learning Rate')
ax6.set_title('Learning Rate Schedule')
ax6.grid(True, alpha=0.3)
ax6.set_xlim(left=0)

plt.tight_layout()
plt.savefig('/media/skr/storage/self_driving/CoPilot4D/advanced_metrics.png', dpi=150, bbox_inches='tight')
print("\nSaved advanced_metrics.png")

# Print summary
print("\n=== Advanced Metrics Summary ===")
print(f"\nTraining Steps: {len(steps)}")
print(f"Step Range: {steps.min()} - {steps.max()}")
print(f"\nTotal Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"VQ Loss: {vq_losses[0]:.4f} -> {vq_losses[-1]:.4f}")
print(f"\nCodebook Collapse Events: {len(collapse_steps)}")
if len(collapse_steps) > 0:
    print(f"  Steps: {collapse_steps[:5]}..." if len(collapse_steps) > 5 else f"  Steps: {collapse_steps}")
print(f"\nCodebook Re-initializations: {len(reinit_steps)}")
if len(reinit_steps) > 0:
    print(f"  Steps: {reinit_steps[:5]}..." if len(reinit_steps) > 5 else f"  Steps: {reinit_steps}")
