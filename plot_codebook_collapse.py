#!/usr/bin/env python3
"""Plot codebook collapse events from training log"""

import re
import matplotlib.pyplot as plt
import numpy as np

# Read the log file
log_file = '/media/skr/storage/self_driving/CoPilot4D/outputs/training_scaled.log'
with open(log_file, 'r') as f:
    lines = f.readlines()

# Parse collapse events
# Pattern: [VQ] Codebook collapse detected: 1023 dead codes (99.9%)
collapse_pattern = r'\[VQ\] Codebook collapse detected: (\d+) dead codes \(([\d.]+)%\)'
step_pattern = r'(\d+)/\d+\s+\['

collapse_steps = []
dead_codes = []
collapse_percentages = []

for i, line in enumerate(lines):
    match = re.search(collapse_pattern, line)
    if match:
        dead = int(match.group(1))
        pct = float(match.group(2))
        
        # Find step number from the same line or previous lines
        step_match = re.search(r'(\d+)/200000', line)
        if step_match:
            step = int(step_match.group(1))
        else:
            # Look backward for step
            for j in range(max(0, i-5), i+1):
                step_match = re.search(r'(\d+)/200000', lines[j])
                if step_match:
                    step = int(step_match.group(1))
                    break
            else:
                step = None
        
        if step is not None:
            collapse_steps.append(step)
            dead_codes.append(dead)
            collapse_percentages.append(pct)
            print(f"Step {step}: {dead} dead codes ({pct}%)")

# Convert to numpy arrays
collapse_steps = np.array(collapse_steps)
collapse_percentages = np.array(collapse_percentages)
dead_codes = np.array(dead_codes)

print(f"\nTotal collapse events: {len(collapse_steps)}")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Collapse percentage vs step
ax1 = axes[0]
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(collapse_steps)))
bars = ax1.bar(range(len(collapse_steps)), collapse_percentages, color=colors, edgecolor='darkred', linewidth=1.5)
ax1.set_xlabel('Collapse Event #', fontsize=12)
ax1.set_ylabel('Dead Code Percentage (%)', fontsize=12)
ax1.set_title('Codebook Collapse Severity', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, pct, step in zip(bars, collapse_percentages, collapse_steps):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{pct:.1f}%\n(step {step})',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Step at which collapse occurs
ax2 = axes[1]
ax2.scatter(collapse_steps, collapse_percentages, s=200, c=collapse_percentages, 
            cmap='Reds', edgecolors='darkred', linewidth=2, zorder=5)
ax2.plot(collapse_steps, collapse_percentages, 'r--', alpha=0.5, linewidth=1.5, zorder=1)

# Add annotations
for step, pct in zip(collapse_steps, collapse_percentages):
    ax2.annotate(f'Step {step}\n{pct:.1f}%', 
                 xy=(step, pct), 
                 xytext=(10, 10), 
                 textcoords='offset points',
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax2.set_xlabel('Training Step', fontsize=12)
ax2.set_ylabel('Dead Code Percentage (%)', fontsize=12)
ax2.set_title('Codebook Collapse Events Over Training', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(left=0)

plt.tight_layout()
plt.savefig('/media/skr/storage/self_driving/CoPilot4D/codebook_collapse.png', dpi=150, bbox_inches='tight')
print("\nSaved codebook_collapse.png")

# Print summary
print("\n" + "="*50)
print("CODEBOOK COLLAPSE SUMMARY")
print("="*50)
for i, (step, dead, pct) in enumerate(zip(collapse_steps, dead_codes, collapse_percentages)):
    print(f"Event {i+1}: Step {step:5d} | {dead:4d} dead codes | {pct:5.1f}%")
print("="*50)
