#!/usr/bin/env python3
"""
Parse training log and plot loss/accuracy curves.
Usage: python scripts/plot_training_curves.py logs/world_model_training.log
"""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """Parse training log to extract metrics."""
    data = {
        'steps': [],
        'loss': [],
        'acc': [],
        'lr': [],
        'obj': [],
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Convert carriage returns to newlines (from tqdm progress bars)
    content = content.replace('\r', '\n')
    
    # Pattern to match: loss=6.9698, acc=0.0000, obj=joi, lr=1.00e-06
    pattern = r'Step\s+(\d+):.*loss=([\d.]+),\s*acc=([\d.]+),\s*obj=(\w+),\s*lr=([\deE.+-]+)'
    
    for match in re.finditer(pattern, content):
        step = int(match.group(1))
        loss = float(match.group(2))
        acc = float(match.group(3))
        obj = match.group(4)
        lr = float(match.group(5))
        
        data['steps'].append(step)
        data['loss'].append(loss)
        data['acc'].append(acc)
        data['obj'].append(obj)
        data['lr'].append(lr)
    
    return data

def plot_curves(data, output_path='figures/training_curves.png'):
    """Plot training curves."""
    steps = np.array(data['steps'])
    loss = np.array(data['loss'])
    acc = np.array(data['acc'])
    lr = np.array(data['lr'])
    obj = np.array(data['obj'])
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CoPilot4D World Model Training Curves', fontsize=14, fontweight='bold')
    
    # 1. Loss curve (colored by objective)
    ax1 = axes[0, 0]
    obj_colors = {'fut': 'blue', 'joi': 'green', 'ind': 'orange'}
    obj_labels = {'fut': 'Future Pred', 'joi': 'Joint Denoise', 'ind': 'Individual Denoise'}
    
    for obj_type in ['fut', 'joi', 'ind']:
        mask = obj == obj_type
        if mask.any():
            ax1.scatter(steps[mask], loss[mask], c=obj_colors[obj_type], 
                       label=obj_labels[obj_type], alpha=0.6, s=20)
    
    # Also plot smoothed overall loss
    window = min(50, len(loss) // 10)
    if window > 1:
        smoothed = np.convolve(loss, np.ones(window)/window, mode='valid')
        ax1.plot(steps[window-1:], smoothed, 'r-', linewidth=2, label=f'Smoothed ({window} steps)', alpha=0.8)
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy curve
    ax2 = axes[0, 1]
    for obj_type in ['fut', 'joi', 'ind']:
        mask = obj == obj_type
        if mask.any():
            ax2.scatter(steps[mask], acc[mask] * 100, c=obj_colors[obj_type],
                       label=obj_labels[obj_type], alpha=0.6, s=20)
    
    if window > 1:
        smoothed_acc = np.convolve(acc * 100, np.ones(window)/window, mode='valid')
        ax2.plot(steps[window-1:], smoothed_acc, 'r-', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Token Prediction Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning rate schedule
    ax3 = axes[1, 0]
    ax3.plot(steps, lr * 1000, 'purple', linewidth=1.5)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Learning Rate (x1000)')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss by objective type (box plot)
    ax4 = axes[1, 1]
    loss_by_obj = [loss[obj == ot] for ot in ['fut', 'joi', 'ind'] if (obj == ot).any()]
    labels_by_obj = [obj_labels[ot] for ot in ['fut', 'joi', 'ind'] if (obj == ot).any()]
    
    if loss_by_obj:
        bp = ax4.boxplot(loss_by_obj, labels=labels_by_obj, patch_artist=True)
        colors = [obj_colors[ot] for ot in ['fut', 'joi', 'ind'] if (obj == ot).any()]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax4.set_ylabel('Loss')
        ax4.set_title('Loss Distribution by Objective')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plots to: {output_path}")
    
    # Print summary statistics
    print("\n=== Training Summary ===")
    print(f"Total steps logged: {len(steps)}")
    print(f"Final loss: {loss[-1]:.4f}")
    print(f"Final accuracy: {acc[-1]*100:.2f}%")
    print(f"Best loss: {loss.min():.4f} (step {steps[loss.argmin()]})")
    print(f"Best accuracy: {acc.max()*100:.2f}% (step {steps[acc.argmax()]})")
    print(f"\nObjective distribution:")
    for obj_type in ['fut', 'joi', 'ind']:
        count = (obj == obj_type).sum()
        pct = count / len(obj) * 100
        print(f"  {obj_labels[obj_type]}: {count} ({pct:.1f}%)")

if __name__ == '__main__':
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'logs/world_model_training.log'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'figures/training_curves.png'
    
    # Create figures directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    data = parse_log_file(log_path)
    if len(data['steps']) == 0:
        print(f"No training data found in {log_path}")
        sys.exit(1)
    
    plot_curves(data, output_path)
