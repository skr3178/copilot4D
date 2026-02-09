"""Visualize Moving MNIST sequence with computed actions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch

from tests.mnist_diffusion.moving_mnist_precomputed import MovingMNISTPrecomputed


def action_to_arrow(action):
    """Convert one-hot action to arrow direction."""
    # action: [up, down, left, right]
    if np.all(action == 0):
        return "·", (0, 0), "gray"  # No motion
    
    if action[0] == 1:
        return "↑", (0, 1), "green"  # Up
    elif action[1] == 1:
        return "↓", (0, -1), "blue"  # Down
    elif action[2] == 1:
        return "←", (-1, 0), "red"  # Left
    elif action[3] == 1:
        return "→", (1, 0), "orange"  # Right
    
    return "·", (0, 0), "gray"


def visualize_sequence(dataset, idx=0, save_path=None):
    """Visualize a single sequence with actions."""
    sample = dataset[idx]
    frames = sample["frames"].numpy()  # (T, H, W) in [0, 1]
    tokens = sample["tokens"].numpy()  # (T, H, W) in [0, 15]
    actions = sample["actions"].numpy()  # (T, 4) one-hot
    
    T, H, W = frames.shape
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(3, T, figure=fig, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.2)
    
    fig.suptitle(f'Moving MNIST Sequence #{idx} - Frame-by-Frame with Actions', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Raw frames
    for t in range(T):
        ax = fig.add_subplot(gs[0, t])
        ax.imshow(frames[t], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Frame {t}', fontsize=10)
        ax.axis('off')
    
    # Row 2: Frames with action overlay
    for t in range(T):
        ax = fig.add_subplot(gs[1, t])
        ax.imshow(frames[t], cmap='gray', vmin=0, vmax=1)
        
        # Add action arrow
        arrow_text, (dx, dy), color = action_to_arrow(actions[t])
        
        # Draw arrow in center of frame
        center_y, center_x = H // 2, W // 2
        
        if np.any(actions[t] != 0):
            # Draw arrow
            ax.annotate('', xy=(center_x + dx * 8, center_y - dy * 8), 
                       xytext=(center_x, center_y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=3))
        
        # Add action label at bottom
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        action_idx = np.argmax(actions[t]) if np.any(actions[t] != 0) else -1
        action_name = action_names[action_idx] if action_idx >= 0 else 'NONE'
        
        ax.text(center_x, H + 2, action_name, ha='center', va='top',
               fontsize=9, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlim(-1, W + 1)
        ax.set_ylim(H + 4, -1)
        ax.axis('off')
    
    # Row 3: Action timeline
    ax_timeline = fig.add_subplot(gs[2, :])
    ax_timeline.set_xlim(-0.5, T - 0.5)
    ax_timeline.set_ylim(-0.5, 3.5)
    
    colors = {'UP': 'green', 'DOWN': 'blue', 'LEFT': 'red', 'RIGHT': 'orange', 'NONE': 'gray'}
    
    for t in range(T):
        arrow_text, _, color = action_to_arrow(actions[t])
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        action_idx = np.argmax(actions[t]) if np.any(actions[t] != 0) else -1
        action_name = action_names[action_idx] if action_idx >= 0 else 'NONE'
        
        # Plot marker
        y_pos = action_idx if action_idx >= 0 else -0.5
        ax_timeline.scatter(t, y_pos, s=200, c=color, zorder=3, edgecolors='black', linewidths=1)
        
        # Add arrow text
        ax_timeline.text(t, y_pos, arrow_text, ha='center', va='center',
                        fontsize=12, fontweight='bold', color='white')
        
        # Connect consecutive actions with line
        if t > 0:
            prev_action_idx = np.argmax(actions[t-1]) if np.any(actions[t-1] != 0) else -0.5
            ax_timeline.plot([t-1, t], [prev_action_idx, y_pos], 'k-', alpha=0.3, zorder=1)
    
    ax_timeline.set_xlabel('Frame', fontsize=12)
    ax_timeline.set_ylabel('Action', fontsize=12)
    ax_timeline.set_xticks(range(T))
    ax_timeline.set_yticks([0, 1, 2, 3])
    ax_timeline.set_yticklabels(['UP', 'DOWN', 'LEFT', 'RIGHT'])
    ax_timeline.grid(True, alpha=0.3)
    ax_timeline.set_title('Action Timeline (computed from center-of-mass motion)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved visualization to {save_path}")
    
    return fig


def create_animation_style(dataset, idx=0, save_path=None):
    """Create a more compact animation-style visualization."""
    sample = dataset[idx]
    frames = sample["frames"].numpy()
    actions = sample["actions"].numpy()
    
    T, H, W = frames.shape
    
    # Create 4x5 grid for 20 frames
    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    fig.suptitle(f'Moving MNIST Sequence #{idx} - Frames with Motion Actions', 
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for t in range(T):
        ax = axes[t]
        
        # Show frame
        ax.imshow(frames[t], cmap='gray', vmin=0, vmax=1)
        
        # Get action
        arrow_text, (dx, dy), color = action_to_arrow(actions[t])
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        action_idx = np.argmax(actions[t]) if np.any(actions[t] != 0) else -1
        action_name = action_names[action_idx] if action_idx >= 0 else 'NONE'
        
        # Draw arrow
        center_y, center_x = H // 2, W // 2
        if np.any(actions[t] != 0):
            ax.annotate('', xy=(center_x + dx * 6, center_y - dy * 6), 
                       xytext=(center_x, center_y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # Title with action
        ax.set_title(f'Frame {t}: {action_name}', fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    # Hide extra subplots if any
    for i in range(T, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved animation-style visualization to {save_path}")
    
    return fig


def analyze_motion_statistics(dataset, num_samples=100):
    """Analyze action distribution in the dataset."""
    action_counts = {'UP': 0, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 0, 'NONE': 0}
    action_transitions = []
    
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        actions = sample["actions"].numpy()
        
        prev_action = None
        for t in range(len(actions)):
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            action_idx = np.argmax(actions[t]) if np.any(actions[t] != 0) else -1
            action_name = action_names[action_idx] if action_idx >= 0 else 'NONE'
            
            action_counts[action_name] += 1
            
            if prev_action is not None:
                action_transitions.append((prev_action, action_name))
            prev_action = action_name
    
    # Plot statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Action distribution
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    colors = ['green', 'blue', 'red', 'orange', 'gray']
    
    axes[0].bar(actions, counts, color=colors, edgecolor='black')
    axes[0].set_xlabel('Action', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'Action Distribution (first {num_samples} sequences)', fontsize=12)
    
    # Add percentage labels
    total = sum(counts)
    for i, (action, count) in enumerate(zip(actions, counts)):
        pct = count / total * 100
        axes[0].text(i, count + total * 0.01, f'{pct:.1f}%', 
                    ha='center', fontsize=10, fontweight='bold')
    
    # Action transition heatmap
    transition_matrix = np.zeros((5, 5))
    action_to_idx = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'NONE': 4}
    
    for prev, curr in action_transitions:
        transition_matrix[action_to_idx[prev], action_to_idx[curr]] += 1
    
    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums
    
    im = axes[1].imshow(transition_matrix, cmap='YlOrRd', aspect='auto')
    axes[1].set_xticks(range(5))
    axes[1].set_yticks(range(5))
    axes[1].set_xticklabels(['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE'])
    axes[1].set_yticklabels(['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE'])
    axes[1].set_xlabel('Next Action', fontsize=12)
    axes[1].set_ylabel('Current Action', fontsize=12)
    axes[1].set_title('Action Transition Probability', fontsize=12)
    
    # Add text annotations
    for i in range(5):
        for j in range(5):
            text = axes[1].text(j, i, f'{transition_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    
    return fig, action_counts


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize Moving MNIST with actions')
    parser.add_argument('--data_path', type=str, default='mnist_test_seq.1.npy')
    parser.add_argument('--seq_idx', type=int, default=0, help='Sequence index to visualize')
    parser.add_argument('--frame_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='outputs/mnist_visualization')
    parser.add_argument('--stats', action='store_true', help='Show action statistics')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Moving MNIST dataset from {args.data_path}...")
    dataset = MovingMNISTPrecomputed(
        data_path=args.data_path,
        seq_len=20,
        num_sequences=100,
        frame_size=args.frame_size,
    )
    print(f"Loaded {len(dataset)} sequences")
    
    # Visualize single sequence
    print(f"\nVisualizing sequence #{args.seq_idx}...")
    
    # Detailed visualization
    fig1 = visualize_sequence(
        dataset, 
        idx=args.seq_idx,
        save_path=output_dir / f'sequence_{args.seq_idx}_detailed.png'
    )
    
    # Animation-style visualization
    fig2 = create_animation_style(
        dataset,
        idx=args.seq_idx,
        save_path=output_dir / f'sequence_{args.seq_idx}_grid.png'
    )
    
    print(f"\nVisualizations saved to {output_dir}/")
    print(f"  - sequence_{args.seq_idx}_detailed.png")
    print(f"  - sequence_{args.seq_idx}_grid.png")
    
    # Show action statistics
    if args.stats:
        print("\nAnalyzing action statistics...")
        fig3, action_counts = analyze_motion_statistics(dataset, num_samples=100)
        fig3.savefig(output_dir / 'action_statistics.png', dpi=150, bbox_inches='tight')
        print(f"Saved action_statistics.png")
        
        print("\nAction distribution:")
        total = sum(action_counts.values())
        for action, count in action_counts.items():
            print(f"  {action}: {count} ({count/total*100:.1f}%)")
    
    plt.show()
    print("\nDone!")


if __name__ == '__main__':
    main()
