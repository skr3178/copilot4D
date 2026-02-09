"""Create video/GIF of Moving MNIST sequence with actions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import imageio
import argparse

from tests.mnist_diffusion.moving_mnist_precomputed import MovingMNISTPrecomputed


def create_video(dataset, idx=0, fps=5, output_path='mnist_video.mp4'):
    """Create video of Moving MNIST sequence with action arrows."""
    sample = dataset[idx]
    frames = sample["frames"].numpy()  # (T, H, W)
    actions = sample["actions"].numpy()  # (T, 4)
    
    T, H, W = frames.shape
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    colors = {'UP': 'lime', 'DOWN': 'cyan', 'LEFT': 'red', 'RIGHT': 'orange', 'NONE': 'gray'}
    
    # Create frames
    video_frames = []
    
    for t in range(T):
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Display frame
        ax.imshow(frames[t], cmap='gray', vmin=0, vmax=1)
        ax.set_xlim(-1, W + 1)
        ax.set_ylim(H + 1, -1)
        ax.axis('off')
        
        # Get action
        action_vec = actions[t]
        if np.any(action_vec != 0):
            action_idx = np.argmax(action_vec)
            action_name = action_names[action_idx]
            color = colors[action_name]
            
            # Draw large arrow in center
            dy, dx = 0, 0
            if action_name == 'UP': dy = -1
            elif action_name == 'DOWN': dy = 1
            elif action_name == 'LEFT': dx = -1
            elif action_name == 'RIGHT': dx = 1
            
            center_y, center_x = H // 2, W // 2
            
            # Draw arrow with annotation
            ax.annotate('', 
                       xy=(center_x + dx * 10, center_y + dy * 10), 
                       xytext=(center_x, center_y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=5, 
                                      mutation_scale=30))
            
            # Add action text
            ax.text(center_x, center_y - 5, action_name, 
                   ha='center', va='bottom', fontsize=20, 
                   fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                            edgecolor=color, linewidth=2, alpha=0.8))
        else:
            action_name = 'START'
            ax.text(W//2, H//2, 'START', 
                   ha='center', va='center', fontsize=20, 
                   fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
        
        # Frame counter
        ax.text(2, 2, f'Frame {t}/19', fontsize=14, color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Title
        ax.set_title(f'Moving MNIST - Sequence #{idx}', fontsize=16, pad=10)
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        video_frames.append(img)
        
        plt.close(fig)
    
    # Save video
    output_path = Path(output_path)
    
    if output_path.suffix == '.gif':
        # Save as GIF
        imageio.mimsave(output_path, video_frames, fps=fps)
    else:
        # Save as MP4
        imageio.mimsave(output_path, video_frames, fps=fps, quality=8)
    
    print(f"Video saved to {output_path}")
    print(f"  Frames: {len(video_frames)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(video_frames)/fps:.1f} seconds")
    
    return output_path


def create_comparison_video(dataset, idx=0, fps=5, output_path='mnist_comparison.mp4'):
    """Create side-by-side video showing frames only and with actions."""
    sample = dataset[idx]
    frames = sample["frames"].numpy()
    actions = sample["actions"].numpy()
    
    T, H, W = frames.shape
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    colors = {'UP': 'lime', 'DOWN': 'cyan', 'LEFT': 'red', 'RIGHT': 'orange', 'NONE': 'gray'}
    
    video_frames = []
    
    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left: Frame only
        axes[0].imshow(frames[t], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Frame Only', fontsize=14)
        axes[0].axis('off')
        axes[0].text(2, 2, f'Frame {t}', fontsize=12, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Right: Frame with action
        axes[1].imshow(frames[t], cmap='gray', vmin=0, vmax=1)
        
        # Get action
        action_vec = actions[t]
        if np.any(action_vec != 0):
            action_idx = np.argmax(action_vec)
            action_name = action_names[action_idx]
            color = colors[action_name]
            
            dy, dx = 0, 0
            if action_name == 'UP': dy = -1
            elif action_name == 'DOWN': dy = 1
            elif action_name == 'LEFT': dx = -1
            elif action_name == 'RIGHT': dx = 1
            
            center_y, center_x = H // 2, W // 2
            axes[1].annotate('', 
                           xy=(center_x + dx * 8, center_y + dy * 8), 
                           xytext=(center_x, center_y),
                           arrowprops=dict(arrowstyle='->', color=color, lw=4))
            axes[1].set_title(f'Action: {action_name}', fontsize=14, color=color, fontweight='bold')
        else:
            axes[1].set_title('Action: START', fontsize=14, color='gray')
        
        axes[1].axis('off')
        axes[1].text(2, 2, f'Frame {t}', fontsize=12, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        fig.suptitle(f'Moving MNIST - Sequence #{idx}', fontsize=16)
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        video_frames.append(img)
        
        plt.close(fig)
    
    # Save video
    output_path = Path(output_path)
    
    if output_path.suffix == '.gif':
        imageio.mimsave(output_path, video_frames, fps=fps)
    else:
        imageio.mimsave(output_path, video_frames, fps=fps, quality=8)
    
    print(f"Comparison video saved to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create Moving MNIST video')
    parser.add_argument('--data_path', type=str, default='mnist_test_seq.1.npy')
    parser.add_argument('--seq_idx', type=int, default=0)
    parser.add_argument('--frame_size', type=int, default=32)
    parser.add_argument('--fps', type=int, default=5, help='Frames per second')
    parser.add_argument('--output', type=str, default='outputs/mnist_video.gif')
    parser.add_argument('--comparison', action='store_true', help='Create side-by-side comparison')
    args = parser.parse_args()
    
    print(f"Loading Moving MNIST dataset...")
    dataset = MovingMNISTPrecomputed(
        data_path=args.data_path,
        seq_len=20,
        num_sequences=args.seq_idx + 1,
        frame_size=args.frame_size,
    )
    
    if args.comparison:
        create_comparison_video(dataset, idx=args.seq_idx, fps=args.fps, output_path=args.output)
    else:
        create_video(dataset, idx=args.seq_idx, fps=args.fps, output_path=args.output)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
