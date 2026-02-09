"""Create comparison video: Averaged COM vs Ego-Centric actions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio

from tests.mnist_diffusion.moving_mnist_precomputed import MovingMNISTPrecomputed
from tests.mnist_diffusion.ego_centric_actions import (
    separate_digits, get_digit_center, compute_ego_actions_for_sequence
)


def compute_averaged_actions(frames):
    """Compute actions using averaged center-of-mass (original method)."""
    T, H, W = frames.shape
    actions = np.zeros((T, 4), dtype=np.float32)
    infos = []
    
    infos.append({'action_name': 'START'})
    
    for t in range(1, T):
        prev_frame = frames[t-1]
        curr_frame = frames[t]
        
        prev_thresh = prev_frame > 50
        curr_thresh = curr_frame > 50
        
        # Average center of mass
        y_coords, x_coords = np.where(prev_thresh)
        if len(y_coords) > 0:
            prev_com = (y_coords.mean(), x_coords.mean())
        else:
            prev_com = None
            
        y_coords, x_coords = np.where(curr_thresh)
        if len(y_coords) > 0:
            curr_com = (y_coords.mean(), x_coords.mean())
        else:
            curr_com = None
        
        action = np.zeros(4)
        info = {}
        
        if prev_com is not None and curr_com is not None:
            dy, dx = curr_com[0] - prev_com[0], curr_com[1] - prev_com[1]
            info['displacement'] = (dy, dx)
            
            if abs(dy) > abs(dx):
                if dy < 0:
                    action[0] = 1.0
                    info['action_name'] = 'UP'
                else:
                    action[1] = 1.0
                    info['action_name'] = 'DOWN'
            else:
                if dx < 0:
                    action[2] = 1.0
                    info['action_name'] = 'LEFT'
                else:
                    action[3] = 1.0
                    info['action_name'] = 'RIGHT'
        else:
            info['action_name'] = 'NONE'
        
        actions[t] = action
        infos.append(info)
    
    return actions, infos


def create_comparison_video(dataset, idx=0, ego_digit_id=0, fps=4, output_path='ego_comparison.gif'):
    """Create side-by-side comparison: Averaged COM vs Ego-Centric."""
    sample = dataset[idx]
    frames = (sample['frames'].numpy() * 255).astype(np.uint8)
    
    T, H, W = frames.shape
    
    # Compute both action types
    avg_actions, avg_infos = compute_averaged_actions(frames)
    ego_actions, ego_infos = compute_ego_actions_for_sequence(frames, ego_digit_id=ego_digit_id)
    
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    colors = {'UP': 'lime', 'DOWN': 'cyan', 'LEFT': 'red', 'RIGHT': 'orange', 
              'START': 'white', 'NONE': 'gray'}
    
    video_frames = []
    
    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Moving MNIST - Frame {t}/19 (Sequence #{idx})', 
                     fontsize=14, fontweight='bold')
        
        # Get action names
        avg_idx = np.argmax(avg_actions[t]) if np.any(avg_actions[t] != 0) else -1
        avg_name = action_names[avg_idx] if avg_idx >= 0 else ('START' if t == 0 else 'NONE')
        
        ego_idx = np.argmax(ego_actions[t]) if np.any(ego_actions[t] != 0) else -1
        ego_name = action_names[ego_idx] if ego_idx >= 0 else ('START' if t == 0 else 'NONE')
        
        # Left: Averaged COM
        axes[0].imshow(frames[t], cmap='gray', vmin=0, vmax=255)
        axes[0].set_title(f'Averaged COM: {avg_name}', fontsize=14, 
                         color=colors.get(avg_name, 'white'), fontweight='bold')
        
        # Draw arrow for averaged action
        center_y, center_x = H // 2, W // 2
        dy, dx = 0, 0
        if avg_name == 'UP': dy = -1
        elif avg_name == 'DOWN': dy = 1
        elif avg_name == 'LEFT': dx = -1
        elif avg_name == 'RIGHT': dx = 1
        
        if dy != 0 or dx != 0:
            axes[0].annotate('', 
                xy=(center_x + dx * 8, center_y + dy * 8),
                xytext=(center_x, center_y),
                arrowprops=dict(arrowstyle='->', color=colors[avg_name], lw=4))
        
        # Draw center of mass marker
        mask = frames[t] > 50
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            com_y, com_x = y_coords.mean(), x_coords.mean()
            circle = Circle((com_x, com_y), 1.5, color='yellow', fill=True, alpha=0.7)
            axes[0].add_patch(circle)
        
        axes[0].axis('off')
        
        # Right: Ego-Centric
        axes[1].imshow(frames[t], cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f'Ego Digit {ego_digit_id}: {ego_name}', fontsize=14,
                         color=colors.get(ego_name, 'white'), fontweight='bold')
        
        # Draw arrow for ego action
        dy, dx = 0, 0
        if ego_name == 'UP': dy = -1
        elif ego_name == 'DOWN': dy = 1
        elif ego_name == 'LEFT': dx = -1
        elif ego_name == 'RIGHT': dx = 1
        
        if dy != 0 or dx != 0:
            axes[1].annotate('',
                xy=(center_x + dx * 8, center_y + dy * 8),
                xytext=(center_x, center_y),
                arrowprops=dict(arrowstyle='->', color=colors[ego_name], lw=4))
        
        # Separate digits and highlight ego
        mask1, mask2 = separate_digits(frames[t])
        ego_mask = mask1 if ego_digit_id == 0 else mask2
        
        # Draw ego center
        ego_center = get_digit_center(ego_mask)
        if ego_center:
            circle = Circle((ego_center[1], ego_center[0]), 1.5, color='cyan', fill=True, alpha=0.8)
            axes[1].add_patch(circle)
            # Add label
            axes[1].text(ego_center[1], ego_center[0] - 3, f'Ego', 
                        ha='center', va='bottom', fontsize=8, color='cyan',
                        fontweight='bold')
        
        # Draw other digit center (optional)
        other_mask = mask2 if ego_digit_id == 0 else mask1
        other_center = get_digit_center(other_mask)
        if other_center:
            circle = Circle((other_center[1], other_center[0]), 1.5, color='magenta', 
                          fill=True, alpha=0.5)
            axes[1].add_patch(circle)
        
        axes[1].axis('off')
        
        # Add legend
        legend_text = 'Yellow dot = Center of Mass'
        fig.text(0.25, 0.02, legend_text, ha='center', fontsize=10, color='yellow')
        
        legend_text_ego = f'Cyan dot = Ego Digit {ego_digit_id}, Magenta = Other'
        fig.text(0.75, 0.02, legend_text_ego, ha='center', fontsize=10, color='cyan')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_rgb = img[:, :, :3]
        video_frames.append(img_rgb)
        
        plt.close(fig)
        
        if t % 5 == 0:
            print(f'  Processed frame {t}')
    
    # Save video
    print(f'Saving to {output_path}...')
    imageio.mimsave(output_path, video_frames, fps=fps)
    
    print(f'  Frames: {len(video_frames)}')
    print(f'  FPS: {fps}')
    print(f'  Duration: {len(video_frames)/fps:.1f} seconds')
    
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create ego-centric comparison video')
    parser.add_argument('--data_path', type=str, default='mnist_test_seq.1.npy')
    parser.add_argument('--seq_idx', type=int, default=0)
    parser.add_argument('--ego_digit', type=int, default=0, choices=[0, 1])
    parser.add_argument('--frame_size', type=int, default=32)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--output', type=str, default='outputs/ego_comparison.gif')
    args = parser.parse_args()
    
    print(f"Loading Moving MNIST dataset...")
    dataset = MovingMNISTPrecomputed(
        data_path=args.data_path,
        seq_len=20,
        num_sequences=args.seq_idx + 1,
        frame_size=args.frame_size,
    )
    
    print(f"\nCreating comparison video (Ego Digit = {args.ego_digit})...")
    create_comparison_video(
        dataset, 
        idx=args.seq_idx, 
        ego_digit_id=args.ego_digit,
        fps=args.fps, 
        output_path=args.output
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
