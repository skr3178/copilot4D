"""Camera Shift approach for Moving MNIST - Ego-centric view simulation.

Instead of tracking individual digits, we artificially shift the entire canvas
to simulate an ego-vehicle camera view. The action is the global camera displacement.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import imageio

from tests.mnist_diffusion.moving_mnist_precomputed import MovingMNISTPrecomputed


def apply_camera_shift(frames, shifts):
    """Apply global camera shifts to sequence.
    
    Args:
        frames: [T, H, W] original frames
        shifts: [T, 2] list of (dy, dx) shifts for each frame
        
    Returns:
        shifted_frames: [T, H, W] frames after camera shift
        visible_region: [T, 4] (y_start, y_end, x_start, x_end) for each frame
    """
    T, H, W = frames.shape
    shifted_frames = np.zeros_like(frames)
    visible_regions = []
    
    # Cumulative shift (camera position relative to world)
    cum_dy, cum_dx = 0, 0
    
    for t in range(T):
        # Apply new shift
        if t > 0:
            cum_dy += shifts[t][0]
            cum_dx += shifts[t][1]
        
        # Shift the frame (roll/circular for simplicity, or crop)
        # Use np.roll for circular shift (simulates infinite world)
        shifted = np.roll(frames[t], shift=(cum_dy, cum_dx), axis=(0, 1))
        shifted_frames[t] = shifted
        
        # Track what part of original frame is visible
        # For visualization purposes
        visible_regions.append((cum_dy, cum_dx))
    
    return shifted_frames, visible_regions


def compute_camera_shift_to_follow_ego(frames, target_digit_id=0):
    """Compute camera shifts to keep target digit roughly centered.
    
    This simulates the camera "following" the ego digit.
    
    Args:
        frames: [T, H, W] frames
        target_digit_id: Which digit to follow (0 or 1)
        
    Returns:
        shifts: [T, 2] camera shifts (dy, dx)
        actions: [T, 4] action vectors representing camera motion
    """
    from scipy import ndimage
    
    T, H, W = frames.shape
    shifts = [(0, 0)]  # First frame: no shift
    actions = [np.zeros(4)]  # First frame: no action
    
    prev_center = None
    
    for t in range(T):
        # Separate digits
        binary = frames[t] > 50
        labeled, num_features = ndimage.label(binary)
        
        if num_features >= 2:
            # Get component centers
            component_indices = np.arange(1, min(num_features + 1, 3))
            centers = []
            for idx in component_indices:
                y, x = np.where(labeled == idx)
                if len(y) > 0:
                    centers.append((y.mean(), x.mean()))
            
            # Select target digit center
            if len(centers) >= 2:
                # Sort by y position to be consistent
                centers = sorted(centers, key=lambda c: c[0])
                target_center = centers[target_digit_id]
            elif len(centers) == 1:
                target_center = centers[0]
            else:
                target_center = None
        elif num_features == 1:
            y, x = np.where(binary)
            target_center = (y.mean(), x.mean()) if len(y) > 0 else None
        else:
            target_center = None
        
        if t > 0:
            if target_center is not None and prev_center is not None:
                # Camera should shift to counteract digit motion
                # If digit moves RIGHT, camera shifts RIGHT to keep it centered
                dy = int(round(target_center[0] - prev_center[0]))
                dx = int(round(target_center[1] - prev_center[1]))
                
                # Limit shift to reasonable values
                dy = np.clip(dy, -3, 3)
                dx = np.clip(dx, -3, 3)
                
                shifts.append((dy, dx))
                
                # Convert to action
                action = np.zeros(4)
                if abs(dy) > abs(dx):
                    if dy < 0:
                        action[0] = 1.0  # camera UP
                    else:
                        action[1] = 1.0  # camera DOWN
                else:
                    if dx < 0:
                        action[2] = 1.0  # camera LEFT
                    else:
                        action[3] = 1.0  # camera RIGHT
                actions.append(action)
            else:
                shifts.append((0, 0))
                actions.append(np.zeros(4))
        
        prev_center = target_center
    
    return shifts, np.array(actions)


def compute_camera_shift_absolute(frames, shifts_list):
    """Apply arbitrary camera shifts (e.g., constant pan)."""
    T = len(frames)
    actions = [np.zeros(4)]
    
    for t in range(1, T):
        dy, dx = shifts_list[t]
        action = np.zeros(4)
        if abs(dy) > abs(dx):
            action[0 if dy < 0 else 1] = 1.0
        elif dx != 0:
            action[2 if dx < 0 else 3] = 1.0
        actions.append(action)
    
    return shifts_list, np.array(actions)


def create_camera_shift_comparison_video(
    dataset, 
    idx=0, 
    mode='follow_ego',  # 'follow_ego' or 'fixed_pan'
    fps=4,
    output_path='camera_shift_comparison.gif'
):
    """Create video comparing original vs camera-shifted views."""
    sample = dataset[idx]
    frames = (sample['frames'].numpy() * 255).astype(np.uint8)
    T, H, W = frames.shape
    
    # Compute camera shifts
    if mode == 'follow_ego':
        shifts, actions = compute_camera_shift_to_follow_ego(frames, target_digit_id=0)
        mode_str = "Camera Follows Ego Digit 0"
    else:
        # Constant pan to the right
        shifts_list = [(0, 0)] + [(0, 1)] * (T - 1)  # Pan right
        shifts, actions = compute_camera_shift_absolute(frames, shifts_list)
        mode_str = "Camera Pans Right"
    
    # Apply shifts
    shifted_frames, visible_regions = apply_camera_shift(frames, shifts)
    
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    colors = {'UP': 'lime', 'DOWN': 'cyan', 'LEFT': 'red', 'RIGHT': 'orange', 
              'START': 'white', 'NONE': 'gray'}
    
    video_frames = []
    
    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{mode_str} - Frame {t}/19', fontsize=14, fontweight='bold')
        
        # Get action name
        action_idx = np.argmax(actions[t]) if np.any(actions[t] != 0) else -1
        action_name = action_names[action_idx] if action_idx >= 0 else ('START' if t == 0 else 'NONE')
        
        # Left: Original View (Fixed Camera)
        axes[0].imshow(frames[t], cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original View\n(Fixed Camera)', fontsize=12)
        
        # Mark center
        axes[0].axhline(y=H//2, color='r', linestyle='--', alpha=0.3)
        axes[0].axvline(x=W//2, color='r', linestyle='--', alpha=0.3)
        axes[0].text(W//2, 2, 'CENTER', ha='center', color='red', fontsize=9)
        axes[0].axis('off')
        
        # Right: Camera-Shifted View (Ego-Centric)
        axes[1].imshow(shifted_frames[t], cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f'Camera-Shifted View\n(Action: {action_name})', 
                         fontsize=12, color=colors.get(action_name, 'white'))
        
        # Draw action arrow
        center_y, center_x = H // 2, W // 2
        dy, dx = shifts[t] if t < len(shifts) else (0, 0)
        
        if abs(dy) > 0 or abs(dx) > 0:
            # Normalize for display
            scale = 8
            axes[1].annotate('',
                xy=(center_x + dx * scale, center_y + dy * scale),
                xytext=(center_x, center_y),
                arrowprops=dict(arrowstyle='->', color=colors.get(action_name, 'white'), 
                              lw=4, mutation_scale=30))
        
        # Mark center
        axes[1].axhline(y=H//2, color='lime', linestyle='--', alpha=0.5)
        axes[1].axvline(x=W//2, color='lime', linestyle='--', alpha=0.5)
        axes[1].text(W//2, 2, 'EGO CENTER', ha='center', color='lime', fontsize=9)
        
        # Add camera shift info
        cum_dy = sum(s[0] for s in shifts[:t+1])
        cum_dx = sum(s[1] for s in shifts[:t+1])
        axes[1].text(2, H-2, f'Camera shift: ({cum_dy}, {cum_dx})', 
                    ha='left', va='bottom', fontsize=9, color='yellow',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        axes[1].axis('off')
        
        # Add explanatory text
        explanation = (
            "Concept: Camera moves to follow ego-digit\n"
            "Action = Camera displacement\n"
            "Challenge: Predict other digit motion relative to camera"
        )
        fig.text(0.5, 0.02, explanation, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
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


def create_three_way_comparison(dataset, idx=0, fps=4, output_path='three_way_comparison.gif'):
    """Compare: Original | Ego-Tracked | Camera-Shifted."""
    sample = dataset[idx]
    frames = (sample['frames'].numpy() * 255).astype(np.uint8)
    T, H, W = frames.shape
    
    # Compute ego-tracked actions (my original implementation)
    from tests.mnist_diffusion.ego_centric_actions import compute_ego_actions_for_sequence
    ego_actions, ego_infos = compute_ego_actions_for_sequence(frames, ego_digit_id=0)
    
    # Compute camera-shifted
    shifts, cam_actions = compute_camera_shift_to_follow_ego(frames, target_digit_id=0)
    shifted_frames, _ = apply_camera_shift(frames, shifts)
    
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    colors = {'UP': 'lime', 'DOWN': 'cyan', 'LEFT': 'red', 'RIGHT': 'orange', 
              'START': 'white', 'NONE': 'gray'}
    
    video_frames = []
    
    for t in range(T):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Moving MNIST Action Comparison - Frame {t}/19', fontsize=14, fontweight='bold')
        
        # Original
        axes[0].imshow(frames[t], cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original\n(Averaged COM)', fontsize=11)
        axes[0].axis('off')
        
        # Ego-tracked
        axes[1].imshow(frames[t], cmap='gray', vmin=0, vmax=255)
        ego_name = ego_infos[t].get('action_name', 'NONE')
        axes[1].set_title(f'Ego-Tracked\n(Digit 0: {ego_name})', fontsize=11,
                         color=colors.get(ego_name, 'white'))
        axes[1].axis('off')
        
        # Camera-shifted
        axes[2].imshow(shifted_frames[t], cmap='gray', vmin=0, vmax=255)
        cam_idx = np.argmax(cam_actions[t]) if np.any(cam_actions[t] != 0) else -1
        cam_name = action_names[cam_idx] if cam_idx >= 0 else ('START' if t == 0 else 'NONE')
        axes[2].set_title(f'Camera-Shifted\n(Camera: {cam_name})', fontsize=11,
                         color=colors.get(cam_name, 'white'))
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_rgb = img[:, :, :3]
        video_frames.append(img_rgb)
        
        plt.close(fig)
        
        if t % 5 == 0:
            print(f'  Processed frame {t}')
    
    # Save
    print(f'Saving to {output_path}...')
    imageio.mimsave(output_path, video_frames, fps=fps)
    
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create camera shift visualization')
    parser.add_argument('--data_path', type=str, default='mnist_test_seq.1.npy')
    parser.add_argument('--seq_idx', type=int, default=0)
    parser.add_argument('--mode', type=str, default='follow_ego', choices=['follow_ego', 'fixed_pan'])
    parser.add_argument('--frame_size', type=int, default=32)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--output', type=str, default='outputs/camera_shift.gif')
    parser.add_argument('--three_way', action='store_true', help='Create three-way comparison')
    args = parser.parse_args()
    
    print(f"Loading Moving MNIST dataset...")
    dataset = MovingMNISTPrecomputed(
        data_path=args.data_path,
        seq_len=20,
        num_sequences=args.seq_idx + 1,
        frame_size=args.frame_size,
    )
    
    if args.three_way:
        print(f"\nCreating three-way comparison...")
        create_three_way_comparison(dataset, idx=args.seq_idx, fps=args.fps, 
                                   output_path=args.output)
    else:
        print(f"\nCreating camera shift video (mode: {args.mode})...")
        create_camera_shift_comparison_video(
            dataset, 
            idx=args.seq_idx, 
            mode=args.mode,
            fps=args.fps, 
            output_path=args.output
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
