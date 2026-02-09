"""Ego-centric action computation for Moving MNIST.

Treats one digit as "ego vehicle" and tracks its motion specifically.
Other digit becomes part of the "environment" to predict.
"""

import numpy as np
from typing import Tuple, Optional


def separate_digits(frame: np.ndarray, threshold: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Separate two digits in a frame using connected components.
    
    Args:
        frame: [H, W] grayscale frame
        threshold: Pixel value threshold for digit detection
        
    Returns:
        mask1: Binary mask for digit 1
        mask2: Binary mask for digit 2
    """
    from scipy import ndimage
    
    # Threshold to get digit pixels
    binary = frame > threshold
    
    # Label connected components
    labeled, num_features = ndimage.label(binary)
    
    if num_features < 2:
        # Only one digit visible (might be overlapping or at edge)
        # Return the one digit and empty mask
        if num_features == 1:
            mask1 = (labeled == 1)
            mask2 = np.zeros_like(mask1)
        else:
            mask1 = np.zeros_like(binary)
            mask2 = np.zeros_like(binary)
        return mask1, mask2
    
    # Get the two largest components
    component_sizes = np.bincount(labeled.ravel())[1:]  # Skip background (0)
    largest_indices = np.argsort(component_sizes)[-2:] + 1  # +1 because bincount starts at 0
    
    mask1 = (labeled == largest_indices[0])
    mask2 = (labeled == largest_indices[1])
    
    return mask1, mask2


def get_digit_center(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Get center of mass of a digit mask.
    
    Args:
        mask: Binary mask [H, W]
        
    Returns:
        (y, x) center coordinates or None if empty
    """
    y_coords, x_coords = np.where(mask)
    if len(y_coords) == 0:
        return None
    return (y_coords.mean(), x_coords.mean())


def compute_ego_action(
    frame_t: np.ndarray,
    frame_t1: np.ndarray,
    ego_digit_id: int = 0,
    threshold: int = 50
) -> Tuple[np.ndarray, dict]:
    """Compute ego-centric action (track one specific digit).
    
    Args:
        frame_t: Frame at time t [H, W]
        frame_t1: Frame at time t+1 [H, W]
        ego_digit_id: Which digit to track (0 or 1)
        threshold: Pixel threshold
        
    Returns:
        action: [4] one-hot action vector [up, down, left, right]
        info: Dictionary with debug info
    """
    # Separate digits in both frames
    mask1_t, mask2_t = separate_digits(frame_t, threshold)
    mask1_t1, mask2_t1 = separate_digits(frame_t1, threshold)
    
    # Select ego digit
    ego_mask_t = mask1_t if ego_digit_id == 0 else mask2_t
    ego_mask_t1 = mask1_t1 if ego_digit_id == 0 else mask2_t1
    
    # Get centers
    center_t = get_digit_center(ego_mask_t)
    center_t1 = get_digit_center(ego_mask_t1)
    
    info = {
        'ego_center_t': center_t,
        'ego_center_t1': center_t1,
        'ego_digit_id': ego_digit_id,
        'separated': center_t is not None and center_t1 is not None
    }
    
    if center_t is None or center_t1 is None:
        # Ego digit not visible (occluded or at edge)
        action = np.zeros(4, dtype=np.float32)
        info['action_name'] = 'NONE'
        return action, info
    
    # Compute displacement
    dy, dx = center_t1[0] - center_t[0], center_t1[1] - center_t[1]
    info['displacement'] = (dy, dx)
    
    # Map to cardinal direction
    action = np.zeros(4, dtype=np.float32)
    
    if abs(dy) > abs(dx):
        # Vertical motion dominates
        if dy < 0:
            action[0] = 1.0  # up
            info['action_name'] = 'UP'
        else:
            action[1] = 1.0  # down
            info['action_name'] = 'DOWN'
    else:
        # Horizontal motion dominates
        if dx < 0:
            action[2] = 1.0  # left
            info['action_name'] = 'LEFT'
        else:
            action[3] = 1.0  # right
            info['action_name'] = 'RIGHT'
    
    return action, info


def compute_ego_actions_for_sequence(
    frames: np.ndarray,
    ego_digit_id: int = 0,
    threshold: int = 50
) -> Tuple[np.ndarray, list]:
    """Compute ego-centric actions for entire sequence.
    
    Args:
        frames: [T, H, W] sequence of frames
        ego_digit_id: Which digit to track (0 or 1)
        threshold: Pixel threshold
        
    Returns:
        actions: [T, 4] action vectors
        infos: List of debug info dicts
    """
    T = len(frames)
    actions = np.zeros((T, 4), dtype=np.float32)
    infos = []
    
    # First frame has no action
    infos.append({'action_name': 'START', 'ego_digit_id': ego_digit_id})
    
    for t in range(1, T):
        action, info = compute_ego_action(
            frames[t-1], frames[t],
            ego_digit_id=ego_digit_id,
            threshold=threshold
        )
        actions[t] = action
        infos.append(info)
    
    return actions, infos


def visualize_ego_tracking(frames, ego_actions, ego_infos, save_path=None):
    """Visualize ego-centric tracking.
    
    Shows:
    - Raw frame
    - Separated digits with ego highlighted
    - Ego trajectory over time
    - Action at each frame
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    T, H, W = frames.shape
    
    # Create figure with subplots
    n_cols = min(T, 10)  # Show max 10 frames
    n_rows = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2.5))
    
    for t in range(n_cols):
        # Row 1: Raw frame
        axes[0, t].imshow(frames[t], cmap='gray', vmin=0, vmax=255)
        axes[0, t].set_title(f'Frame {t}')
        axes[0, t].axis('off')
        
        # Row 2: Separated with ego highlighted
        mask1, mask2 = separate_digits(frames[t])
        
        # Color code: digit 1 = cyan, digit 2 = magenta, ego = highlighted
        colored = np.zeros((H, W, 3), dtype=np.uint8)
        colored[mask1] = [0, 255, 255]  # Cyan
        colored[mask2] = [255, 0, 255]  # Magenta
        
        axes[1, t].imshow(colored)
        
        # Add ego center marker
        ego_mask = mask1 if ego_infos[0]['ego_digit_id'] == 0 else mask2
        center = get_digit_center(ego_mask)
        if center:
            circle = Circle((center[1], center[0]), 2, color='yellow', fill=True)
            axes[1, t].add_patch(circle)
        
        axes[1, t].set_title(f'Ego={ego_infos[0]["ego_digit_id"]}')
        axes[1, t].axis('off')
        
        # Row 3: Action
        axes[2, t].imshow(frames[t], cmap='gray', vmin=0, vmax=255)
        
        if t < len(ego_infos):
            action_name = ego_infos[t].get('action_name', 'NONE')
            color = {'UP': 'lime', 'DOWN': 'cyan', 'LEFT': 'red', 
                    'RIGHT': 'orange', 'START': 'white', 'NONE': 'gray'}[action_name]
            
            # Draw arrow
            center_y, center_x = H // 2, W // 2
            dy, dx = 0, 0
            if action_name == 'UP': dy = -1
            elif action_name == 'DOWN': dy = 1
            elif action_name == 'LEFT': dx = -1
            elif action_name == 'RIGHT': dx = 1
            
            if dy != 0 or dx != 0:
                axes[2, t].annotate('', 
                    xy=(center_x + dx * 8, center_y + dy * 8),
                    xytext=(center_x, center_y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=3))
            
            axes[2, t].set_title(action_name, color=color, fontweight='bold')
        axes[2, t].axis('off')
    
    plt.suptitle(f'Ego-Centric Tracking (Ego Digit = {ego_infos[0]["ego_digit_id"]})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"Saved to {save_path}")
    
    return fig


def compare_action_methods(frames, save_path=None):
    """Compare averaged actions vs ego-centric actions side by side."""
    import matplotlib.pyplot as plt
    
    T, H, W = frames.shape
    
    # Compute both types of actions
    ego_actions_0, ego_infos_0 = compute_ego_actions_for_sequence(frames, ego_digit_id=0)
    ego_actions_1, ego_infos_1 = compute_ego_actions_for_sequence(frames, ego_digit_id=1)
    
    # Compute averaged actions (original method)
    avg_actions = []
    for t in range(T):
        if t == 0:
            avg_actions.append(np.zeros(4))
        else:
            # Average center of mass
            prev_frame = frames[t-1]
            curr_frame = frames[t]
            
            prev_thresh = prev_frame > 50
            curr_thresh = curr_frame > 50
            
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
            if prev_com is not None and curr_com is not None:
                dy, dx = curr_com[0] - prev_com[0], curr_com[1] - prev_com[1]
                if abs(dy) > abs(dx):
                    action[0 if dy < 0 else 1] = 1.0
                else:
                    action[2 if dx < 0 else 3] = 1.0
            avg_actions.append(action)
    
    avg_actions = np.array(avg_actions)
    
    # Create visualization
    n_show = min(T, 10)
    fig, axes = plt.subplots(4, n_show, figsize=(n_show*2, 8))
    
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    for t in range(n_show):
        # Row 1: Frame
        axes[0, t].imshow(frames[t], cmap='gray', vmin=0, vmax=255)
        axes[0, t].set_title(f'Frame {t}')
        axes[0, t].axis('off')
        
        # Row 2: Average action
        axes[1, t].imshow(frames[t], cmap='gray', vmin=0, vmax=255)
        avg_idx = np.argmax(avg_actions[t]) if np.any(avg_actions[t] != 0) else -1
        avg_name = action_names[avg_idx] if avg_idx >= 0 else ('START' if t == 0 else 'NONE')
        axes[1, t].set_title(f'Avg: {avg_name}', fontsize=10)
        axes[1, t].axis('off')
        
        # Row 3: Ego 0 action
        axes[2, t].imshow(frames[t], cmap='gray', vmin=0, vmax=255)
        ego0_idx = np.argmax(ego_actions_0[t]) if np.any(ego_actions_0[t] != 0) else -1
        ego0_name = action_names[ego0_idx] if ego0_idx >= 0 else ('START' if t == 0 else 'NONE')
        axes[2, t].set_title(f'Ego0: {ego0_name}', fontsize=10, color='cyan')
        axes[2, t].axis('off')
        
        # Row 4: Ego 1 action
        axes[3, t].imshow(frames[t], cmap='gray', vmin=0, vmax=255)
        ego1_idx = np.argmax(ego_actions_1[t]) if np.any(ego_actions_1[t] != 0) else -1
        ego1_name = action_names[ego1_idx] if ego1_idx >= 0 else ('START' if t == 0 else 'NONE')
        axes[3, t].set_title(f'Ego1: {ego1_name}', fontsize=10, color='magenta')
        axes[3, t].axis('off')
    
    axes[0, 0].set_ylabel('Frame', fontsize=12, rotation=0, ha='right', va='center')
    axes[1, 0].set_ylabel('Avg COM', fontsize=12, rotation=0, ha='right', va='center')
    axes[2, 0].set_ylabel('Ego Digit 0', fontsize=12, rotation=0, ha='right', va='center', color='cyan')
    axes[3, 0].set_ylabel('Ego Digit 1', fontsize=12, rotation=0, ha='right', va='center', color='magenta')
    
    plt.suptitle('Action Computation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"Saved comparison to {save_path}")
    
    return fig


if __name__ == '__main__':
    # Test ego-centric actions
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from tests.mnist_diffusion.moving_mnist_precomputed import MovingMNISTPrecomputed
    
    print("Loading Moving MNIST...")
    dataset = MovingMNISTPrecomputed(
        data_path='mnist_test_seq.1.npy',
        seq_len=20,
        num_sequences=1,
        frame_size=32,
    )
    
    # Get frames (uint8, 0-255)
    sample = dataset[0]
    frames_uint8 = (sample['frames'].numpy() * 255).astype(np.uint8)
    
    print(f"\nComputing ego-centric actions...")
    print("=" * 60)
    
    # Compare methods
    fig = compare_action_methods(frames_uint8, save_path='outputs/action_comparison.png')
    
    # Detailed ego tracking for digit 0
    print("\nEgo Digit 0 tracking:")
    ego_actions_0, ego_infos_0 = compute_ego_actions_for_sequence(frames_uint8, ego_digit_id=0)
    for t, (action, info) in enumerate(zip(ego_actions_0, ego_infos_0)):
        action_name = info.get('action_name', 'NONE')
        disp = info.get('displacement', (0, 0))
        print(f"  Frame {t:2d}: {action_name:6s} (dy={disp[0]:+5.1f}, dx={disp[1]:+5.1f})")
    
    print("\nEgo Digit 1 tracking:")
    ego_actions_1, ego_infos_1 = compute_ego_actions_for_sequence(frames_uint8, ego_digit_id=1)
    for t, (action, info) in enumerate(zip(ego_actions_1, ego_infos_1)):
        action_name = info.get('action_name', 'NONE')
        disp = info.get('displacement', (0, 0))
        print(f"  Frame {t:2d}: {action_name:6s} (dy={disp[0]:+5.1f}, dx={disp[1]:+5.1f})")
    
    print("\n" + "=" * 60)
    print("See outputs/action_comparison.png for visualization")
