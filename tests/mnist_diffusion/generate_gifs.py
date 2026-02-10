#!/usr/bin/env python
"""Generate continuous GIFs from model predictions."""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_mnist_full import SimpleVideoTransformer, cosine_mask_schedule
from moving_mnist_cached import MovingMNISTCached


def tokens_to_image(tokens, num_levels=16):
    """Convert token indices to grayscale image."""
    # tokens: (H, W) with values 0-15
    img = (tokens.float() / (num_levels - 1) * 255).cpu().numpy().astype(np.uint8)
    return img


@torch.no_grad()
def predict_future_gif(model, dataset, sample_idx, num_past_frames, device, num_steps=20):
    """Generate future prediction and return as frame list."""
    model.eval()
    
    sample = dataset[sample_idx]
    tokens = sample["tokens"].to(device)  # (T, H, W)
    actions = sample["actions"].to(device)  # (T, 2)
    
    T, H, W = tokens.shape
    N = H * W
    
    # Ground truth for comparison
    gt_tokens = tokens.clone()
    
    # Prepare input: past frames visible, future masked
    input_tokens = tokens.clone()
    input_tokens[num_past_frames:] = 16  # mask token
    
    # Causal mask
    causal_mask = torch.full((T, T), float("-inf"), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    # Iterative denoising
    current_tokens = input_tokens.clone().unsqueeze(0)  # (1, T, H, W)
    actions_batch = actions.unsqueeze(0)  # (1, T, 2)
    
    frames_history = []
    
    # Store initial state (all masked future)
    frames_history.append(current_tokens[0].clone())
    
    for step in range(num_steps):
        # Forward pass
        logits = model(current_tokens, actions_batch, causal_mask)  # (1, T, N, V)
        
        # Sample from logits
        probs = F.softmax(logits[0, num_past_frames:], dim=-1)  # (T_future, N, V)
        
        # Greedy decode
        pred_tokens_future = torch.argmax(probs, dim=-1)  # (T_future, N)
        pred_tokens_future = pred_tokens_future.reshape(T - num_past_frames, H, W)
        
        # Update current tokens
        current_tokens[0, num_past_frames:] = pred_tokens_future
        
        # Store every few steps for visualization
        if step % 4 == 0 or step == num_steps - 1:
            frames_history.append(current_tokens[0].clone())
    
    return gt_tokens, frames_history


def create_comparison_gif(gt_tokens, pred_frames_list, num_past, output_path, fps=5):
    """Create a GIF showing prediction progress."""
    T, H, W = gt_tokens.shape
    
    gif_frames = []
    
    for pred_step, pred_tokens in enumerate(pred_frames_list):
        # Create side-by-side comparison
        # Left: Ground Truth, Right: Prediction
        
        # Create grid: [Past | Future GT | Future Pred]
        # Each row shows different frames
        
        # For visualization, show selected frames
        display_frames = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]  # 10 frames
        
        # Create image: 3 rows x 10 cols
        # Row 0: Ground Truth
        # Row 1: Input (past visible, future masked)
        # Row 2: Prediction
        
        cell_size = 32  # Each frame 32x32
        margin = 2
        header = 15
        
        img_height = cell_size * 3 + margin * 4 + header
        img_width = cell_size * len(display_frames) + margin * (len(display_frames) + 1)
        
        img = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Add header
        from PIL import ImageDraw, ImageFont
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        step_text = f"Step {pred_step}/{len(pred_frames_list)-1}"
        draw.text((5, 2), step_text, fill=255)
        draw.text((img_width//2 - 30, 2), "GT | Input | Pred", fill=255)
        
        img = np.array(pil_img)
        
        for col_idx, frame_idx in enumerate(display_frames):
            x = margin + col_idx * (cell_size + margin)
            
            # Row 0: Ground Truth
            y = header + margin
            gt_frame = (gt_tokens[frame_idx].float() / 15 * 255).cpu().numpy().astype(np.uint8)
            img[y:y+cell_size, x:x+cell_size] = gt_frame
            
            # Row 1: Input (past visible, future masked in black)
            y = header + margin * 2 + cell_size
            if frame_idx < num_past:
                input_frame = (gt_tokens[frame_idx].float() / 15 * 255).cpu().numpy().astype(np.uint8)
            else:
                # Check if masked
                if pred_step == 0:
                    input_frame = np.zeros((cell_size, cell_size), dtype=np.uint8)  # Masked
                else:
                    input_frame = (pred_tokens[frame_idx].float() / 15 * 255).cpu().numpy().astype(np.uint8)
            img[y:y+cell_size, x:x+cell_size] = input_frame
            
            # Row 2: Prediction
            y = header + margin * 3 + cell_size * 2
            pred_frame = (pred_tokens[frame_idx].float() / 15 * 255).cpu().numpy().astype(np.uint8)
            img[y:y+cell_size, x:x+cell_size] = pred_frame
        
        gif_frames.append(img)
    
    # Save GIF
    imageio.mimsave(output_path, gif_frames, fps=fps)
    print(f"  Saved GIF: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='outputs/gifs')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--data_path', default='mnist_test_seq.1.npy')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--frame_size', type=int, default=32)
    parser.add_argument('--num_token_levels', type=int, default=16)
    parser.add_argument('--num_past_frames', type=int, default=10)
    parser.add_argument('--num_sampling_steps', type=int, default=20)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = SimpleVideoTransformer(
        vocab_size=args.num_token_levels + 1,
        mask_token_id=args.num_token_levels,
        num_frames=20,
        height=args.frame_size,
        width=args.frame_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.0,
        action_dim=2,
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded Epoch {checkpoint['epoch']}")
    
    # Load dataset
    print("\nLoading validation dataset...")
    val_dataset = MovingMNISTCached(
        data_path=args.data_path,
        seq_len=20,
        num_sequences=500,
        start_idx=2000,
        frame_size=args.frame_size,
        num_token_levels=args.num_token_levels,
        use_ego_centric=True,
        ego_digit_id=0,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {args.num_samples} GIFs...")
    print("="*60)
    
    for i in range(args.num_samples):
        print(f"\nSample {i+1}/{args.num_samples}")
        
        gt_tokens, pred_history = predict_future_gif(
            model, val_dataset, i, args.num_past_frames, device, args.num_sampling_steps
        )
        
        output_path = output_dir / f"prediction_sample_{i:02d}.gif"
        create_comparison_gif(gt_tokens, pred_history, args.num_past_frames, output_path)
    
    print(f"\n{'='*60}")
    print(f"All GIFs saved to: {output_dir}")


if __name__ == '__main__':
    main()
