#!/usr/bin/env python
"""Generate continuous video GIFs comparing GT vs Prediction."""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_mnist_full import SimpleVideoTransformer
from moving_mnist_cached import MovingMNISTCached


def create_video_comparison_gif(gt_tokens, pred_tokens, num_past, output_path, fps=10):
    """Create a GIF showing the video playing with GT vs Pred side by side."""
    T, H, W = gt_tokens.shape
    
    gif_frames = []
    
    # For each time step, create a frame showing:
    # - Ground truth at that timestep
    # - Prediction at that timestep
    # - Label showing if it's "Past" (given) or "Future" (predicted)
    
    cell_size = 64  # Upscale for visibility
    margin = 5
    label_height = 25
    
    img_width = cell_size * 2 + margin * 3
    img_height = cell_size + label_height + margin * 2
    
    for t in range(T):
        # Create image
        img_array = np.zeros((img_height, img_width), dtype=np.uint8)
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        
        # Ground truth frame (left)
        gt_frame = (gt_tokens[t].float() / 15 * 255).cpu().numpy().astype(np.uint8)
        gt_pil = Image.fromarray(gt_frame).resize((cell_size, cell_size), Image.NEAREST)
        img.paste(gt_pil, (margin, margin + label_height))
        
        # Prediction frame (right)
        pred_frame = (pred_tokens[t].float() / 15 * 255).cpu().numpy().astype(np.uint8)
        pred_pil = Image.fromarray(pred_frame).resize((cell_size, cell_size), Image.NEAREST)
        img.paste(pred_pil, (margin * 2 + cell_size, margin + label_height))
        
        # Labels
        label_color = 255
        draw.text((margin + 15, 5), "Ground Truth", fill=label_color)
        draw.text((margin * 2 + cell_size + 20, 5), "Prediction", fill=label_color)
        
        # Time step indicator
        if t < num_past:
            status = f"Frame {t:2d} [PAST - Given]"
            status_color = (0, 255, 0)  # Green for past
        else:
            status = f"Frame {t:2d} [FUTURE - Predicted]"
            status_color = (255, 255, 0)  # Yellow for future
        
        # Convert to RGB for colored text
        img_rgb = img.convert('RGB')
        draw_rgb = ImageDraw.Draw(img_rgb)
        
        # Add progress bar at bottom
        bar_y = img_height - 8
        progress = (t + 1) / T
        bar_width = int((img_width - margin * 2) * progress)
        draw_rgb.rectangle([margin, bar_y, margin + bar_width, bar_y + 4], 
                          fill=(0, 200, 0) if t < num_past else (200, 200, 0))
        
        # Add frame number at top center
        frame_text = f"Frame {t}/{T-1}"
        draw_rgb.text((img_width//2 - 30, 3), frame_text, fill=(255, 255, 255))
        
        # Add separator line
        sep_x = margin * 2 + cell_size
        draw_rgb.line([(sep_x - 2, label_height), (sep_x - 2, img_height - 10)], 
                     fill=(100, 100, 100), width=2)
        
        gif_frames.append(np.array(img_rgb))
    
    # Save GIF
    imageio.mimsave(output_path, gif_frames, fps=fps)
    print(f"  Saved: {output_path}")


@torch.no_grad()
def predict_future(model, dataset, sample_idx, num_past_frames, device, num_steps=20):
    """Generate future prediction."""
    model.eval()
    
    sample = dataset[sample_idx]
    tokens = sample["tokens"].to(device)
    actions = sample["actions"].to(device)
    
    T, H, W = tokens.shape
    
    # Prepare input
    input_tokens = tokens.clone()
    input_tokens[num_past_frames:] = 16  # mask token
    
    # Causal mask
    causal_mask = torch.full((T, T), float("-inf"), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    # Iterative denoising
    current_tokens = input_tokens.clone().unsqueeze(0)
    actions_batch = actions.unsqueeze(0)
    
    for step in range(num_steps):
        logits = model(current_tokens, actions_batch, causal_mask)
        probs = F.softmax(logits[0, num_past_frames:], dim=-1)
        pred_tokens_future = torch.argmax(probs, dim=-1)
        pred_tokens_future = pred_tokens_future.reshape(T - num_past_frames, H, W)
        current_tokens[0, num_past_frames:] = pred_tokens_future
    
    return tokens, current_tokens[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='outputs/video_gifs')
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
    epoch = checkpoint['epoch']
    val_loss = checkpoint.get('val_loss', 0)
    print(f"  Loaded Epoch {epoch} (Val Loss: {val_loss:.4f})")
    
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
    
    # Create output directory with epoch info
    output_dir = Path(args.output_dir) / f"epoch{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {args.num_samples} video GIFs...")
    print("="*60)
    
    for i in range(args.num_samples):
        print(f"Sample {i+1}/{args.num_samples}")
        
        gt_tokens, pred_tokens = predict_future(
            model, val_dataset, i, args.num_past_frames, device, args.num_sampling_steps
        )
        
        output_path = output_dir / f"video_sample_{i:02d}.gif"
        create_video_comparison_gif(gt_tokens, pred_tokens, args.num_past_frames, output_path)
    
    print(f"\n{'='*60}")
    print(f"All GIFs saved to: {output_dir}")
    print(f"\nFiles generated:")
    for f in sorted(output_dir.glob("*.gif")):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
