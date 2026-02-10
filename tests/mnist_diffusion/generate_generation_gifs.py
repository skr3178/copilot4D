#!/usr/bin/env python3
"""
Generate animated GIFs for pure generation samples (no conditioning on past frames).
Shows frames evolving over time.
"""

import torch
import numpy as np
import imageio
import os
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from simple_model import SimpleVideoTransformer


def tokens_to_rgb(tokens, num_levels=16):
    """Convert discrete tokens back to RGB frames."""
    # tokens: [T, H, W]
    normalized = tokens.float() / (num_levels - 1)
    rgb = normalized.unsqueeze(-1).repeat(1, 1, 1, 3)  # [T, H, W, 3]
    rgb = (rgb * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
    return rgb


def create_generation_gif(model, device, num_frames=20, height=32, width=32, 
                          num_sampling_steps=20, num_gifs=5, output_dir="generation_gifs"):
    """Generate pure generation samples and save as animated GIFs."""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    vocab_size = model.vocab_size  # 17
    mask_token_id = vocab_size  # 16
    
    print(f"\nGenerating {num_gifs} pure generation GIFs...")
    print(f"  Sampling steps: {num_sampling_steps}")
    print(f"  Frames: {num_frames}, Resolution: {height}x{width}")
    
    all_gifs = []
    
    with torch.no_grad():
        for gif_idx in tqdm(range(num_gifs), desc="Generating samples"):
            with autocast():
                # Start with all MASK tokens
                tokens = torch.full((1, num_frames, height, width), 
                                   mask_token_id, dtype=torch.long, device=device)
                
                # Dummy actions (no conditioning) - ego-centric action dim is 2
                actions = torch.zeros((1, num_frames, 2), device=device)
                
                # Iterative denoising schedule (cosine)
                u_values = np.linspace(0, 1, num_sampling_steps)
                mask_ratios = np.cos(u_values * np.pi / 2)  # 1 -> 0
                
                # Iterative denoising
                for step in range(num_sampling_steps):
                    logits = model(tokens, actions, None)
                    # logits: (B, T, H*W, V)
                    B, T, N, V = logits.shape
                    probs = torch.softmax(logits, dim=-1)
                    
                    flat_probs = probs.reshape(-1, V)
                    samples = torch.multinomial(flat_probs, num_samples=1).reshape(B, T, N)
                    samples = samples.reshape(B, T, height, width)
                    
                    mask = (tokens == mask_token_id)
                    tokens = torch.where(mask, samples, tokens)
                
                # Final unmasking
                if (tokens == mask_token_id).any():
                    logits = model(tokens, actions, None)
                    probs = torch.softmax(logits, dim=-1)
                    B, T, N, V = logits.shape
                    samples = torch.multinomial(probs.reshape(-1, V), num_samples=1).reshape(B, T, N)
                    samples = samples.reshape(B, T, height, width)
                    mask = (tokens == mask_token_id)
                    tokens = torch.where(mask, samples, tokens)
                
                all_gifs.append(tokens[0].cpu())
    
    # Create GIFs showing temporal evolution
    print("\nCreating GIFs...")
    for gif_idx, tokens in enumerate(all_gifs):
        frames_for_gif = []
        
        for t in range(num_frames):
            # Create frame visualization
            frame_rgb = tokens_to_rgb(tokens[t:t+1], vocab_size - 1)[0]  # [H, W, 3]
            
            # Scale up for visibility (4x)
            from PIL import Image
            h, w = frame_rgb.shape[:2]
            frame_scaled = np.array(Image.fromarray(frame_rgb).resize((w*8, h*8), Image.NEAREST))
            
            # Add padding and color bar
            bar_height = 20
            frame_with_bar = np.zeros((frame_scaled.shape[0] + bar_height, 
                                       frame_scaled.shape[1], 3), dtype=np.uint8)
            frame_with_bar[bar_height:, :] = frame_scaled
            
            # Color bar - all frames are "generated" (yellow)
            frame_with_bar[:bar_height, :] = [255, 200, 0]  # Yellow
            
            # Try to add frame number
            try:
                pil_img = Image.fromarray(frame_with_bar)
                draw = ImageDraw.Draw(pil_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                except:
                    font = ImageFont.load_default()
                text = f"Frame {t:2d} (GEN)"
                draw.text((5, 2), text, fill=(0, 0, 0), font=font)
                frame_with_bar = np.array(pil_img)
            except:
                pass
            
            frames_for_gif.append(frame_with_bar)
        
        # Save GIF
        output_path = os.path.join(output_dir, f"generation_sample_{gif_idx:03d}.gif")
        imageio.mimsave(output_path, frames_for_gif, fps=5, loop=0)
        print(f"  Saved: {output_path}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate animated GIFs for pure generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/mnist_diffusion_fast/generation_gifs", 
                       help="Output directory for GIFs")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--frame_size", type=int, default=32)
    parser.add_argument("--num_token_levels", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--num_gifs", type=int, default=5, help="Number of GIFs to generate")
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    print(f"Creating model (dim={args.embed_dim}, layers={args.num_layers}, heads={args.num_heads})...")
    model = SimpleVideoTransformer(
        vocab_size=args.num_token_levels + 1,
        mask_token_id=args.num_token_levels,
        num_frames=args.num_frames,
        height=args.frame_size,
        width=args.frame_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        action_dim=2,
    ).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"\nCheckpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    # Import PIL here
    global Image, ImageDraw, ImageFont
    from PIL import Image, ImageDraw, ImageFont
    
    output_dir = create_generation_gif(
        model, device,
        num_frames=args.num_frames,
        height=args.frame_size,
        width=args.frame_size,
        num_sampling_steps=args.num_sampling_steps,
        num_gifs=args.num_gifs,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*60}")
    print(f"Generation GIFs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
