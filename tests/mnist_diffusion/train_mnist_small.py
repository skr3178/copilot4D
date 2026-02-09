#!/usr/bin/env python
"""
Smaller model training for Moving MNIST with improved settings.
Optimized for faster convergence with limited data.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_mnist_full import (
    SimpleVideoTransformer, DiscreteDiffusionMasker,
    MovingMNISTPrecomputed, Config
)


def create_small_model(vocab_size=16, embed_dim=192, num_layers=4, num_heads=4, 
                       frame_size=32, num_frames=20):
    """Create a smaller model for faster training."""
    return SimpleVideoTransformer(
        vocab_size=vocab_size,
        mask_token_id=vocab_size,
        num_frames=num_frames,
        height=frame_size,
        width=frame_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,  # Add dropout for regularization
    )


def train_epoch_small(model, train_loader, masker, optimizer, scheduler, scaler, cfg, epoch, writer):
    """Train one epoch with simplified approach."""
    model.train()
    
    total_loss = 0
    total_samples = 0
    
    pbar = enumerate(train_loader)
    
    for batch_idx, batch in pbar:
        # Get tokens
        tokens = batch["tokens"].to(cfg.device)
        actions = batch["actions"].to(cfg.device)
        
        B, T, H, W = tokens.shape
        N = H * W
        
        # Simplified: Always do future prediction (most important objective)
        num_past = cfg.num_past_frames
        
        # Random masking schedule for future
        mask_ratio = np.random.uniform(0.5, 1.0)
        
        # Create masked version
        masked_tokens = tokens.clone()
        mask = torch.rand(B, T - num_past, H, W, device=cfg.device) < mask_ratio
        masked_tokens[:, num_past:][mask] = cfg.vocab_size  # mask token
        
        # Create targets (only compute loss on masked positions)
        targets = tokens[:, num_past:].reshape(B, (T - num_past) * N)
        
        # Forward
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            # Flatten for model
            tokens_flat = masked_tokens.reshape(B, T * N)
            logits = model(tokens_flat)  # [B, T*N, vocab_size]
            logits = logits.reshape(B, T, N, cfg.vocab_size)
            
            # Only compute loss on future masked positions
            future_logits = logits[:, num_past:]  # [B, T_future, N, vocab_size]
            future_logits_flat = future_logits.reshape(B * (T - num_past) * N, cfg.vocab_size)
            targets_flat = targets.reshape(-1)
            
            # Only compute loss on masked positions
            mask_flat = mask.reshape(-1)
            if mask_flat.sum() > 0:
                loss = F.cross_entropy(
                    future_logits_flat[mask_flat],
                    targets_flat[mask_flat],
                    reduction='mean'
                )
            else:
                continue
        
        # Backward
        if cfg.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item() * B
        total_samples += B
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            avg_loss = total_loss / total_samples
            lr = scheduler.get_last_lr()[0]
            print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
    
    return total_loss / total_samples


@torch.no_grad()
def validate_small(model, val_loader, cfg):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    for batch in val_loader:
        tokens = batch["tokens"].to(cfg.device)
        
        B, T, H, W = tokens.shape
        N = H * W
        num_past = cfg.num_past_frames
        
        # Mask all future
        masked_tokens = tokens.clone()
        masked_tokens[:, num_past:] = cfg.vocab_size
        
        # Forward
        tokens_flat = masked_tokens.reshape(B, T * N)
        logits = model(tokens_flat)
        logits = logits.reshape(B, T, N, cfg.vocab_size)
        
        # Loss on future only
        future_logits = logits[:, num_past:]
        targets = tokens[:, num_past:].reshape(-1)
        future_logits_flat = future_logits.reshape(-1, cfg.vocab_size)
        
        loss = F.cross_entropy(future_logits_flat, targets, reduction='sum')
        
        total_loss += loss.item()
        total_samples += B * (T - num_past) * N
    
    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser()
    
    # Model config (SMALL)
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--frame_size', type=int, default=32)
    parser.add_argument('--num_token_levels', type=int, default=16)
    
    # Training config
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_past_frames', type=int, default=10)
    
    # Data
    parser.add_argument('--data_path', default='mnist_test_seq.1.npy')
    parser.add_argument('--output_dir', default='outputs/mnist_diffusion_small')
    
    args = parser.parse_args()
    
    cfg = Config(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        frame_size=args.frame_size,
        num_token_levels=args.num_token_levels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_past_frames=args.num_past_frames,
    )
    cfg.vocab_size = args.num_token_levels + 1  # +1 for mask token
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.device = device
    print(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    print(f"\nCreating small model:")
    print(f"  Embed dim: {cfg.embed_dim}")
    print(f"  Layers: {cfg.num_layers}")
    print(f"  Heads: {cfg.num_heads}")
    
    model = create_small_model(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        frame_size=cfg.frame_size,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params/1e6:.2f}M")
    
    # Data
    train_dataset = MovingMNISTPrecomputed(
        data_path=args.data_path,
        seq_len=20,
        frame_size=cfg.frame_size,
        num_token_levels=cfg.num_token_levels,
        num_sequences=8000,
        start_idx=0,
    )
    
    val_dataset = MovingMNISTPrecomputed(
        data_path=args.data_path,
        seq_len=20,
        frame_size=cfg.frame_size,
        num_token_levels=cfg.num_token_levels,
        num_sequences=2000,
        start_idx=8000,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    # Optimizer with cosine schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    masker = DiscreteDiffusionMasker(cfg.vocab_size - 1)
    
    writer = SummaryWriter(output_dir / 'logs')
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print("Training small model")
    print(f"{'='*60}\n")
    
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        print('-' * 40)
        
        # Train
        train_loss = train_epoch_small(
            model, train_loader, masker, optimizer, scheduler, scaler, cfg, epoch, writer
        )
        
        # Validate
        val_loss = validate_small(model, val_loader, cfg)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Regular checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, output_dir / f'checkpoint_epoch{epoch}.pt')
    
    writer.close()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
