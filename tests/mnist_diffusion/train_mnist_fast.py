#!/usr/bin/env python
"""Fast training with cached dataset and smaller subset."""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_mnist_full import SimpleVideoTransformer, cosine_mask_schedule
from moving_mnist_cached import create_cached_dataloaders


class SimpleConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def train_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, epoch, writer):
    """Train one epoch."""
    model.train()
    
    total_loss = 0
    total_samples = 0
    total_correct = 0
    total_tokens = 0
    
    for batch_idx, batch in enumerate(train_loader):
        tokens = batch["tokens"].to(cfg.device)
        actions = batch["actions"].to(cfg.device)
        
        B, T, H, W = tokens.shape
        N = H * W
        num_past = cfg.num_past_frames
        
        # Random masking
        mask_ratio = np.random.uniform(0.5, 1.0)
        masked_tokens = tokens.clone()
        mask = torch.rand(B, T - num_past, H, W, device=cfg.device) < mask_ratio
        masked_tokens[:, num_past:][mask] = cfg.vocab_size
        
        # Causal mask
        causal_mask = torch.full((T, T), float("-inf"), device=cfg.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=cfg.use_amp):
            logits = model(masked_tokens, actions, causal_mask)
            
            future_logits = logits[:, num_past:]
            future_targets = tokens[:, num_past:]
            
            B_f, T_f, N_f, V = future_logits.shape
            future_logits_flat = future_logits.reshape(B_f * T_f * N_f, V)
            future_targets_flat = future_targets.reshape(-1)
            
            loss = F.cross_entropy(future_logits_flat, future_targets_flat)
        
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
        
        with torch.no_grad():
            pred = future_logits.argmax(dim=-1)
            future_targets_flat = future_targets.reshape(B, T - num_past, N)
            correct = (pred == future_targets_flat).sum().item()
            total = future_targets_flat.numel()
        
        total_loss += loss.item() * B
        total_samples += B
        total_correct += correct
        total_tokens += total
        
        if batch_idx % 50 == 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_tokens
            lr = scheduler.get_last_lr()[0]
            print(f"  Batch {batch_idx:3d}/{len(train_loader)} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.3f} | LR: {lr:.6f}")
            
            # Log to tensorboard
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', avg_loss, step)
            writer.add_scalar('train/acc', avg_acc, step)
    
    return total_loss / total_samples, total_correct / total_tokens


@torch.no_grad()
def validate(model, val_loader, cfg):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    
    for batch in val_loader:
        tokens = batch["tokens"].to(cfg.device)
        actions = batch["actions"].to(cfg.device)
        
        B, T, H, W = tokens.shape
        N = H * W
        num_past = cfg.num_past_frames
        
        masked_tokens = tokens.clone()
        masked_tokens[:, num_past:] = cfg.vocab_size
        
        causal_mask = torch.full((T, T), float("-inf"), device=cfg.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        
        logits = model(masked_tokens, actions, causal_mask)
        
        future_logits = logits[:, num_past:]
        future_targets = tokens[:, num_past:]
        
        B_f, T_f, N_f, V = future_logits.shape
        future_logits_flat = future_logits.reshape(B_f * T_f * N_f, V)
        future_targets_flat = future_targets.reshape(-1)
        
        loss = F.cross_entropy(future_logits_flat, future_targets_flat, reduction='sum')
        
        pred = future_logits.argmax(dim=-1)
        future_targets_reshaped = future_targets.reshape(B_f, T_f, N_f)
        correct = (pred == future_targets_reshaped).sum().item()
        
        total_loss += loss.item()
        total_tokens += future_targets_flat.numel()
        total_correct += correct
    
    return total_loss / total_tokens, total_correct / total_tokens


def main():
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--frame_size', type=int, default=32)
    parser.add_argument('--num_token_levels', type=int, default=16)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_past_frames', type=int, default=10)
    
    # Data (smaller subset for fast iteration)
    parser.add_argument('--num_train', type=int, default=8000, help='Number of training sequences (default: 8000, full=8000)')
    parser.add_argument('--num_val', type=int, default=2000, help='Number of validation sequences (default: 2000, full=2000)')
    parser.add_argument('--data_path', default='mnist_test_seq.1.npy')
    parser.add_argument('--cache_dir', default='data/mnist_cache_full', help='Cache directory for precomputed actions (default: data/mnist_cache_full for full 10k dataset)')
    parser.add_argument('--output_dir', default='outputs/mnist_diffusion_fast')
    parser.add_argument('--ego_digit_id', type=int, default=0)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders with cached dataset
    print(f"\nLoading data ({args.num_train} train, {args.num_val} val)...")
    train_loader, val_loader = create_cached_dataloaders(
        data_path=args.data_path,
        seq_len=20,
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        num_token_levels=args.num_token_levels,
        frame_size=args.frame_size,
        use_ego_centric=True,
        ego_digit_id=args.ego_digit_id,
        cache_dir=args.cache_dir,
    )
    
    cfg = SimpleConfig(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        frame_size=args.frame_size,
        num_token_levels=args.num_token_levels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_past_frames=args.num_past_frames,
        use_amp=True,
        device=device,
        vocab_size=args.num_token_levels + 1,
    )
    
    # Model
    print(f"\nCreating model:")
    print(f"  Embed dim: {cfg.embed_dim}")
    print(f"  Layers: {cfg.num_layers}")
    print(f"  Heads: {cfg.num_heads}")
    
    model = SimpleVideoTransformer(
        vocab_size=cfg.vocab_size,
        mask_token_id=cfg.vocab_size,
        num_frames=20,
        height=cfg.frame_size,
        width=cfg.frame_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=0.1,
        action_dim=2,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params/1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp)
    
    writer = SummaryWriter(output_dir / 'logs')
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print("Training with cached dataset")
    print(f"{'='*60}\n")
    
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        print('-' * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, cfg)
        
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.3f}")
        print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.3f}")
        
        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        writer.add_scalar('epoch/val_loss', val_loss, epoch)
        writer.add_scalar('epoch/train_acc', train_acc, epoch)
        writer.add_scalar('epoch/val_acc', val_acc, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model")
        
        if epoch % 10 == 0:
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
