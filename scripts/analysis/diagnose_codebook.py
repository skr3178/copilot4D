"""Diagnose codebook collapse issues in VQ training."""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from copilot4d.utils.config import TokenizerConfig
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.data.kitti_dataset import KITTITokenizerDataset, tokenizer_collate_fn
from torch.utils.data import DataLoader


def check_codebook_usage(model, loader, device, num_batches=5):
    """Check how many codes are actually being used."""
    model.eval()
    
    all_indices = []
    codebook_size = model.vq.codebook_size
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
                
            features = batch["features"].to(device)
            num_points = batch["num_points"].to(device)
            coords = batch["coords"].to(device)
            batch_size = batch["batch_size"]
            
            # Encode to get indices
            bev = model.encode_voxels(features, num_points, coords, batch_size)
            encoder_out = model.encoder(bev)
            
            # Get VQ indices
            B, N, D = encoder_out.shape
            x_normed = model.vq.pre_norm(encoder_out.float())
            z_e = model.vq.pre_proj(x_normed)
            flat = z_e.reshape(-1, model.vq.codebook_dim)
            
            # Find nearest codes
            z_e_sq = (flat ** 2).sum(dim=1, keepdim=True)
            e_sq = (model.vq.embed ** 2).sum(dim=1, keepdim=True)
            dist = z_e_sq - 2.0 * flat @ model.vq.embed.t() + e_sq.t()
            indices = dist.argmin(dim=1)
            
            all_indices.append(indices.cpu())
    
    # Analyze codebook usage
    all_indices = torch.cat(all_indices)
    unique_codes = torch.unique(all_indices)
    usage_count = torch.bincount(all_indices, minlength=codebook_size)
    
    # Perplexity: exp(-sum(p * log(p)))
    probs = usage_count.float() / usage_count.sum()
    probs = probs[probs > 0]  # Remove zeros for log
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs)))
    
    print(f"\n{'='*60}")
    print(f"CODEBOOK USAGE ANALYSIS")
    print(f"{'='*60}")
    print(f"Codebook size: {codebook_size}")
    print(f"Active codes: {len(unique_codes)} ({100*len(unique_codes)/codebook_size:.1f}%)")
    print(f"Codebook perplexity: {perplexity:.1f} / {codebook_size}")
    print(f"Entropy: {torch.log(perplexity):.2f} nats")
    
    # Find dead codes
    dead_codes = (usage_count == 0).sum().item()
    print(f"Dead codes (0 usage): {dead_codes} ({100*dead_codes/codebook_size:.1f}%)")
    
    # Top used codes
    top_k = 10
    top_indices = torch.topk(usage_count, top_k).indices
    print(f"\nTop {top_k} most used codes:")
    for idx in top_indices:
        print(f"  Code {idx}: {usage_count[idx]} times ({100*usage_count[idx]/all_indices.numel():.1f}%)")
    
    print(f"\nUsage distribution:")
    print(f"  Mean: {usage_count.float().mean():.1f}")
    print(f"  Std: {usage_count.float().std():.1f}")
    print(f"  Max: {usage_count.max()}")
    print(f"  Min (non-zero): {usage_count[usage_count > 0].min()}")
    
    return {
        "active_codes": len(unique_codes),
        "perplexity": perplexity.item(),
        "dead_codes": dead_codes,
    }


def check_gradient_flow(model, batch, device):
    """Check if gradients flow through VQ layer properly."""
    model.train()
    
    features = batch["features"].to(device)
    num_points = batch["num_points"].to(device)
    coords = batch["coords"].to(device)
    batch_size = batch["batch_size"]
    ray_origins = batch["ray_origins"].to(device)
    ray_directions = batch["ray_directions"].to(device)
    ray_depths = batch["ray_depths"].to(device)
    gt_occupancy = batch["gt_occupancy"].to(device)
    
    # Forward pass
    outputs = model(
        features=features,
        num_points=num_points,
        coords=coords,
        batch_size=batch_size,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        gt_depths=ray_depths,
        gt_occupancy=gt_occupancy,
    )
    
    # Backward on VQ loss
    vq_loss = outputs["vq_loss"]
    vq_loss.backward(retain_graph=True)
    
    print(f"\n{'='*60}")
    print(f"GRADIENT FLOW CHECK (VQ loss only)")
    print(f"{'='*60}")
    
    # Check encoder layers
    encoder_grads = {}
    for name, param in model.encoder.named_parameters():
        if param.grad is not None:
            encoder_grads[name] = param.grad.abs().mean().item()
    
    if encoder_grads:
        print(f"\nEncoder gradient norms:")
        for name, grad_norm in list(encoder_grads.items())[:5]:
            print(f"  {name}: {grad_norm:.6f}")
        print(f"  ... ({len(encoder_grads)} total layers)")
    else:
        print("WARNING: No gradients in encoder!")
    
    # Check pre/post projection
    print(f"\nVQ projection gradients:")
    for name, param in model.vq.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad.abs().mean().item():.6f}")
        else:
            print(f"  {name}: NO GRADIENT!")
    
    model.zero_grad()
    
    # Now check total loss
    total_loss = outputs["losses"]["total"]
    total_loss.backward()
    
    print(f"\n{'='*60}")
    print(f"GRADIENT FLOW CHECK (Total loss)")
    print(f"{'='*60}")
    
    encoder_grads_total = {}
    for name, param in model.encoder.named_parameters():
        if param.grad is not None:
            encoder_grads_total[name] = param.grad.abs().mean().item()
    
    if encoder_grads_total:
        avg_grad = sum(encoder_grads_total.values()) / len(encoder_grads_total)
        print(f"Average encoder gradient norm: {avg_grad:.6f}")
        
        # Find any zero gradients
        zero_grads = [name for name, g in encoder_grads_total.items() if g < 1e-10]
        if zero_grads:
            print(f"WARNING: {len(zero_grads)} layers with near-zero gradients")
    
    model.zero_grad()


def check_memory_bank(model):
    """Check memory bank state."""
    print(f"\n{'='*60}")
    print(f"MEMORY BANK STATE")
    print(f"{'='*60}")
    
    bank = model.vq.memory_bank
    bank_ptr = model.vq.bank_ptr.item()
    
    # Check how much of the bank is filled
    filled = (bank.abs().sum(dim=1) > 0).sum().item()
    total = bank.shape[0]
    
    print(f"Memory bank size: {total}")
    print(f"Filled entries: {filled} ({100*filled/total:.1f}%)")
    print(f"Bank pointer: {bank_ptr}")
    
    if filled > 0:
        # Check variance of stored samples
        filled_data = bank[:filled] if filled < total else bank
        print(f"Stored samples shape: {filled_data.shape}")
        print(f"Mean norm: {filled_data.norm(dim=1).mean():.4f}")
        print(f"Std norm: {filled_data.norm(dim=1).std():.4f}")


def main():
    print("Loading model and data...")
    
    cfg = TokenizerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = CoPilot4DTokenizer(cfg).to(device)
    
    # Try to load checkpoint if exists
    checkpoint_path = Path("outputs/tokenizer/checkpoint_step_5000.pt")
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Resumed from step {checkpoint['step']}")
    else:
        print("No checkpoint found, using randomly initialized model")
    
    # Create dataloader
    dataset = KITTITokenizerDataset(cfg, split="val")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=tokenizer_collate_fn,
    )
    
    # Run diagnostics
    batch = next(iter(loader))
    
    check_codebook_usage(model, loader, device)
    check_gradient_flow(model, batch, device)
    check_memory_bank(model)
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
