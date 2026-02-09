#!/usr/bin/env python3
"""Compare two checkpoint files to analyze codebook usage metrics."""

import torch
import sys
from pathlib import Path
import numpy as np
import yaml

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from copilot4d.utils.config import TokenizerConfig
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.data.kitti_dataset import KITTITokenizerDataset, tokenizer_collate_fn
from torch.utils.data import DataLoader


def load_config_from_yaml(yaml_path):
    """Load config from YAML file matching checkpoint training config."""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create TokenizerConfig with values from YAML
    cfg = TokenizerConfig()
    for key, value in config_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    
    return cfg


def analyze_checkpoint(checkpoint_path, device, num_batches=10):
    """Analyze a checkpoint for codebook usage and metrics."""
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {checkpoint_path.name}")
    print(f"{'='*70}")
    
    # Load the config that matches the checkpoint
    config_path = Path("/media/skr/storage/self_driving/CoPilot4D/configs/tokenizer_memory_efficient.yaml")
    cfg = load_config_from_yaml(config_path)
    print(f"Config: {config_path.name}")
    model = CoPilot4DTokenizer(cfg).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    step = checkpoint.get("step", "unknown")
    
    print(f"Checkpoint step: {step}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    model.eval()
    
    # Create dataloader
    dataset = KITTITokenizerDataset(cfg, split="val")
    loader = DataLoader(
        dataset,
        batch_size=1,  # Reduced to avoid OOM
        shuffle=False,
        num_workers=0,
        collate_fn=tokenizer_collate_fn,
    )
    
    # Collect all indices
    all_indices = []
    codebook_size = model.vq.codebook_size
    total_tokens = 0
    
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
            total_tokens += indices.numel()
    
    # Analyze codebook usage
    all_indices = torch.cat(all_indices)
    unique_codes = torch.unique(all_indices)
    usage_count = torch.bincount(all_indices, minlength=codebook_size)
    
    # Perplexity
    probs = usage_count.float() / usage_count.sum()
    probs = probs[probs > 0]
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs)))
    entropy = torch.log(perplexity)
    
    # Dead codes
    dead_codes = (usage_count == 0).sum().item()
    
    # Usage statistics
    usage_nonzero = usage_count[usage_count > 0]
    
    results = {
        "step": step,
        "active_codes": len(unique_codes),
        "active_pct": 100 * len(unique_codes) / codebook_size,
        "dead_codes": dead_codes,
        "dead_pct": 100 * dead_codes / codebook_size,
        "perplexity": perplexity.item(),
        "entropy": entropy.item(),
        "codebook_size": codebook_size,
        "total_tokens": total_tokens,
        "usage_mean": usage_count.float().mean().item(),
        "usage_std": usage_count.float().std().item(),
        "usage_min": usage_count.min().item(),
        "usage_max": usage_count.max().item(),
        "usage_median": usage_count.float().median().item(),
        "top_10_usage_pct": 100 * torch.topk(usage_count, 10).values.sum().item() / total_tokens,
        "bottom_50_usage_pct": 100 * torch.topk(usage_count, 50, largest=False).values.sum().item() / total_tokens,
    }
    
    # Print summary
    print(f"\n📊 CODEBOOK USAGE SUMMARY")
    print(f"  Codebook size: {codebook_size}")
    print(f"  Active codes: {results['active_codes']} ({results['active_pct']:.1f}%)")
    print(f"  Dead codes: {results['dead_codes']} ({results['dead_pct']:.1f}%)")
    print(f"  Perplexity: {results['perplexity']:.1f} / {codebook_size}")
    print(f"  Entropy: {results['entropy']:.2f} nats")
    print(f"  Normalized entropy: {results['entropy']/np.log(codebook_size):.3f}")
    
    print(f"\n📈 USAGE DISTRIBUTION")
    print(f"  Mean: {results['usage_mean']:.1f}")
    print(f"  Median: {results['usage_median']:.1f}")
    print(f"  Std: {results['usage_std']:.1f}")
    print(f"  Min: {results['usage_min']}")
    print(f"  Max: {results['usage_max']}")
    
    print(f"\n🔝 CODE CONCENTRATION")
    print(f"  Top 10 codes usage: {results['top_10_usage_pct']:.1f}% of all tokens")
    print(f"  Bottom 50 codes usage: {results['bottom_50_usage_pct']:.2f}% of all tokens")
    
    # Memory bank check
    bank = model.vq.memory_bank
    bank_ptr = model.vq.bank_ptr.item()
    filled = (bank.abs().sum(dim=1) > 0).sum().item()
    
    print(f"\n💾 MEMORY BANK")
    print(f"  Filled: {filled} / {bank.shape[0]} ({100*filled/bank.shape[0]:.1f}%)")
    print(f"  Bank pointer: {bank_ptr}")
    
    return results, model


def compare_checkpoints(results_22k, results_30k):
    """Print side-by-side comparison of two checkpoints."""
    
    print(f"\n{'='*70}")
    print(f"COMPARISON: Step {results_22k['step']} vs Step {results_30k['step']}")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<30} {'Step 22,000':>15} {'Step 30,000':>15} {'Change':>10}")
    print("-" * 70)
    
    metrics = [
        ("Active codes (%)", "active_pct", "%.1f%%"),
        ("Dead codes (%)", "dead_pct", "%.1f%%"),
        ("Perplexity", "perplexity", "%.1f"),
        ("Entropy (nats)", "entropy", "%.2f"),
        ("Usage mean", "usage_mean", "%.1f"),
        ("Usage std", "usage_std", "%.1f"),
        ("Top 10 codes (% tokens)", "top_10_usage_pct", "%.1f%%"),
        ("Bottom 50 codes (% tokens)", "bottom_50_usage_pct", "%.2f%%"),
    ]
    
    for name, key, fmt in metrics:
        v1 = results_22k[key]
        v2 = results_30k[key]
        change = v2 - v1
        
        # Format values
        if "%%" in fmt:
            s1 = fmt % v1
            s2 = fmt % v2
            s_change = f"{change:+.1f}%"
        else:
            s1 = fmt % v1
            s2 = fmt % v2
            s_change = f"{change:+.2f}"
        
        print(f"{name:<30} {s1:>15} {s2:>15} {s_change:>10}")
    
    # Interpretation
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")
    
    active_diff = results_30k['active_pct'] - results_22k['active_pct']
    dead_diff = results_30k['dead_pct'] - results_22k['dead_pct']
    entropy_diff = results_30k['entropy'] - results_22k['entropy']
    
    if active_diff > 0:
        print(f"✅ Codebook utilization IMPROVED by {active_diff:.1f}%")
    else:
        print(f"⚠️  Codebook utilization DECREASED by {abs(active_diff):.1f}%")
    
    if dead_diff < 0:
        print(f"✅ Dead codes REDUCED by {abs(dead_diff):.1f}%")
    else:
        print(f"⚠️  Dead codes INCREASED by {dead_diff:.1f}%")
    
    if entropy_diff > 0:
        print(f"✅ Entropy INCREASED by {entropy_diff:.2f} nats (more uniform)")
    else:
        print(f"⚠️  Entropy DECREASED by {abs(entropy_diff):.2f} nats (less uniform)")
    
    # Perplexity ratio
    ppl_ratio = results_30k['perplexity'] / results_22k['perplexity']
    print(f"\n📊 Perplexity ratio (30k/22k): {ppl_ratio:.2f}x")
    
    if ppl_ratio > 1.1:
        print("   → Significant improvement in code diversity")
    elif ppl_ratio < 0.9:
        print("   → Code diversity decreased")
    else:
        print("   → Code diversity relatively stable")


def main():
    # Use CPU to avoid OOM with other processes on GPU
    device = torch.device("cpu")
    print(f"Using device: {device} (avoiding GPU OOM)")
    print(f"Using device: {device}")
    
    # Checkpoint paths
    checkpoint_dir = Path("/media/skr/storage/self_driving/CoPilot4D/outputs/tokenizer_memory_efficient")
    ckpt_22k = checkpoint_dir / "checkpoint_step_22000.pt"
    ckpt_30k = checkpoint_dir / "checkpoint_step_30000.pt"
    
    if not ckpt_22k.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_22k}")
        return
    if not ckpt_30k.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_30k}")
        return
    
    print(f"\nFound checkpoints:")
    print(f"  - {ckpt_22k.name}")
    print(f"  - {ckpt_30k.name}")
    
    # Analyze both checkpoints
    results_22k, model_22k = analyze_checkpoint(ckpt_22k, device)
    results_30k, model_30k = analyze_checkpoint(ckpt_30k, device)
    
    # Compare
    compare_checkpoints(results_22k, results_30k)
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
