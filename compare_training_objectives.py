#!/usr/bin/env python3
"""
Detailed comparison of the 3 training objectives in Algorithm 1.

Shows:
1. What the model sees (input/masked tokens)
2. What the model needs to predict (targets)
3. Temporal attention patterns
4. Loss computation
5. Code implementation details
"""

import sys
sys.path.insert(0, '/media/skr/storage/self_driving/CoPilot4D')

import torch
import torch.nn.functional as F
import random
import math
from typing import Dict, List

from copilot4d.world_model.masking import (
    DiscreteDiffusionMasker,
    cosine_mask_schedule,
    compute_diffusion_loss
)
from copilot4d.utils.config import WorldModelConfig


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print("\n" + char*80)
    print(f" {title}")
    print(char*80)


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n{'─'*60}")
    print(f" {title}")
    print(f"{'─'*60}")


def print_tensor(tokens: torch.Tensor, title: str = ""):
    """Print tensor with formatting."""
    if title:
        print(f"\n{title}")
    H, W = tokens.shape[-2:]
    for i in range(H):
        row = []
        for j in range(W):
            val = tokens[..., i, j].item() if tokens.dim() >= 2 else tokens[i*W + j].item()
            if val == 1024:
                row.append(" M ")
            elif val == 1:
                row.append(" 1 ")
            else:
                row.append(f"{val:2d}")
        print("  [" + "][".join(row) + "]")


def analyze_training_example(objective: str, tokens: torch.Tensor, masker: DiscreteDiffusionMasker, cfg: WorldModelConfig):
    """
    Complete analysis of a training example.
    
    Shows:
    - Input to model (masked tokens)
    - Target (original tokens)
    - Temporal mask
    - What positions contribute to loss
    """
    torch.manual_seed(42)
    random.seed(42)
    
    result = masker.prepare_batch(tokens.clone(), objective=objective)
    
    masked_tokens = result['tokens']      # What model sees
    targets = result['targets']            # What model should predict
    temporal_mask = result['temporal_mask']
    was_masked = result['was_masked']
    
    B, T, H, W = tokens.shape
    
    print_subsection(f"OBJECTIVE: {objective.upper().replace('_', ' ')}")
    
    # Show configuration
    print(f"\n📋 Configuration:")
    print(f"   num_past_frames: {cfg.num_past_frames}")
    print(f"   num_frames: {cfg.num_frames}")
    print(f"   mask_schedule: {cfg.mask_schedule} (γ(u) = cos(u·π/2))")
    print(f"   noise_eta: {cfg.noise_eta}%")
    print(f"   label_smoothing: {cfg.label_smoothing}")
    
    # Show mask ratio used
    print(f"\n📊 Masking Parameters:")
    print(f"   Sampled u₀: {result.get('mask_ratio', 'N/A')}")
    
    # Show each frame
    print(f"\n📝 Frame-by-Frame Analysis:")
    print(f"   Legend: [1] = Ground Truth  [M] = Mask Token (1024)  [xx] = Random Noise")
    
    for t in range(T):
        is_past = t < cfg.num_past_frames
        frame_type = "PAST" if is_past else "FUTURE"
        
        # Count statistics
        gt_count = (masked_tokens[0, t] == targets[0, t]).sum().item()
        mask_count = (masked_tokens[0, t] == 1024).sum().item()
        noise_count = (~(masked_tokens[0, t] == targets[0, t]) & ~(masked_tokens[0, t] == 1024)).sum().item()
        total = H * W
        
        print(f"\n   Frame {t} ({frame_type}):")
        print(f"   ┌──────────────────────────────────────────┐")
        
        # Show input (what model sees)
        print(f"   │ INPUT (what model sees):                 │")
        for i in range(H):
            row_vals = []
            for j in range(W):
                val = masked_tokens[0, t, i, j].item()
                if val == 1024:
                    row_vals.append(" M ")
                elif val == 1:
                    row_vals.append(" 1 ")
                else:
                    row_vals.append(f"{val:2d} ")
            print(f"   │   [{']['.join(row_vals)}] │")
        
        # Show target (what model should predict)
        print(f"   │ TARGET (ground truth):                   │")
        for i in range(H):
            row_vals = []
            for j in range(W):
                val = targets[0, t, i, j].item()
                if val == 1:
                    row_vals.append(" 1 ")
                else:
                    row_vals.append(f"{val:2d} ")
            print(f"   │   [{']['.join(row_vals)}] │")
        
        # Show statistics
        print(f"   │ Statistics:                              │")
        print(f"   │   • Ground Truth preserved: {gt_count:2d}/{total}         │")
        print(f"   │   • Masked (→ M/1024):    {mask_count:2d}/{total}         │")
        print(f"   │   • Noised (→ random):    {noise_count:2d}/{total}         │")
        print(f"   └──────────────────────────────────────────┘")
    
    # Show temporal mask
    print(f"\n⏱️  Temporal Attention Mask:")
    print(f"   (Shows which frames can attend to which)")
    T_size = temporal_mask.shape[0]
    header = "       " + "  ".join([f"T{i}" for i in range(T_size)])
    print(f"   {header}")
    for i in range(T_size):
        row_vals = []
        for j in range(T_size):
            val = temporal_mask[i, j].item()
            if val == float('-inf'):
                row_vals.append("-∞ ")
            else:
                row_vals.append(f"{val:2.0f} ")
        print(f"      T{i}   " + " ".join(row_vals))
    
    # Explain the mask
    print(f"\n   Mask Interpretation:")
    for t in range(T_size):
        can_attend = []
        for j in range(T_size):
            if temporal_mask[t, j].item() == 0:
                can_attend.append(f"T{j}")
        print(f"   • Frame {t} can attend to: {', '.join(can_attend)}")
    
    # Loss computation details
    print(f"\n🎯 Loss Computation:")
    print(f"   Loss = F.cross_entropy(logits, targets, label_smoothing={cfg.label_smoothing})")
    print(f"   ")
    print(f"   ⚠️  IMPORTANT: Loss is computed on ALL positions!")
    print(f"      Not just masked positions.")
    print(f"   ")
    print(f"   For each position (t, h, w):")
    print(f"     - Model outputs logits over vocab [0-{cfg.codebook_size}] (size={cfg.vocab_size})")
    print(f"     - Target is ground truth token (from targets tensor)")
    print(f"     - Cross-entropy measures prediction quality")
    print(f"     - Label smoothing {cfg.label_smoothing} prevents overconfidence")
    
    return {
        'masked_tokens': masked_tokens,
        'targets': targets,
        'temporal_mask': temporal_mask
    }


def show_code_implementation():
    """Show the actual code implementation."""
    print_section("IMPLEMENTATION IN CODE", char="#")
    
    code = '''
# File: copilot4d/world_model/masking.py

class DiscreteDiffusionMasker:
    def prepare_batch(self, tokens, objective=None):
        """
        Prepare a batch for training with discrete diffusion.
        
        Three objectives:
        1. future_prediction (50%): Past=GT, Future=masked+noised, Causal mask
        2. joint_denoise (40%): All frames masked+noised, Causal mask  
        3. individual_denoise (10%): All frames masked+noised, Identity mask
        """
        # Sample objective if not specified
        if objective is None:
            objective = self.sample_objective()  # 50/40/10 split
        
        # Sample mask ratio: u0 ~ Uniform(0,1), ratio = gamma(u0)
        u0 = random.random()
        mask_ratio = cosine_mask_schedule(torch.tensor(u0)).item()
        
        # Sample noise ratio: u1 ~ Uniform(0,1), noise = u1 * eta%
        u1 = random.random()
        noise_ratio = u1 * self.cfg.noise_eta / 100.0
        
        if objective == "future_prediction":
            # Past frames: ground truth (no masking)
            # Future frames: masked + noised
            temporal_mask = self._make_causal_mask(T, device)
            
            for b in range(B):
                future_tokens = tokens[b, num_past:]
                masked_future, was_masked_future = self.apply_random_masking(
                    future_tokens, mask_ratio, noise_ratio
                )
                out_tokens[b, num_past:] = masked_future
                
        elif objective == "joint_denoise":
            # All frames partially masked + noised
            temporal_mask = self._make_causal_mask(T, device)
            
            for b in range(B):
                masked, was_m = self.apply_random_masking(
                    tokens[b], mask_ratio, noise_ratio
                )
                out_tokens[b] = masked
                
        elif objective == "individual_denoise":
            # Each frame independently masked + noised
            temporal_mask = self._make_identity_mask(T, device)  # KEY DIFFERENCE!
            
            for b in range(B):
                for t in range(T):  # Process each frame independently
                    masked, was_m = self.apply_random_masking(
                        tokens[b, t], mask_ratio, noise_ratio
                    )
                    out_tokens[b, t] = masked
        
        return {
            "tokens": out_tokens,      # Input to model (masked)
            "targets": tokens,          # Target (original GT)
            "temporal_mask": temporal_mask,
            "objective": objective,
        }


# File: scripts/train_world_model.py

def train_step(model, masker, batch, optimizer, scaler, cfg, device):
    tokens = batch["tokens"].to(device)  # (B, T, H, W) - all GT
    actions = batch["actions"].to(device)
    
    # Apply discrete diffusion masking
    masked_batch = masker.prepare_batch(tokens)  # Apply Algorithm 1
    masked_tokens = masked_batch["tokens"]       # What model sees
    targets = masked_batch["targets"]            # What to predict
    temporal_mask = masked_batch["temporal_mask"]
    
    with torch.cuda.amp.autocast(enabled=cfg.amp):
        # Forward pass
        logits = model(masked_tokens, actions, temporal_mask)
        # logits: (B, T, H*W, vocab_size=1025)
        
        # Compute loss on ALL positions
        loss = compute_diffusion_loss(logits, targets, cfg.label_smoothing)
        # loss = F.cross_entropy(logits_flat, targets_flat, label_smoothing=0.1)
    
    # Backward pass
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    
    return loss
'''
    print(code)


def create_comparison_table():
    """Create a comprehensive comparison table."""
    print_section("COMPREHENSIVE COMPARISON TABLE")
    
    table = """
    ┌─────────────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┐
    │         Aspect          │    Future Prediction    │      Joint Denoise      │   Individual Denoise    │
    ├─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
    │                         │                         │                         │                         │
    │  PROBABILITY            │         50%             │          40%            │          10%            │
    │                         │                         │                         │                         │
    ├─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
    │                         │                         │                         │                         │
    │  PAST FRAMES (T=0,1,2)  │   Ground Truth (GT)     │   Masked + Noised       │   Masked + Noised       │
    │                         │   [1][1][1][1]          │   [M][1][M][R]          │   [M][1][M][R]          │
    │                         │   No corruption         │   Partial corruption    │   Independent mask      │
    │                         │                         │                         │                         │
    ├─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
    │                         │                         │                         │                         │
    │  FUTURE FRAMES (T=3,4,5)│   Fully Masked+Noised   │   Masked + Noised       │   Masked + Noised       │
    │                         │   [M][R][M][M]          │   [M][1][M][R]          │   [M][1][M][R]          │
    │                         │   High corruption       │   Partial corruption    │   Independent mask      │
    │                         │                         │                         │                         │
    ├─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
    │                         │                         │                         │                         │
    │  TEMPORAL MASK          │      Causal             │      Causal             │      Identity           │
    │                         │                         │                         │                         │
    │      T0 T1 T2 T3 T4 T5  │      T0 T1 T2 T3 T4 T5  │      T0 T1 T2 T3 T4 T5  │      T0 T1 T2 T3 T4 T5  │
    │  T0  [0  -  -  -  -  -] │  T0  [0  -  -  -  -  -] │  T0  [0  -  -  -  -  -] │  T0  [0  -  -  -  -  -] │
    │  T1  [0  0  -  -  -  -] │  T1  [0  0  -  -  -  -] │  T1  [0  0  -  -  -  -] │  T1  [-  0  -  -  -  -] │
    │  T2  [0  0  0  -  -  -] │  T2  [0  0  0  -  -  -] │  T2  [0  0  0  -  -  -] │  T2  [-  -  0  -  -  -] │
    │  T3  [0  0  0  0  -  -] │  T3  [0  0  0  0  -  -] │  T3  [0  0  0  0  -  -] │  T3  [-  -  -  0  -  -] │
    │  T4  [0  0  0  0  0  -] │  T4  [0  0  0  0  0  -] │  T4  [0  0  0  0  0  -] │  T4  [-  -  -  -  0  -] │
    │  T5  [0  0  0  0  0  0] │  T5  [0  0  0  0  0  0] │  T5  [0  0  0  0  0  0] │  T5  [-  -  -  -  -  0] │
    │                         │                         │                         │                         │
    │  0 = can attend         │  Same causal mask       │  Same causal mask       │  Diagonal only          │
    │  - = blocked (-inf)     │                         │                         │                         │
    │                         │                         │                         │                         │
    ├─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
    │                         │                         │                         │                         │
    │  WHAT MODEL LEARNS      │  Predict future from    │  Jointly denoise        │  Unconditional          │
    │                         │  clean past             │  entire sequence        │  generation             │
    │                         │                         │                         │                         │
    │                         │  p(x_future | x_past)   │  p(x_all | x_corrupted) │  p(x | masked_x)        │
    │                         │                         │                         │                         │
    ├─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
    │                         │                         │                         │                         │
    │  LOSS COMPUTATION       │  Cross-entropy on ALL   │  Cross-entropy on ALL   │  Cross-entropy on ALL   │
    │                         │  positions              │  positions              │  positions              │
    │                         │                         │                         │                         │
    │  F.cross_entropy(       │  F.cross_entropy(       │  F.cross_entropy(       │  F.cross_entropy(       │
    │    logits,              │    logits,              │    logits,              │    logits,              │
    │    targets,             │    targets,             │    targets,             │    targets,             │
    │    label_smoothing=0.1  │    label_smoothing=0.1  │    label_smoothing=0.1  │    label_smoothing=0.1  │
    │  )                      │  )                      │  )                      │  )                      │
    │                         │                         │                         │                         │
    │  Applied to:            │  Applied to:            │  Applied to:            │  Applied to:            │
    │  • Masked positions     │  • Masked positions     │  • Masked positions     │  • Masked positions     │
    │  • Unmasked positions   │  • Unmasked positions   │  • Unmasked positions   │  • Unmasked positions   │
    │  • Past frames          │  • Past frames          │  • Past frames          │  • Past frames          │
    │  • Future frames        │  • Future frames        │  • Future frames        │  • Future frames        │
    │                         │                         │                         │                         │
    ├─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
    │                         │                         │                         │                         │
    │  PURPOSE                │  Main training          │  Auxiliary training     │  Enable CFG at          │
    │                         │  objective for          │  for robustness         │  inference time         │
    │                         │  prediction             │                         │                         │
    │                         │                         │                         │                         │
    │                         │                         │                         │  Learn unconditional    │
    │                         │                         │                         │  model p(x) for         │
    │                         │                         │                         │  classifier-free        │
    │                         │                         │                         │  guidance:              │
    │                         │                         │                         │  logit = cond + w ×    │
    │                         │                         │                         │         (cond-uncond)   │
    │                         │                         │                         │                         │
    └─────────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┘
    """
    print(table)


def show_key_differences():
    """Highlight key differences between objectives."""
    print_section("KEY DIFFERENCES SUMMARY")
    
    print("""
    1️⃣  FUTURE PREDICTION vs JOINT DENOISE
    ───────────────────────────────────────
    
    DIFFERENCE: How past frames are treated
    
    Future Prediction:
      Past: [1][1][1][1]  ← Clean GT (given to model as-is)
      
    Joint Denoise:
      Past: [M][1][M][R]  ← Corrupted (masked + noised)
    
    IMPLICATION:
    • Future Prediction: Model learns to predict future from CLEAN past
    • Joint Denoise: Model learns to denoise CORRUPTED sequences
    • Joint is harder → makes model more robust
    
    
    2️⃣  JOINT DENOISE vs INDIVIDUAL DENOISE
    ───────────────────────────────────────
    
    DIFFERENCE: Temporal attention pattern
    
    Joint Denoise (Causal):
      Frame 3 can attend to: T0, T1, T2, T3
      Frame 5 can attend to: T0, T1, T2, T3, T4, T5
      
    Individual Denoise (Identity):
      Frame 3 can attend to: T3 ONLY
      Frame 5 can attend to: T5 ONLY
    
    IMPLICATION:
    • Joint: Model uses temporal context from other frames
    • Individual: Each frame processed independently
    • Individual trains unconditional generation for CFG
    
    
    3️⃣  WHY 50/40/10 SPLIT?
    ─────────────────────────
    
    50% Future Prediction:
      → Main task: Predict future from past
      → Most important for world modeling
      
    40% Joint Denoise:
      → Auxiliary task: Improve robustness
      → Helps with partial observations
      
    10% Individual Denoise:
      → Required for CFG at inference
      → Learn unconditional p(x)
      → Lower weight because not directly used for prediction
    """)


def verify_implementation_consistency():
    """Verify that implementation matches paper specification."""
    print_section("VERIFICATION: Implementation vs Paper")
    
    cfg = WorldModelConfig()
    masker = DiscreteDiffusionMasker(cfg)
    
    # Create test data
    tokens = torch.ones(1, 6, 4, 4, dtype=torch.long)
    
    checks = []
    
    # Check 1: Future Prediction keeps past as GT
    torch.manual_seed(42)
    random.seed(42)
    result = masker.prepare_batch(tokens.clone(), objective="future_prediction")
    past_unchanged = torch.all(result['tokens'][0, :3] == tokens[0, :3]).item()
    checks.append(("Future Prediction: Past frames unchanged", past_unchanged))
    
    # Check 2: Future Prediction corrupts future
    future_corrupted = not torch.all(result['tokens'][0, 3:] == tokens[0, 3:]).item()
    checks.append(("Future Prediction: Future frames corrupted", future_corrupted))
    
    # Check 3: Joint Denoise corrupts past
    torch.manual_seed(42)
    random.seed(42)
    result = masker.prepare_batch(tokens.clone(), objective="joint_denoise")
    past_corrupted = not torch.all(result['tokens'][0, :3] == tokens[0, :3]).item()
    checks.append(("Joint Denoise: Past frames corrupted", past_corrupted))
    
    # Check 4: Individual uses identity mask
    result = masker.prepare_batch(tokens.clone(), objective="individual_denoise")
    temporal_mask = result['temporal_mask']
    is_identity = True
    for i in range(temporal_mask.shape[0]):
        for j in range(temporal_mask.shape[1]):
            expected = 0.0 if i == j else float('-inf')
            if temporal_mask[i, j].item() != expected:
                is_identity = False
    checks.append(("Individual Denoise: Uses identity temporal mask", is_identity))
    
    # Check 5: Causal mask for Future/Joint
    for obj in ["future_prediction", "joint_denoise"]:
        result = masker.prepare_batch(tokens.clone(), objective=obj)
        temporal_mask = result['temporal_mask']
        is_causal = True
        for i in range(temporal_mask.shape[0]):
            for j in range(temporal_mask.shape[1]):
                expected = 0.0 if j <= i else float('-inf')
                if temporal_mask[i, j].item() != expected:
                    is_causal = False
        checks.append((f"{obj}: Uses causal temporal mask", is_causal))
    
    # Print results
    print("\n✓ VERIFICATION RESULTS:")
    print(f"{'─'*60}")
    for desc, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {desc}")
    
    all_passed = all(passed for _, passed in checks)
    print(f"{'─'*60}")
    print(f"  Overall: {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")
    
    return all_passed


def show_training_loop_integration():
    """Show how objectives integrate into training loop."""
    print_section("TRAINING LOOP INTEGRATION")
    
    code = '''
# File: scripts/train_world_model.py

def train_epoch(model, masker, dataloader, optimizer, scheduler, scaler, cfg, device, step):
    """Main training epoch."""
    model.train()
    
    for batch in dataloader:
        # 1. Get ground truth tokens (B, T, H, W)
        tokens = batch["tokens"].to(device)      # All GT
        actions = batch["actions"].to(device)
        
        # 2. Apply Algorithm 1 masking (RANDOM objective each iteration!)
        masked_batch = masker.prepare_batch(tokens)  # ← Randomly picks objective
        masked_tokens = masked_batch["tokens"]       # What model sees
        targets = masked_batch["targets"]            # What to predict
        temporal_mask = masked_batch["temporal_mask"]
        objective = masked_batch["objective"]        # Which objective was used
        
        # 3. Forward pass
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            logits = model(masked_tokens, actions, temporal_mask)
            # logits: (B, T, H*W, vocab_size=1025)
            
            # 4. Compute loss on ALL positions
            loss = compute_diffusion_loss(logits, targets, cfg.label_smoothing)
        
        # 5. Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 6. Logging
        if step % cfg.log_every_steps == 0:
            print(f"Step {step}: loss={loss.item():.4f}, objective={objective}")
        
        step += 1
    
    return step


# Example training log output:
# Step 100: loss=2.3456, objective=future_prediction
# Step 200: loss=2.1234, objective=joint_denoise
# Step 300: loss=2.5678, objective=future_prediction
# Step 400: loss=2.0123, objective=individual_denoise
# ...
'''
    print(code)


def main():
    """Run all comparisons."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  COMPARISON OF 3 TRAINING OBJECTIVES IN ALGORITHM 1".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # Setup
    cfg = WorldModelConfig(
        codebook_size=1024,
        num_frames=6,
        num_past_frames=3,
        noise_eta=20.0,
        label_smoothing=0.1,
    )
    
    B, T, H, W = 1, 6, 4, 4
    tokens = torch.ones(B, T, H, W, dtype=torch.long)
    masker = DiscreteDiffusionMasker(cfg)
    
    # Detailed analysis of each objective
    for objective in ["future_prediction", "joint_denoise", "individual_denoise"]:
        analyze_training_example(objective, tokens.clone(), masker, cfg)
    
    # Show code implementation
    show_code_implementation()
    
    # Show comparison table
    create_comparison_table()
    
    # Show key differences
    show_key_differences()
    
    # Verify implementation
    verify_implementation_consistency()
    
    # Show training loop integration
    show_training_loop_integration()
    
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  ANALYSIS COMPLETE".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)


if __name__ == "__main__":
    main()
