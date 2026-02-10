#!/usr/bin/env python
"""Simple training monitor script."""
import sys
import time
import glob
import os
import torch

def get_latest_checkpoint():
    checkpoints = glob.glob("outputs/mnist_diffusion_fast/*.pt")
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def main():
    print("="*60)
    print("Training Monitor - Press Ctrl+C to exit")
    print("="*60)
    
    last_epoch = 0
    while True:
        try:
            # Get checkpoint info
            ckpt_path = get_latest_checkpoint()
            if ckpt_path:
                try:
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    epoch = ckpt.get('epoch', 0)
                    val_loss = ckpt.get('val_loss', 0)
                    
                    if epoch != last_epoch:
                        print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch}: Val Loss = {val_loss:.4f}")
                        last_epoch = epoch
                    else:
                        print(f".", end="", flush=True)
                        
                except Exception as e:
                    print(f"Error reading checkpoint: {e}")
            
            time.sleep(30)  # Update every 30 seconds
            
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")
            break

if __name__ == "__main__":
    main()
