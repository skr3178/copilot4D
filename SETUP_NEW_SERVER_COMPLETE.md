# CoPilot4D New Server Setup - Complete Guide

## Overview
- **Transfer size**: ~79 GB (ZIPs only) instead of ~238 GB
- **Total time**: ~2.5-3 hours
- **Disk space saved on new server**: ~80 GB (using symlinks)

---

## Step-by-Step Instructions

### **PREP: Update Config in Scripts**

Edit these variables in the scripts:
```bash
# In transfer_and_setup_new_server.sh:
OLD_SERVER="user@YOUR_CURRENT_SERVER_IP"
NEW_BASE="/your/new/server/path"
```

---

### **STEP 1: Run Transfer Script (on NEW server)**

```bash
# On new server
mkdir -p /your/new/server/path
cd /your/new/server/path

# Copy scripts from old server first (small files)
scp user@old-server:/media/skr/storage/self_driving/CoPilot4D/*.sh ./

# Run transfer
bash transfer_and_setup_new_server.sh
```

**This transfers:**
- Code & configs (~10 MB) - 5 minutes
- Checkpoint step 40000 (~100 MB) - 2 minutes  
- KITTI ZIP files (~79 GB) - 1.5-2 hours
- pykitti repo (~1 MB) - instant

---

### **STEP 2: Extract KITTI (on NEW server)**

```bash
cd /your/new/server/path/CoPilot4D

# Extract and setup with symlinks
bash extract_kitti_optimized.sh /your/new/server/path/data/kitti
```

**This creates:**
- `dataset/` - Full KITTI data (poses, sequences with velodyne)
- `pykitti/dataset/` - Symlinks to dataset/ (saves 80 GB!)
- `devkit/`, `kitti-devkit-odom/` - Development tools

**Disk usage after extraction:** ~80 GB (not 160 GB!)

---

### **STEP 3: Setup Conda Environment (on NEW server)**

```bash
cd /your/new/server/path/CoPilot4D
bash setup_conda_env.sh
```

**This installs:**
- Python 3.9
- PyTorch 2.5.1 with CUDA 12.1
- All required packages

---

### **STEP 4: Update Config Path**

```bash
# Edit the config file
nano /your/new/server/path/CoPilot4D/configs/tokenizer_memory_efficient.yaml

# Update this line:
kitti_root: "/your/new/server/path/data/kitti/pykitti"
```

---

### **STEP 5: Start Training (on NEW server)**

```bash
cd /your/new/server/path/CoPilot4D
bash start_training.sh /your/new/server/path/CoPilot4D 40000
```

**Monitor:**
```bash
tail -f /your/new/server/path/CoPilot4D/training_resume_40000.log
nvidia-smi -l 1
```

---

## Expected Output

When training starts successfully, you should see:
```
Resuming from step 40000
Loading checkpoint: outputs/tokenizer_memory_efficient/checkpoint_step_40000.pt
Step 40001 | loss: 0.XXXX | vq: 0.XXXX | use: XXX | perp: XXX
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `batch_size` in config (try 2 or 1) |
| `KITTI data not found` | Check `kitti_root` path in config |
| `checkpoint not found` | Verify checkpoint transferred: `ls outputs/tokenizer_memory_efficient/` |
| `ImportError` | Ensure conda env activated: `conda activate nuscenes` |
| `Permission denied` | Run: `chmod +x *.sh` |

---

## Quick Reference Commands

```bash
# Check transfer progress (on new server)
watch -n 5 'du -sh /your/new/server/path/data/kitti/*.zip'

# Check extraction progress
watch -n 5 'du -sh /your/new/server/path/data/kitti/dataset'

# Verify symlinks work
ls -la /your/new/server/path/data/kitti/pykitti/dataset/sequences/00/

# Check training log
tail -f /your/new/server/path/CoPilot4D/training_resume_40000.log

# GPU monitoring
watch -n 1 nvidia-smi
```

---

## File Checklist After Setup

On new server, verify these exist:
```bash
# Data
/your/new/server/path/data/kitti/
├── dataset/sequences/00/velodyne/000000.bin  # Sample velodyne file
├── dataset/poses/00.txt                      # Sample pose file
└── pykitti/dataset/sequences/00/velodyne -> ../../../dataset/sequences/00/velodyne  # Symlink

# Code
/your/new/server/path/CoPilot4D/
├── copilot4d/
├── scripts/
├── configs/tokenizer_memory_efficient.yaml
└── outputs/tokenizer_memory_efficient/checkpoint_step_40000.pt
```

---

## Time Estimates

| Step | Duration | Can Parallel? |
|------|----------|---------------|
| Transfer code/checkpoint | ~10 min | ✓ |
| Transfer KITTI ZIPs | 1.5-2 hours | - |
| Extract KITTI | 20-30 min | After transfer |
| Setup conda env | 10-15 min | ✓ (during transfer) |
| Start training | 2 min | After all above |
| **Total** | **~2-3 hours** | |

---

Ready to transfer! 🚀
