# CoPilot4D Transfer Checklist

## Pre-Transfer Checklist

### On NEW Server:
- [ ] Verify CUDA 12.1+ is installed: `nvidia-smi`
- [ ] Install conda/miniconda: https://docs.conda.io/en/latest/miniconda.html
- [ ] Ensure sufficient disk space: 250+ GB free
- [ ] Test SSH connectivity to old server

---

## Phase 1: Transfer KITTI Dataset (238 GB)

### Option 1: Direct rsync (Recommended for LAN/High bandwidth)
```bash
# On new server
mkdir -p /new/path/data
screen -S kitti-transfer

rsync -avzP --partial --inplace \
    user@old-server:/media/skr/storage/self_driving/CoPilot4D/data/kitti \
    /new/path/data/

# Detach: Ctrl+A, D
# Check progress: screen -r kitti-transfer
```

### Option 2: If bandwidth is limited
```bash
# Compress first on old server
tar -czf /tmp/kitti.tar.gz -C /media/skr/storage/self_driving/CoPilot4D/data kitti

# Transfer compressed file
scp /tmp/kitti.tar.gz user@new-server:/tmp/

# Extract on new server
tar -xzf /tmp/kitti.tar.gz -C /new/path/data/
```

### Verification:
```bash
# Check sizes match
du -sh /new/path/data/kitti  # Should be ~238 GB
```

---

## Phase 2: Transfer Code & Checkpoints

```bash
# On new server
mkdir -p /new/path/CoPilot4D
cd /new/path/CoPilot4D

# Transfer code files (small, fast)
rsync -avzP user@old-server:/media/skr/storage/self_driving/CoPilot4D/copilot4d/ ./copilot4d/
rsync -avzP user@old-server:/media/skr/storage/self_driving/CoPilot4D/scripts/ ./scripts/
rsync -avzP user@old-server:/media/skr/storage/self_driving/CoPilot4D/configs/ ./configs/
rsync -avzP user@old-server:/media/skr/storage/self_driving/CoPilot4D/requirements.txt ./
rsync -avzP user@old-server:/media/skr/storage/self_driving/CoPilot4D/*.py ./

# Transfer checkpoints (important for resuming)
mkdir -p outputs/tokenizer_memory_efficient
rsync -avzP user@old-server:/media/skr/storage/self_driving/CoPilot4D/outputs/tokenizer_memory_efficient/checkpoint_step_40000.pt \
    outputs/tokenizer_memory_efficient/

# Optional: Transfer all checkpoints
rsync -avzP user@old-server:/media/skr/storage/self_driving/CoPilot4D/outputs/tokenizer_memory_efficient/ \
    outputs/tokenizer_memory_efficient/
```

---

## Phase 3: Setup Environment

```bash
# On new server
cd /new/path/CoPilot4D

# Run setup script
chmod +x setup_new_server.sh
./setup_new_server.sh

# Or manual setup:
conda create -n nuscenes python=3.9 -y
conda activate nuscenes
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Phase 4: Update Configurations

### 1. Update paths in config file:
```bash
# Edit configs/tokenizer_memory_efficient.yaml
# Change: kitti_root: "/new/path/data/kitti/pykitti"
```

### 2. Create symbolic link (optional):
```bash
# If you want to use same paths as old server
sudo mkdir -p /media/skr/storage/self_driving
sudo ln -s /new/path/data/kitti /media/skr/storage/self_driving/CoPilot4D/data/kitti
```

---

## Phase 5: Resume Training

```bash
# Activate environment
conda activate nuscenes

# Test data loading first
cd /new/path/CoPilot4D
python -c "from copilot4d.data.kitti_dataset import KITTITokenizerDataset; print('Data loading OK')"

# Resume training from step 40000
nohup python -u scripts/train_tokenizer.py \
    --config configs/tokenizer_memory_efficient.yaml \
    --resume outputs/tokenizer_memory_efficient/checkpoint_step_40000.pt \
    --device cuda > training_resume_40000.log 2>&1 &

# Monitor
tail -f training_resume_40000.log
```

---

## Post-Transfer Verification

- [ ] KITTI data size matches: `du -sh /new/path/data/kitti` ≈ 238 GB
- [ ] Environment activates: `conda activate nuscenes`
- [ ] PyTorch sees GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Can load checkpoint: Check log shows "Resuming from step 40000"
- [ ] Training starts without errors
- [ ] Loss values look reasonable (comparable to old server)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size in config |
| Data not found | Check kitti_root path in config |
| Import errors | Ensure you're in `nuscenes` conda env |
| Checkpoint load fails | Verify checkpoint transferred completely |
| Slow training | Check GPU utilization with `nvidia-smi -l 1` |
