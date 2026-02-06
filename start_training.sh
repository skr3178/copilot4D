#!/bin/bash
# Start training on new server

set -e

COPILOT_DIR="${1:-/new/path/CoPilot4D}"
CHECKPOINT_STEP="${2:-40000}"

echo "=========================================="
echo "CoPilot4D Training - Resume from Step $CHECKPOINT_STEP"
echo "=========================================="
echo ""

cd "$COPILOT_DIR"

# Verify conda environment
if [ "$CONDA_DEFAULT_ENV" != "nuscenes" ]; then
    echo "Activating conda environment 'nuscenes'..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate nuscenes
fi

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Verify checkpoint exists
CHECKPOINT="outputs/tokenizer_memory_efficient/checkpoint_step_${CHECKPOINT_STEP}.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Available checkpoints:"
    ls -la outputs/tokenizer_memory_efficient/checkpoint_step_*.pt 2>/dev/null || echo "  None found"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo "Config: configs/tokenizer_memory_efficient.yaml"
echo ""

# Update config path if needed
CONFIG="configs/tokenizer_memory_efficient.yaml"

# Verify data path
echo "Verifying data path..."
python -c "
import yaml
with open('$CONFIG', 'r') as f:
    cfg = yaml.safe_load(f)
kitti_root = cfg.get('kitti_root', 'NOT SET')
print(f'  Config kitti_root: {kitti_root}')
import os
if os.path.exists(kitti_root):
    print(f'  ✓ Path exists')
else:
    print(f'  ✗ Path does NOT exist - UPDATE CONFIG!')
    exit(1)
"
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: KITTI data path not found!"
    echo "Update $CONFIG with correct kitti_root path"
    exit 1
fi

echo ""
echo "Starting training..."
echo "Log: training_resume_${CHECKPOINT_STEP}.log"
echo ""

# Start training
nohup python -u scripts/train_tokenizer.py \
    --config "$CONFIG" \
    --resume "$CHECKPOINT" \
    --device cuda \
    > "training_resume_${CHECKPOINT_STEP}.log" 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo ""
echo "Monitor with:"
echo "  tail -f $COPILOT_DIR/training_resume_${CHECKPOINT_STEP}.log"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Check initial output:"
sleep 2
tail -n 20 "training_resume_${CHECKPOINT_STEP}.log" 2>/dev/null || echo "(log file not ready yet)"
