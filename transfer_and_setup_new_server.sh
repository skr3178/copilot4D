#!/bin/bash
# Complete transfer and setup for new server (run on NEW server)
# Transfers only ZIP files (~79 GB) instead of extracted folders (~238 GB)

set -e

# ═══════════════════════════════════════════════════════════════════
# CONFIG - Update these values
# ═══════════════════════════════════════════════════════════════════
OLD_SERVER="user@100.98.123.127"  # Your current server
OLD_KITTI="/media/skr/storage/self_driving/CoPilot4D/data/kitti"
OLD_COPILOT="/media/skr/storage/self_driving/CoPilot4D"

NEW_BASE="/new/path"  # Your new server base path
NEW_KITTI="$NEW_BASE/data/kitti"
NEW_COPILOT="$NEW_BASE/CoPilot4D"
# ═══════════════════════════════════════════════════════════════════

echo "=========================================="
echo "CoPilot4D Transfer & Setup - NEW SERVER"
echo "=========================================="
echo ""
echo "Source: $OLD_SERVER"
echo "Dest:   $NEW_BASE"
echo ""

# Step 1: Create directories
mkdir -p "$NEW_KITTI"
mkdir -p "$NEW_COPILOT"
mkdir -p "$NEW_COPILOT/outputs/tokenizer_memory_efficient"

# Step 2: Transfer code (fast - ~5 min)
echo ""
echo "[PHASE 1/4] Transferring code and configs..."
rsync -avzP "$OLD_SERVER:$OLD_COPILOT/copilot4d/" "$NEW_COPILOT/copilot4d/"
rsync -avzP "$OLD_SERVER:$OLD_COPILOT/scripts/" "$NEW_COPILOT/scripts/"
rsync -avzP "$OLD_SERVER:$OLD_COPILOT/configs/" "$NEW_COPILOT/configs/"
rsync -avzP "$OLD_SERVER:$OLD_COPILOT/requirements.txt" "$NEW_COPILOT/"
rsync -avzP "$OLD_SERVER:$OLD_COPILOT/*.py" "$NEW_COPILOT/" 2>/dev/null || true
rsync -avzP "$OLD_SERVER:$OLD_COPILOT/*.sh" "$NEW_COPILOT/" 2>/dev/null || true

# Step 3: Transfer checkpoint (critical for resuming)
echo ""
echo "[PHASE 2/4] Transferring checkpoint (step 40000)..."
rsync -avzP --partial "$OLD_SERVER:$OLD_COPILOT/outputs/tokenizer_memory_efficient/checkpoint_step_40000.pt" \
    "$NEW_COPILOT/outputs/tokenizer_memory_efficient/"

# Optional: Transfer all checkpoints
# rsync -avzP --partial "$OLD_SERVER:$OLD_COPILOT/outputs/tokenizer_memory_efficient/" \
#     "$NEW_COPILOT/outputs/tokenizer_memory_efficient/"

# Step 4: Transfer only ZIP files (~79 GB)
echo ""
echo "[PHASE 3/4] Transferring KITTI ZIP files (~79 GB)..."
echo "This will take 1.5-2 hours depending on connection..."

rsync -avzP --partial "$OLD_SERVER:$OLD_KITTI/data_odometry_velodyne.zip" "$NEW_KITTI/"
rsync -avzP --partial "$OLD_SERVER:$OLD_KITTI/data_odometry_calib.zip" "$NEW_KITTI/"
rsync -avzP --partial "$OLD_SERVER:$OLD_KITTI/data_odometry_poses.zip" "$NEW_KITTI/"
rsync -avzP --partial "$OLD_SERVER:$OLD_KITTI/devkit_odometry.zip" "$NEW_KITTI/"

# Step 5: Transfer pykitti git repo (small - source code only)
echo ""
echo "[PHASE 4/4] Transferring pykitti package..."
rsync -avzP --exclude 'dataset' "$OLD_SERVER:$OLD_KITTI/pykitti/" "$NEW_KITTI/pykitti/"

# Also transfer devkit folders (small)
rsync -avzP "$OLD_SERVER:$OLD_KITTI/devkit/" "$NEW_KITTI/devkit/" 2>/dev/null || true
rsync -avzP "$OLD_SERVER:$OLD_KITTI/kitti-devkit-odom/" "$NEW_KITTI/kitti-devkit-odom/" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Transfer complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run extraction: bash extract_kitti_optimized.sh"
echo "  2. Setup conda env: bash setup_conda_env.sh"
echo "  3. Start training: bash start_training.sh"
echo ""
