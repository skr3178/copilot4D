#!/bin/bash
# Extract KITTI with optimized structure (no duplication) - run on NEW server

set -e

KITTI_DIR="${1:-/new/path/data/kitti}"  # Pass path as argument or edit default
cd "$KITTI_DIR"

echo "=========================================="
echo "KITTI Optimized Extraction"
echo "=========================================="
echo "Target: $KITTI_DIR"
echo ""

# Function to check if extraction needed
need_extract() {
    if [ -d "$1" ] && [ "$(ls -A $1 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  ✓ $1 already exists, skipping"
        return 1
    fi
    return 0
}

# ═══════════════════════════════════════════════════════════════════
# Step 1: Extract main dataset
# ═══════════════════════════════════════════════════════════════════

echo "[1/6] Extracting devkit_odometry.zip..."
if need_extract "devkit"; then
    unzip -q devkit_odometry.zip
    echo "  ✓ devkit extracted"
fi

echo ""
echo "[2/6] Extracting data_odometry_calib.zip..."
if need_extract "dataset"; then
    unzip -q data_odometry_calib.zip
    mv data_odometry_calib dataset 2>/dev/null || true
    echo "  ✓ Calibration data extracted"
fi

echo ""
echo "[3/6] Extracting data_odometry_poses.zip..."
if need_extract "dataset/poses" || [ -z "$(ls -A dataset/poses 2>/dev/null)" ]; then
    unzip -q data_odometry_poses.zip
    mkdir -p dataset/poses
    mv data_odometry_poses/* dataset/poses/ 2>/dev/null || true
    rm -rf data_odometry_poses 2>/dev/null || true
    echo "  ✓ Poses extracted"
fi

echo ""
echo "[4/6] Extracting data_odometry_velodyne.zip (LARGEST - ~10-20 min)..."
if need_extract "dataset/sequences" || [ -z "$(ls -A dataset/sequences 2>/dev/null)" ]; then
    echo "  Extracting 79 GB velodyne data..."
    unzip -q data_odometry_velodyne.zip
    
    # Move sequences to dataset/
    if [ -d "data_odometry_velodyne" ]; then
        mkdir -p dataset
        mv data_odometry_velodyne/sequences dataset/ 2>/dev/null || true
        rm -rf data_odometry_velodyne 2>/dev/null || true
    fi
    echo "  ✓ Velodyne data extracted"
fi

# ═══════════════════════════════════════════════════════════════════
# Step 2: Create pykitti structure with symlinks (saves 80 GB!)
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "[5/6] Setting up pykitti/dataset with symlinks (saves 80 GB space)..."

mkdir -p pykitti/dataset

# Create symlinks for sequences (each sequence needs: calib.txt, times.txt, velodyne)
for seq in 00 01 02 03 04 05 06 07 08 09 10; do
    if [ -d "dataset/sequences/$seq" ]; then
        mkdir -p "pykitti/dataset/sequences/$seq"
        
        # Symlink velodyne (the big data)
        if [ ! -L "pykitti/dataset/sequences/$seq/velodyne" ] && [ -d "dataset/sequences/$seq/velodyne" ]; then
            ln -s "../../../dataset/sequences/$seq/velodyne" "pykitti/dataset/sequences/$seq/velodyne"
        fi
        
        # Symlink or copy small files
        if [ ! -e "pykitti/dataset/sequences/$seq/calib.txt" ] && [ -f "dataset/sequences/$seq/calib.txt" ]; then
            cp "dataset/sequences/$seq/calib.txt" "pykitti/dataset/sequences/$seq/calib.txt"
        fi
        
        if [ ! -e "pykitti/dataset/sequences/$seq/times.txt" ] && [ -f "dataset/sequences/$seq/times.txt" ]; then
            cp "dataset/sequences/$seq/times.txt" "pykitti/dataset/sequences/$seq/times.txt"
        fi
    fi
done

# Symlink poses directory
if [ ! -L "pykitti/dataset/poses" ] && [ -d "dataset/poses" ]; then
    ln -s ../dataset/poses pykitti/dataset/poses
fi

echo "  ✓ pykitti structure created with symlinks"

# ═══════════════════════════════════════════════════════════════════
# Step 3: Verify structure
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "[6/6] Verifying structure..."

echo ""
echo "Expected structure:"
cat << 'EOF'
kitti/
├── dataset/
│   ├── poses/           # Ground truth poses
│   └── sequences/       # Full KITTI sequences
│       ├── 00/
│       │   ├── calib.txt
│       │   ├── times.txt
│       │   ├── image_*/      # (optional - images)
│       │   └── velodyne/     # LiDAR point clouds
│       ├── 01/ ... 10/
├── pykitti/             # Python package
│   ├── pykitti/         # Source code
│   └── dataset/         # -> symlinks to dataset/
│       ├── poses -> ../dataset/poses
│       └── sequences/
│           └── 00/velodyne -> ../../../dataset/sequences/00/velodyne
├── devkit/
└── kitti-devkit-odom/
EOF

echo ""
echo "Actual structure:"
find "$KITTI_DIR" -maxdepth 3 -type d 2>/dev/null | head -25

echo ""
echo "Symlink verification:"
ls -la "$KITTI_DIR/pykitti/dataset/" 2>/dev/null | head -10

echo ""
echo "Size check:"
du -sh "$KITTI_DIR"/* 2>/dev/null | sort -hr | head -10

# Calculate actual disk usage vs apparent size
echo ""
echo "Disk usage (with symlinks):"
du -sh "$KITTI_DIR"

# ═══════════════════════════════════════════════════════════════════
# Step 4: Cleanup option
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "=========================================="
echo "Extraction complete!"
echo "=========================================="
echo ""
echo "Disk space saved with symlinks: ~80 GB"
echo ""

read -p "Delete ZIP files to save ~79 GB? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f *.zip
    echo "  ✓ ZIP files deleted"
    echo "  Final KITTI size: $(du -sh $KITTI_DIR | cut -f1)"
fi

echo ""
echo "=========================================="
echo "Update config: kitti_root: '$KITTI_DIR/pykitti'"
echo "=========================================="
