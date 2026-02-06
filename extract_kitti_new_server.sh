#!/bin/bash
# Extract KITTI ZIP files to match current folder structure (run on NEW server)

set -e

KITTI_DIR="/new/path/data/kitti"  # Update this path
cd "$KITTI_DIR"

echo "=========================================="
echo "KITTI Extraction Script"
echo "=========================================="

# Function to check if folder exists and has content
check_folder() {
    if [ -d "$1" ] && [ "$(ls -A $1 2>/dev/null)" ]; then
        echo "  ✓ $1 already exists with content, skipping extraction"
        return 1
    fi
    return 0
}

echo ""
echo "[1/4] Extracting devkit_odometry.zip -> devkit/"
if check_folder "devkit"; then
    unzip -q devkit_odometry.zip
    echo "  ✓ Extracted devkit"
fi

echo ""
echo "[2/4] Extracting data_odometry_calib.zip -> dataset/"
if check_folder "dataset"; then
    unzip -q data_odometry_calib.zip
    # Rename if needed to match structure
    if [ -d "data_odometry_calib" ]; then
        mv data_odometry_calib dataset
    fi
    echo "  ✓ Extracted calibration data"
fi

echo ""
echo "[3/4] Extracting data_odometry_poses.zip -> dataset/poses/"
if [ ! -d "dataset/poses" ] || [ -z "$(ls -A dataset/poses 2>/dev/null)" ]; then
    unzip -q data_odometry_poses.zip
    # Move poses to correct location
    if [ -d "data_odometry_poses" ]; then
        mkdir -p dataset/poses
        mv data_odometry_poses/* dataset/poses/ 2>/dev/null || true
        rm -rf data_odometry_poses
    fi
    echo "  ✓ Extracted poses"
else
    echo "  ✓ Poses already exist, skipping"
fi

echo ""
echo "[4/4] Extracting data_odometry_velodyne.zip -> dataset/velodyne/ (LARGEST - takes time)"
if check_folder "dataset/velodyne"; then
    echo "  This is 79 GB, extraction will take 10-20 minutes..."
    unzip -q data_odometry_velodyne.zip
    # Move to correct structure
    if [ -d "data_odometry_velodyne" ]; then
        mkdir -p dataset
        mv data_odometry_velodyne/* dataset/ 2>/dev/null || true
        rm -rf data_odometry_velodyne
    fi
    echo "  ✓ Extracted velodyne data"
fi

echo ""
echo "=========================================="
echo "Verifying folder structure..."
echo "=========================================="

echo ""
echo "Expected structure:"
echo "  kitti/"
echo "    ├── dataset/"
echo "    │   ├── poses/"
echo "    │   ├── sequences/"
echo "    │   └── velodyne/"
echo "    ├── devkit/"
echo "    ├── kitti-devkit-odom/"
echo "    └── pykitti/"

echo ""
echo "Actual structure:"
find "$KITTI_DIR" -maxdepth 2 -type d | head -20

echo ""
echo "Size check:"
du -sh "$KITTI_DIR"/* | sort -hr | head -10

echo ""
echo "=========================================="
echo "Extraction complete!"
echo "=========================================="

# Optional: Clean up ZIP files to save space
echo ""
read -p "Delete ZIP files to save ~79 GB space? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f *.zip
    echo "  ✓ ZIP files deleted"
fi

echo ""
echo "Ready to train! Update config: kitti_root: '$KITTI_DIR/pykitti'"
