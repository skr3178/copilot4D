#!/bin/bash
# Transfer only ZIP files from KITTI (run this on NEW server)

set -e

OLD_SERVER="user@100.98.123.127"  # Update with your current server IP/user
OLD_KITTI="/media/skr/storage/self_driving/CoPilot4D/data/kitti"
NEW_KITTI="/new/path/data/kitti"  # Update with your new server path

echo "=========================================="
echo "KITTI ZIP Transfer Script"
echo "=========================================="
mkdir -p "$NEW_KITTI"

# Transfer only essential ZIP files
echo ""
echo "[1/5] Transfering data_odometry_velodyne.zip (79 GB)..."
rsync -avzP --partial "$OLD_SERVER:$OLD_KITTI/data_odometry_velodyne.zip" "$NEW_KITTI/"

echo ""
echo "[2/5] Transfering calibration ZIP..."
rsync -avzP --partial "$OLD_SERVER:$OLD_KITTI/data_odometry_calib.zip" "$NEW_KITTI/"

echo ""
echo "[3/5] Transfering poses ZIP..."
rsync -avzP --partial "$OLD_SERVER:$OLD_KITTI/data_odometry_poses.zip" "$NEW_KITTI/"

echo ""
echo "[4/5] Transfering devkit ZIP..."
rsync -avzP --partial "$OLD_SERVER:$OLD_KITTI/devkit_odometry.zip" "$NEW_KITTI/"

echo ""
echo "[5/5] Transfering small devkit folders..."
rsync -avzP --partial "$OLD_SERVER:$OLD_KITTI/devkit/" "$NEW_KITTI/devkit/"
rsync -avzP --partial "$OLD_SERVER:$OLD_KITTI/kitti-devkit-odom/" "$NEW_KITTI/kitti-devkit-odom/"

echo ""
echo "=========================================="
echo "Transfer complete! Run extraction script next."
echo "=========================================="
