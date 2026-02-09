#!/bin/bash
# Check training progress

echo "=== Training Status ==="
echo "Time: $(date)"
echo ""

echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader

echo ""
echo "Recent Training Progress:"
strings /media/skr/storage/self_driving/CoPilot4D/outputs/train_full.log | grep "loss=" | tail -5

echo ""
echo "Latest Epoch Info:"
strings /media/skr/storage/self_driving/CoPilot4D/outputs/train_full.log | grep -E "(Epoch [0-9]+ -|Val Loss|Saved best)" | tail -5
