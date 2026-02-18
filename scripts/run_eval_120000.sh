#!/bin/bash
# Evaluate CoPilot4D tokenizer checkpoint_step_120000.pt

set -e

echo "=========================================="
echo "Evaluating checkpoint_step_120000.pt"
echo "=========================================="

# Configuration
CHECKPOINT="outputs/tokenizer_memory_efficient/checkpoint_step_120000.pt"
CONFIG="configs/tokenizer_memory_efficient.yaml"
OUTPUT_DIR="outputs/tokenizer_memory_efficient/reconstruction_step_120000"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Running reconstruction..."
echo "  Checkpoint: $CHECKPOINT"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo ""

python scripts/reconstruct_lidar.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --sample_idx 0 \
    --split val \
    --sequence 00 \
    --output_dir "$OUTPUT_DIR" \
    --dense \
    --chunk_size 512

echo ""
echo "=========================================="
echo "Reconstruction complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"
