#!/bin/bash
# Full discrete diffusion test - ~30 minutes
# This script trains a decent-sized model and samples from it

set -e  # Exit on error

echo "=================================="
echo "MOVING MNIST DISCRETE DIFFUSION"
echo "Full Test Suite (~30 minutes)"
echo "=================================="
echo ""

# Configuration
DATA_PATH="mnist_test_seq.1.npy"
OUTPUT_DIR="outputs/mnist_full_test"
FRAME_SIZE=64
SEQ_LEN=20
NUM_TRAIN=2000
NUM_VAL=500

# Model size (decent but fits in 12GB GPU)
EMBED_DIM=256
NUM_LAYERS=4
NUM_HEADS=8
BATCH_SIZE=4

# Training
EPOCHS=20
LR=1e-3

echo "Configuration:"
echo "  Frame size: ${FRAME_SIZE}x${FRAME_SIZE}"
echo "  Sequence length: ${SEQ_LEN}"
echo "  Training samples: ${NUM_TRAIN}"
echo "  Model: ${NUM_LAYERS} layers, ${EMBED_DIM} dim, ${NUM_HEADS} heads"
echo "  Epochs: ${EPOCHS}"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Step 1: Training
echo "=================================="
echo "STEP 1: Training (${EPOCHS} epochs)"
echo "Estimated time: ~20-25 minutes"
echo "=================================="
python tests/mnist_diffusion/train_diffusion.py \
  --data_path ${DATA_PATH} \
  --num_train ${NUM_TRAIN} \
  --num_val ${NUM_VAL} \
  --seq_len ${SEQ_LEN} \
  --frame_size ${FRAME_SIZE} \
  --batch_size ${BATCH_SIZE} \
  --embed_dim ${EMBED_DIM} \
  --num_layers ${NUM_LAYERS} \
  --num_heads ${NUM_HEADS} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --num_workers 4 \
  --output_dir ${OUTPUT_DIR} \
  --save_interval 5

echo ""
echo "=================================="
echo "STEP 2: Sampling - Pure Generation"
echo "Estimated time: ~3-5 minutes"
echo "=================================="

BEST_CHECKPOINT="${OUTPUT_DIR}/best_model.pt"
if [ ! -f "$BEST_CHECKPOINT" ]; then
    echo "Error: Best checkpoint not found at ${BEST_CHECKPOINT}"
    exit 1
fi

python tests/mnist_diffusion/sample_diffusion.py \
  --checkpoint ${BEST_CHECKPOINT} \
  --data_path ${DATA_PATH} \
  --seq_len ${SEQ_LEN} \
  --frame_size ${FRAME_SIZE} \
  --embed_dim ${EMBED_DIM} \
  --num_layers ${NUM_LAYERS} \
  --num_heads ${NUM_HEADS} \
  --num_samples 10 \
  --num_steps 12 \
  --temperature 1.0 \
  --mode generation \
  --output_dir ${OUTPUT_DIR}/samples_generation

echo ""
echo "=================================="
echo "STEP 3: Sampling - Future Prediction"
echo "Estimated time: ~3-5 minutes"
echo "=================================="

python tests/mnist_diffusion/sample_diffusion.py \
  --checkpoint ${BEST_CHECKPOINT} \
  --data_path ${DATA_PATH} \
  --seq_len ${SEQ_LEN} \
  --frame_size ${FRAME_SIZE} \
  --embed_dim ${EMBED_DIM} \
  --num_layers ${NUM_LAYERS} \
  --num_heads ${NUM_HEADS} \
  --num_samples 10 \
  --num_steps 8 \
  --temperature 1.0 \
  --mode future_prediction \
  --num_context 10 \
  --output_dir ${OUTPUT_DIR}/samples_future_pred

echo ""
echo "=================================="
echo "STEP 4: Analysis"
echo "Estimated time: ~1 minute"
echo "=================================="

python tests/mnist_diffusion/analyze_results.py \
  --output_dir ${OUTPUT_DIR}

echo ""
echo "=================================="
echo "TEST COMPLETE!"
echo "=================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key outputs:"
echo "  - Training logs: ${OUTPUT_DIR}/logs/"
echo "  - Best model: ${BEST_CHECKPOINT}"
echo "  - Generated samples (pure): ${OUTPUT_DIR}/samples_generation/"
echo "  - Generated samples (future): ${OUTPUT_DIR}/samples_future_pred/"
echo "  - Analysis plots: */metrics_analysis.png"
echo "  - Metrics: */metrics.txt"
echo ""
echo "To view samples, look at the PNG files in the sample directories."
