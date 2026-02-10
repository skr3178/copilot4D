#!/bin/bash
# Training script for small model

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nuscenes

# Change to project directory
cd /media/skr/storage/self_driving/CoPilot4D

# Run training with batch_size=2 to avoid OOM
python tests/mnist_diffusion/train_mnist_small.py \
  --embed_dim 192 \
  --num_layers 4 \
  --num_heads 4 \
  --frame_size 32 \
  --batch_size 2 \
  --epochs 20 \
  --save_interval 5 \
  --use_ego_centric \
  --output_dir outputs/mnist_diffusion_small \
  2>&1 | tee outputs/mnist_diffusion_small/training.log

