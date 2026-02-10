#!/bin/bash
# Fast training with cached dataset and smaller model

source $(conda info --base)/etc/profile.d/conda.sh
conda activate nuscenes
cd /media/skr/storage/self_driving/CoPilot4D

python tests/mnist_diffusion/train_mnist_fast.py \
  --embed_dim 128 \
  --num_layers 2 \
  --num_heads 4 \
  --batch_size 4 \
  --epochs 50 \
  --num_train 2000 \
  --num_val 500 \
  --output_dir outputs/mnist_diffusion_fast \
  2>&1 | tee outputs/mnist_diffusion_fast/training.log
