#!/bin/bash
# Setup conda environment on new server

set -e

echo "=========================================="
echo "CoPilot4D Conda Environment Setup"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# Create environment
echo ""
echo "[1/4] Creating conda environment 'nuscenes'..."
conda create -n nuscenes python=3.9 -y

# Activate environment
echo ""
echo "[2/4] Installing PyTorch with CUDA 12.1..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nuscenes

# Install PyTorch
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo ""
echo "[3/4] Installing core dependencies..."
pip install numpy>=1.20.0 pyyaml>=5.4.0 tqdm>=4.60.0 pytest>=7.0.0

# Install additional packages needed
echo ""
echo "[4/4] Installing additional packages..."
pip install matplotlib scipy scikit-learn opencv-python-headless pillow pandas

# Install 3D/computer vision packages
pip install open3d addict

# Development tools (optional)
pip install ipython jupyter-core

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "
import torch
import numpy
import yaml
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ NumPy: {numpy.__version__}')
print(f'✓ PyYAML: {yaml.__version__}')
"

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "Activate with: conda activate nuscenes"
echo ""
