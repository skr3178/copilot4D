#!/bin/bash
# Setup script for CoPilot4D on new server
# Run this on the NEW server after data transfer

set -e

echo "============================================"
echo "CoPilot4D Environment Setup"
echo "============================================"

# 1. Check CUDA availability
echo ""
echo "Checking CUDA..."
nvidia-smi
python3 --version

# 2. Create conda environment
echo ""
echo "Creating conda environment 'nuscenes'..."
conda create -n nuscenes python=3.9 -y

# 3. Activate and install PyTorch
echo ""
echo "Installing PyTorch with CUDA 12.1..."
conda activate nuscenes
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install numpy>=1.20.0 pyyaml>=5.4.0 tqdm>=4.60.0 pytest>=7.0.0

# 5. Install additional required packages
echo ""
echo "Installing additional packages..."
pip install matplotlib scipy scikit-learn opencv-python-headless pillow

# 6. Install project-specific packages
echo ""
echo "Installing project packages..."
pip install open3d addict open3d nbformat fastjsonschema fire flask dash

# 7. Install development tools
echo ""
echo "Installing development tools..."
pip install ipython jupyterlab jupyter-core

# 8. Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo ""
echo "============================================"
echo "Setup complete! Activate with: conda activate nuscenes"
echo "============================================"
