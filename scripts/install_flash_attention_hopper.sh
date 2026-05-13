#!/bin/bash
# -----------------------------------------------------------------------------
# Install Flash Attention for NVIDIA Hopper (H100) GPUs
#
# This script installs both:
#   1. Flash Attention 2 (standard) - supports Ampere, Ada, Hopper
#   2. Flash Attention 3 (Hopper-optimized) - specifically for H100/H800
#
# Requirements:
#   - H100/H800 GPU (for FA3)
#   - CUDA >= 12.3 (recommend CUDA 12.8)
#   - PyTorch 2.2+
#   - ninja, packaging, psutil Python packages
#
# Usage:
#   bash scripts/install_flash_attention_hopper.sh
# -----------------------------------------------------------------------------

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Flash Attention Installer for Hopper (H100) ===${NC}"

# Check if running on GPU node
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. Make sure you're on a GPU node.${NC}"
fi

# Load modules (same as other scripts in this repo)
echo -e "${GREEN}Loading CUDA modules...${NC}"
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0 || {
    echo -e "${YELLOW}Warning: Could not load modules. Ensure CUDA 12.x is available.${NC}"
}

# Activate virtual environment
cd "$(dirname "$(realpath "$0")")/.."
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo -e "${GREEN}Activated virtual environment${NC}"
else
    echo -e "${RED}Error: .venv not found. Run 'uv sync' first.${NC}"
    exit 1
fi

# Verify PyTorch and CUDA
echo -e "${GREEN}Checking PyTorch and CUDA versions...${NC}"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Install prerequisites
echo -e "${GREEN}Installing prerequisites...${NC}"
pip install -U ninja packaging psutil wheel

# Verify ninja works
if ! ninja --version &> /dev/null; then
    echo -e "${RED}Error: ninja is not working correctly. Please reinstall:${NC}"
    echo "  pip uninstall -y ninja && pip install ninja"
    exit 1
fi

# Set compilation limits for machines with limited RAM
# Flash Attention compilation can use a lot of memory
export MAX_JOBS=${MAX_JOBS:-4}
echo -e "${GREEN}Using MAX_JOBS=${MAX_JOBS} for compilation${NC}"

# Install standard Flash Attention 2
echo -e "${GREEN}Installing Flash Attention 2 (standard)...${NC}"
echo -e "${YELLOW}This may take 5-10 minutes...${NC}"
pip install flash-attn --no-build-isolation || {
    echo -e "${RED}Error: Failed to install flash-attn${NC}"
    echo "Try building from source:"
    echo "  git clone https://github.com/Dao-AILab/flash-attention.git"
    echo "  cd flash-attention"
    echo "  MAX_JOBS=4 python setup.py install"
    exit 1
}

# Verify FA2 installation
echo -e "${GREEN}Verifying Flash Attention 2 installation...${NC}"
python -c "
from flash_attn import flash_attn_func
print('Flash Attention 2 imported successfully')
"

# Install Flash Attention 3 (Hopper-optimized)
# FA3 is in the hopper/ directory of the flash-attention repo
echo -e "${GREEN}Installing Flash Attention 3 (Hopper-optimized)...${NC}"
echo -e "${YELLOW}This is specifically optimized for H100/H800 GPUs${NC}"

# Check if we have the flash-attention repo locally
FLASH_ATTN_DIR=""
if [ -d "experiments/validate_compute/megatron-lm/flash-attention" ]; then
    FLASH_ATTN_DIR="experiments/validate_compute/megatron-lm/flash-attention"
elif python -c "import flash_attn; print(flash_attn.__file__)" &> /dev/null; then
    # Find the flash-attn package directory
    FLASH_ATTN_PKG=$(python -c "import flash_attn; import os; print(os.path.dirname(flash_attn.__file__))")
    # The hopper dir should be in the repo, not in the installed package
    # Let's clone the repo instead
    echo -e "${YELLOW}Cloning flash-attention repository for FA3...${NC}"
    git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
    FLASH_ATTN_DIR="/tmp/flash-attention"
else
    echo -e "${YELLOW}Cloning flash-attention repository for FA3...${NC}"
    git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
    FLASH_ATTN_DIR="/tmp/flash-attention"
fi

if [ -d "${FLASH_ATTN_DIR}/hopper" ]; then
    cd "${FLASH_ATTN_DIR}/hopper"
    echo -e "${GREEN}Building Flash Attention 3 from ${FLASH_ATTN_DIR}/hopper...${NC}"
    MAX_JOBS=${MAX_JOBS} python setup.py install || {
        echo -e "${YELLOW}Warning: FA3 build failed. This may be expected if you're not on an H100.${NC}"
        echo -e "${YELLOW}FA2 should still work on H100 with good performance.${NC}"
    }
    cd -
else
    echo -e "${YELLOW}Warning: hopper/ directory not found. FA3 not installed.${NC}"
fi

# Verify FA3 installation
echo -e "${GREEN}Verifying Flash Attention 3 installation...${NC}"
python -c "
try:
    import flash_attn_interface
    print('Flash Attention 3 imported successfully')
except ImportError:
    print('Flash Attention 3 not available (expected if not on H100)')
" || true

# Also check if megatron-lm imports work
echo -e "${GREEN}Checking Megatron-LM flash attention imports...${NC}"
python -c "
try:
    from flashattn_hopper.flash_attn_interface import _flash_attn_forward
    print('flashattn_hopper imported successfully')
except ImportError:
    print('flashattn_hopper not available')

try:
    import flash_attn_3
    print('flash_attn_3 imported successfully')
except ImportError:
    print('flash_attn_3 not available')
" || true

echo -e "${GREEN}=== Installation complete ===${NC}"
echo ""
echo "Summary:"
echo "  - Flash Attention 2: $(pip show flash-attn 2>/dev/null | grep Version || echo 'Not installed')"
echo ""
echo "To use in your training scripts, add:"
echo "  --use-flash-attn"
echo ""
echo "Note: If you see import errors, make sure you're using the same Python"
echo "      environment where flash-attn was installed."
