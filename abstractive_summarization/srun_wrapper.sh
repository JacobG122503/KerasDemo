#!/bin/bash
# Wrapper for srun to setup environment on cluster nodes

# Load shared environment
source "$HOME/shared_cluster_venv/bin/activate"

# Load modules
module load python/3.10 2>/dev/null || true
module load cuda/12.3.2-uxohlz5 2>/dev/null || module load cuda/12.4.1-cz3ljd3 2>/dev/null || module load cuda 2>/dev/null || true
module load cudnn/8.9.7.29-12-cuda12-ivkl2ud 2>/dev/null || module load cudnn 2>/dev/null || true

# Setup LD_LIBRARY_PATH for JAX/TF GPU backend
# Add more aggressive paths for system and PyPI CUDA/CUDNN libraries
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/curand/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export TF_CUDA_PATHS="${CUDA_HOME}"
export PYTHONUNBUFFERED=1

# Execute Python
python3 -u train.py "$@"