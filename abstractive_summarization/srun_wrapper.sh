#!/bin/bash
# Wrapper for srun to setup environment on cluster nodes

# Load shared environment
source "$HOME/shared_cluster_venv/bin/activate"

# Load explicit modules available on the system
module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true
module load cudnn 2>/dev/null || true

# Force pip to install all NVIDIA packages required by tensorflow/jax via the official extra
pip install --no-cache-dir -U "tensorflow[and-cuda]"

# Aggressively build LD_LIBRARY_PATH from the local python site-packages
SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])')

export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/curand/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${CUDA_HOME:-/usr/local/cuda}/lib:${LD_LIBRARY_PATH:-}"

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export TF_CUDA_PATHS="${CUDA_HOME}"
export PYTHONUNBUFFERED=1

echo "Checking for GPU availability..."
python3 -c "
import tensorflow as tf
import sys
import os
import subprocess

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print('\n' + '='*70)
    print('ERROR: NO GPU DETECTED!')
    print('TensorFlow could not find any GPUs. This job will NOT default to CPU.')
    print('Debug Information follows:')
    print('='*70)
    print(f'LD_LIBRARY_PATH: {os.environ.get(\"LD_LIBRARY_PATH\", \"\")}')
    print(f'CUDA_HOME: {os.environ.get(\"CUDA_HOME\", \"\")}')
    print('-'*70)
    print('Running nvidia-smi:')
    subprocess.run(['nvidia-smi'], stderr=subprocess.STDOUT)
    print('='*70 + '\n')
    sys.exit(1)
print(f'SUCCESS: Found GPUs: {gpus}')
" || exit 1

# Execute Python
python3 -u train.py "$@"