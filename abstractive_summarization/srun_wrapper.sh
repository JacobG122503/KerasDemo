#!/bin/bash
# Wrapper for srun to setup environment on cluster nodes

# Load shared environment
source "$HOME/shared_cluster_venv/bin/activate"

# Load modules
module load python/3.10 2>/dev/null || true
module load cuda/12.3.2-uxohlz5 2>/dev/null || module load cuda/12.4.1-cz3ljd3 2>/dev/null || module load cuda 2>/dev/null || true
module load cudnn/8.9.7.29-12-cuda12-ivkl2ud 2>/dev/null || module load cudnn 2>/dev/null || true

# Setup LD_LIBRARY_PATH for JAX/TF GPU backend
SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])')
PIP_NVIDIA="${SITE_PACKAGES}/nvidia"

PIP_LIB=""
for sub in cudnn cublas cuda_runtime cuda_nvrtc cuda_cupti curand cufft cusolver cusparse nccl nvjitlink; do
    [ -d "${PIP_NVIDIA}/${sub}/lib" ] && PIP_LIB="${PIP_NVIDIA}/${sub}/lib:${PIP_LIB}"
done

CUDA_LIB=""
for d in "${CUDA_HOME:-}/lib64" "${CUDA_HOME:-}/lib" "${CUDA_DIR:-}/lib64" "${CUDA_DIR:-}/lib"; do
    [ -d "$d" ] && CUDA_LIB="$d" && break
done

DRIVER_LIB=""
for d in /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib64; do
    if [ -f "$d/libcuda.so.1" ] || [ -f "$d/libcuda.so" ]; then
        DRIVER_LIB="$d"
        break
    fi
done
if [ -z "$DRIVER_LIB" ]; then
    DRIVER_LIB=$(ldconfig -p 2>/dev/null | grep 'libcuda.so ' | head -1 | sed 's|.*=> *||; s|/[^/]*$||' || true)
fi

export LD_LIBRARY_PATH="${PIP_LIB}${CUDA_LIB:+${CUDA_LIB}:}${DRIVER_LIB:+${DRIVER_LIB}:}${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME:-/usr/local/cuda}"
export TF_CUDA_PATHS="${CUDA_HOME:-/usr/local/cuda}"
export PYTHONUNBUFFERED=1

# Execute Python
python3 -u train.py "$@"