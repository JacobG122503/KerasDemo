#!/bin/bash
#SBATCH --job-name=keras_breakout
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/training_log_%j.out
#SBATCH --account=s2026.se.4390.01
#SBATCH --partition=instruction
#SBATCH --mail-user=jacobgar@iastate.edu
#SBATCH --mail-type=END,FAIL

EMAIL_TO="jacobgar@iastate.edu"
LOG_FILE="logs/training_log_${SLURM_JOB_ID:-unknown}.out"

send_failure_email() {
    local exit_code="$1"
    local python_bin=""
    local body=""
    local sendmail_bin=""

    if [ "$exit_code" -eq 0 ]; then
        return 0
    fi

    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        python_bin="${VIRTUAL_ENV}/bin/python"
    else
        python_bin=$(command -v python3 || command -v python || true)
    fi

    if [ -z "$python_bin" ]; then
        echo "Unable to locate Python for failure notification." >&2
        return 0
    fi

    echo "Failure handler triggered (exit code: ${exit_code})."

    body="Keras Breakout DQN training failed.\n\n"
    body+="Host: $(hostname)\n"
    body+="SLURM_JOB_ID=${SLURM_JOB_ID:-not set}\n"
    body+="Exit code: ${exit_code}\n"
    body+="Log file: ${LOG_FILE}\n"

    if [ -f "$LOG_FILE" ]; then
        body+="\nLast 120 log lines:\n"
        body+="$(tail -n 120 "$LOG_FILE")"
    else
        body+="\nThe log file was not found when the failure handler ran."
    fi

    sendmail_bin=$(command -v sendmail || true)
    if [ -n "$sendmail_bin" ]; then
        {
            echo "To: ${EMAIL_TO}"
            echo "From: ${EMAIL_TO}"
            echo "Subject: Breakout DQN Training Failed (job ${SLURM_JOB_ID:-unknown})"
            echo "Content-Type: text/plain; charset=UTF-8"
            echo
            printf "%b\n" "$body"
        } | "$sendmail_bin" -t -oi && {
            echo "Failure email sent via sendmail."
            return 0
        }
    fi

    echo "sendmail unavailable, failure notification skipped."
}

trap 'send_failure_email "$?"' EXIT

# Load necessary modules
module load python/3.10 2>/dev/null || true
module load cuda/12.3.2-uxohlz5 2>/dev/null || module load cuda/12.4.1-cz3ljd3 2>/dev/null || module load cuda 2>/dev/null || true
module load cudnn/8.9.7.29-12-cuda12-ivkl2ud 2>/dev/null || module load cudnn 2>/dev/null || true
module list 2>&1 || true

# Show what GPU hardware SLURM allocated
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-not set}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
if ! nvidia-smi 2>/dev/null; then
    echo "ERROR: nvidia-smi unavailable. This job is not on a GPU-ready node."
    exit 1
fi

# Set up a virtual environment on the cluster if it doesn't exist
if [ ! -d "cluster_venv" ]; then
    echo "Setting up cluster Python environment (first run takes a few minutes)..."
    python3 -m venv cluster_venv
    source cluster_venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install --no-cache-dir -U -r requirements.txt
else
    source cluster_venv/bin/activate
    pip install --no-cache-dir -U -r requirements.txt
fi

# ── Build LD_LIBRARY_PATH from pip-installed nvidia packages + system CUDA ──
SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])')
PIP_NVIDIA="${SITE_PACKAGES}/nvidia"

PIP_LIB=""
for sub in cudnn cublas cuda_runtime cuda_nvrtc cuda_cupti curand cufft cusolver cusparse nccl nvjitlink; do
    [ -d "${PIP_NVIDIA}/${sub}/lib" ] && PIP_LIB="${PIP_NVIDIA}/${sub}/lib:${PIP_LIB}"
done

CUDA_LIB=""
for d in "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib" "${CUDA_DIR}/lib64" "${CUDA_DIR}/lib"; do
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
    DRIVER_LIB=$(ldconfig -p 2>/dev/null | grep 'libcuda.so ' | head -1 | sed 's|.*=> *||; s|/[^/]*$||')
fi

export LD_LIBRARY_PATH="${PIP_LIB}${CUDA_LIB:+${CUDA_LIB}:}${DRIVER_LIB:+${DRIVER_LIB}:}${LD_LIBRARY_PATH}"

echo "=== CUDA environment ==="
echo "CUDA_HOME=${CUDA_HOME:-not set}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME:-/usr/local/cuda}"
export TF_CUDA_PATHS="${CUDA_HOME:-/usr/local/cuda}"
export TF_CPP_MIN_LOG_LEVEL=0

# Test GPU visibility
TF_VISIBLE_GPU_COUNT=$(python3 -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices("GPU")))')
echo "TensorFlow GPUs visible: ${TF_VISIBLE_GPU_COUNT}"

if [ "${TF_VISIBLE_GPU_COUNT}" -lt 1 ]; then
    echo "WARNING: TensorFlow cannot see GPUs. Training will fall back to CPU."
fi

mkdir -p logs models

echo "Starting Atari Breakout DQN Training..."
set +e
python3 -u train_headless.py
TRAIN_EXIT_CODE=$?
set -e

if [ "${TRAIN_EXIT_CODE}" -ne 0 ]; then
    echo "ERROR: train_headless.py exited with code ${TRAIN_EXIT_CODE}"
    exit ${TRAIN_EXIT_CODE}
fi

echo "Training script completed successfully."
