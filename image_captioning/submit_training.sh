#!/bin/bash
#SBATCH --job-name=keras_captioning
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=16G
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

    body="Keras cluster training failed.\n\n"
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
            echo "Subject: Keras Training Failed (job ${SLURM_JOB_ID:-unknown})"
            echo "Content-Type: text/plain; charset=UTF-8"
            echo
            printf "%b\n" "$body"
        } | "$sendmail_bin" -t -oi && {
            echo "Failure email sent via sendmail."
            return 0
        }
    fi

    echo "sendmail path failed or unavailable, trying Python fallback."
    "$python_bin" send_email.py \
        "$EMAIL_TO" \
        "Keras Training Failed (job ${SLURM_JOB_ID:-unknown})" \
        "$body" || true
}

trap 'send_failure_email "$?"' EXIT

# Load any necessary modules
module load python/3.10 2>/dev/null || true
module load cuda/12.3.2-uxohlz5 2>/dev/null || module load cuda/12.4.1-cz3ljd3 2>/dev/null || module load cuda 2>/dev/null || true
module load cudnn/8.9.7.29-12-cuda12-ivkl2ud 2>/dev/null || module load cudnn 2>/dev/null || true
module list 2>&1 || true

# ── Download Flickr8k dataset if not already present ──
DATASET_URL="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k"
if [ ! -d "Flicker8k_Dataset" ] || [ ! -f "Flickr8k.token.txt" ]; then
    echo "Flickr8k data not found on cluster — downloading..."
    if [ ! -f "Flickr8k_Dataset.zip" ]; then
        curl -L --fail --retry 3 -o Flickr8k_Dataset.zip "${DATASET_URL}/Flickr8k_Dataset.zip"
    fi
    if [ ! -f "Flickr8k_text.zip" ]; then
        curl -L --fail --retry 3 -o Flickr8k_text.zip "${DATASET_URL}/Flickr8k_text.zip"
    fi
    python3 -c "
import zipfile, os, shutil
if not os.path.isdir('Flicker8k_Dataset'):
    zipfile.ZipFile('Flickr8k_Dataset.zip').extractall('.')
if not os.path.isfile('Flickr8k.token.txt'):
    zipfile.ZipFile('Flickr8k_text.zip').extractall('.')
    src = os.path.join('Flickr8k_text', 'Flickr8k.token.txt')
    if os.path.isfile(src):
        shutil.copy2(src, 'Flickr8k.token.txt')
"
    # Clean up zips to save disk space
    rm -f Flickr8k_Dataset.zip Flickr8k_text.zip
    rm -rf Flickr8k_text __MACOSX
    echo "Dataset ready."
fi

# Show what GPU hardware SLURM allocated
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-not set}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
if ! nvidia-smi 2>/dev/null; then
    echo "ERROR: nvidia-smi unavailable. This job is not on a GPU-ready node."
    exit 1
fi

# Use a shared venv across projects to stay within home-dir quota
SHARED_VENV="$HOME/shared_cluster_venv"
if [ ! -d "$SHARED_VENV" ]; then
    echo "Setting up shared cluster Python environment (first run takes a few minutes)..."
    python3 -m venv "$SHARED_VENV"
    source "$SHARED_VENV/bin/activate"
    pip install --upgrade pip setuptools wheel
    # Use --no-cache-dir to save inodes (nvidia wheels are huge)
    pip install --no-cache-dir -U -r requirements.txt
else
    source "$SHARED_VENV/bin/activate"
    # Quick incremental update (already-installed packages are skipped fast)
    pip install --no-cache-dir -U -r requirements.txt
fi

# ── Build LD_LIBRARY_PATH from pip-installed nvidia packages + system CUDA ──
SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])')
PIP_NVIDIA="${SITE_PACKAGES}/nvidia"

PIP_LIB=""
for sub in cudnn cublas cuda_runtime cuda_nvrtc cuda_cupti curand cufft cusolver cusparse nccl nvjitlink; do
    [ -d "${PIP_NVIDIA}/${sub}/lib" ] && PIP_LIB="${PIP_NVIDIA}/${sub}/lib:${PIP_LIB}"
done

# System CUDA lib dir from modules (for libcuda.so driver stub etc.)
CUDA_LIB=""
for d in "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib" "${CUDA_DIR}/lib64" "${CUDA_DIR}/lib"; do
    [ -d "$d" ] && CUDA_LIB="$d" && break
done

# Locate the NVIDIA driver library (libcuda.so.1) — not in CUDA toolkit or pip
DRIVER_LIB=""
for d in /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib64; do
    if [ -f "$d/libcuda.so.1" ] || [ -f "$d/libcuda.so" ]; then
        DRIVER_LIB="$d"
        break
    fi
done
# Also try ldconfig as last resort
if [ -z "$DRIVER_LIB" ]; then
    DRIVER_LIB=$(ldconfig -p 2>/dev/null | grep 'libcuda.so ' | head -1 | sed 's|.*=> *||; s|/[^/]*$||')
fi

# Pip nvidia libs go FIRST so cuDNN 9 from pip takes priority over cuDNN 8 from module
# Driver lib is added to ensure libcuda.so.1 (GPU driver) is always reachable
export LD_LIBRARY_PATH="${PIP_LIB}${CUDA_LIB:+${CUDA_LIB}:}${DRIVER_LIB:+${DRIVER_LIB}:}${LD_LIBRARY_PATH}"

echo "=== CUDA environment ==="
echo "CUDA_HOME=${CUDA_HOME:-not set}"
echo "CUDA_LIB resolved: ${CUDA_LIB:-NOT FOUND}"
echo "DRIVER_LIB resolved: ${DRIVER_LIB:-NOT FOUND}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "Pip nvidia dir: ${PIP_NVIDIA}"
ls "${PIP_NVIDIA}/" 2>/dev/null || echo "  (no pip nvidia packages)"
python3 -m pip show nvidia-cublas-cu12 nvidia-cusolver-cu12 2>/dev/null | sed -n '1,12p' || true

# Tell the TF CUDA plugin where to find things
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME:-/usr/local/cuda}"
export TF_CUDA_PATHS="${CUDA_HOME:-/usr/local/cuda}"

# Enable verbose TF logging so dlopen failures are visible
export TF_CPP_MIN_LOG_LEVEL=0

# Diagnostic: try loading each GPU library individually
echo "=== Library dlopen diagnostic ==="
python3 - <<'PYDIAG'
import ctypes, os
libs = [
    ("libcuda.so.1",    "NVIDIA driver"),
    ("libcudart.so.12", "CUDA runtime"),
    ("libcublas.so.12", "cuBLAS"),
    ("libcublasLt.so.12","cuBLAS Lt"),
    ("libcufft.so.11",  "cuFFT"),
    ("libcurand.so.10", "cuRAND"),
    ("libcusolver.so.11","cuSOLVER"),
    ("libcusparse.so.12","cuSPARSE"),
    ("libcudnn.so.9",   "cuDNN (umbrella)"),
    ("libcudnn_ops.so.9","cuDNN ops"),
    ("libcudnn_cnn.so.9","cuDNN cnn"),
    ("libcupti.so.12",  "CUPTI"),
    ("libnccl.so.2",    "NCCL"),
    ("libnvJitLink.so.12","nvJitLink"),
]
for so, desc in libs:
    try:
        ctypes.CDLL(so)
        print(f"  OK   {so:28s} ({desc})")
    except OSError as e:
        print(f"  FAIL {so:28s} ({desc}): {e}")
PYDIAG

# Test GPU visibility
TF_VISIBLE_GPU_COUNT=$(python3 - <<'PY'
import tensorflow as tf
print(len(tf.config.list_physical_devices("GPU")))
PY
)

TF_VISIBLE_GPU_DETAILS=$(python3 - <<'PY'
import tensorflow as tf
for gpu in tf.config.list_physical_devices("GPU"):
    print(f"  {gpu}")
PY
)

echo "TensorFlow GPUs visible: ${TF_VISIBLE_GPU_COUNT}"
[ -n "${TF_VISIBLE_GPU_DETAILS}" ] && echo "${TF_VISIBLE_GPU_DETAILS}"

if [ "${TF_VISIBLE_GPU_COUNT}" -lt 1 ]; then
    echo "WARNING: TensorFlow cannot see GPUs. Training will fall back to CPU."
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Clean stale tf.data cache lockfiles from previous runs
rm -f cache/*.lockfile 2>/dev/null

# Run the python training script
echo "Starting Keras Training..."
set +e
python3 -u train_headless.py
TRAIN_EXIT_CODE=$?
set -e

if [ "${TRAIN_EXIT_CODE}" -ne 0 ]; then
    echo "ERROR: train_headless.py exited with code ${TRAIN_EXIT_CODE}"
    exit ${TRAIN_EXIT_CODE}
fi

echo "Training script completed successfully."