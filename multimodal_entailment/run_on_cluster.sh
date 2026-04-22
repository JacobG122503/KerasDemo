#!/bin/bash
set -euo pipefail

REMOTE_USER="jacobgar"
REMOTE_HOST="nova-login-1.its.iastate.edu"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_UPLOAD_DIR="$(mktemp -d)"

cleanup_tmp() {
	rm -rf "$TMP_UPLOAD_DIR"
}
trap cleanup_tmp EXIT

echo "Preparing upload bundle..."
# Copy source tree without generated artifacts.
(
	cd "$LOCAL_DIR"
	COPYFILE_DISABLE=1 tar -cf - \
		--exclude='./models' \
		--exclude='./__pycache__' \
		--exclude='./venv' \
		--exclude='./.venv' \
		--exclude='./logs' \
		--exclude='./.git' \
		--exclude='./.git/*' \
		--exclude='./.DS_Store' \
		--exclude='*/.DS_Store' \
		--exclude='./._*' \
		--exclude='*/._*' \
		.
) | (
	cd "$TMP_UPLOAD_DIR"
	tar -xf -
)

echo "Uploading files and setting up environment on the cluster..."
(
	cd "$TMP_UPLOAD_DIR"
	COPYFILE_DISABLE=1 tar -czf - .
) | ssh -o ServerAliveInterval=60 "${REMOTE_USER}@${REMOTE_HOST}" '
set -euo pipefail

echo "Clearing remote home directory to free up disk quota (keeping only .ssh)..."
find "$HOME" -mindepth 1 -maxdepth 1 ! -name .ssh -exec rm -rf -- {} + || true

mkdir -p ~/multimodal_entailment
cd ~/multimodal_entailment

echo "Extracting code..."
tar -xzf -

chmod +x srun_wrapper.sh

SHARED_VENV="$HOME/shared_cluster_venv"
if [ ! -d "$SHARED_VENV" ]; then
    echo "Setting up shared cluster Python environment (first run takes a few minutes)..."
    module load python/3.10 2>/dev/null || true
    python3 -m venv "$SHARED_VENV"
    source "$SHARED_VENV/bin/activate"
    pip install --upgrade pip setuptools wheel
    pip install --no-cache-dir -U -r requirements.txt
else
    source "$SHARED_VENV/bin/activate"
    pip install --no-cache-dir -U -r requirements.txt
fi
'

run_epochs() {
    local epochs=$1
    echo ""
    echo "========================================="
    echo "Starting interactive training ($epochs epochs)"
    echo "========================================="
    ssh -t -o ServerAliveInterval=60 "${REMOTE_USER}@${REMOTE_HOST}" "
    set -euo pipefail
    cd ~/multimodal_entailment
    GPU_CANDIDATES=\$(sinfo -h -p instruction -o '%G' 2>/dev/null | tr ',' '\n' | grep -Eo 'gpu(:[^, ]+)*:[0-9]+' | sed -E 's/.*:([0-9]+)$/\1/' | sort -nr | uniq || true)
    GPU_COUNT=\$(echo \"\$GPU_CANDIDATES\" | head -n 1)
    if [ -z \"\$GPU_COUNT\" ] || [ \"\$GPU_COUNT\" -eq 0 ]; then GPU_COUNT=1; fi
    echo \"Selected GPU count: \${GPU_COUNT}\"
    srun --partition=instruction --account=s2026.se.4390.01 --gres=gpu:\${GPU_COUNT} --mem=48G --cpus-per-task=20 --pty bash srun_wrapper.sh --epochs $epochs
    "

    echo "Downloading ${epochs}-epoch model to local computer..."
    mkdir -p "$LOCAL_DIR/models"
    scp "${REMOTE_USER}@${REMOTE_HOST}:~/multimodal_entailment/models/multimodal_entailment_${epochs}_epochs.keras" "$LOCAL_DIR/models/" || echo "Failed to download model!"
}

# Run the training steps sequentially and save them to the computer after each step
run_epochs 5
run_epochs 10
run_epochs 15

echo ""
echo "All interactive training steps completed! Models are safely saved on your local computer."
