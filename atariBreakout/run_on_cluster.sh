#!/bin/bash
set -euo pipefail

# squeue -u jacobgar
# ssh jacobgar@nova-login-1.its.iastate.edu "tail -f ~/atariBreakout/logs/training_log_10285292.out"

# Upload code to HPC cluster, submit the SLURM job, and print status.
# Pass --no-resume to start training from scratch (ignores saved checkpoints).

NO_RESUME=false
for arg in "$@"; do
    if [ "$arg" = "--no-resume" ]; then
        NO_RESUME=true
    fi
done

REMOTE_USER="jacobgar"
REMOTE_HOST="nova-login-1.its.iastate.edu"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_UPLOAD_DIR="$(mktemp -d)"

cleanup_tmp() {
	rm -rf "$TMP_UPLOAD_DIR"
}
trap cleanup_tmp EXIT

echo "Preparing minimal upload bundle (code + resume files)..."

# Copy source tree without generated artifacts.
(
	cd "$LOCAL_DIR"
	COPYFILE_DISABLE=1 tar -cf - \
		--exclude='./models' \
		--exclude='./__pycache__' \
		--exclude='./venv' \
		--exclude='./.venv' \
		--exclude='./logs' \
		--exclude='./mp4' \
		--exclude='./mp4/*' \
		--exclude='./Keeping' \
		--exclude='./Keeping/*' \
		--exclude='./.git' \
		--exclude='./.git/*' \
		--exclude='./.DS_Store' \
		--exclude='*/.DS_Store' \
		--exclude='./._*' \
		--exclude='*/._*' \
		--exclude='./cluster_venv' \
		.
) | (
	cd "$TMP_UPLOAD_DIR"
	tar -xf -
)

# Add bare-minimum resume artifacts from local models.
mkdir -p "$TMP_UPLOAD_DIR/models"

RESUME_STATE_SRC="$LOCAL_DIR/models/training_state.json"
RESUME_MODEL_SRC=""

if ls "$LOCAL_DIR"/models/dqn_episode_*.keras >/dev/null 2>&1; then
	RESUME_MODEL_SRC=$(ls -1 "$LOCAL_DIR"/models/dqn_episode_*.keras \
		| sed -E 's|.*dqn_episode_([0-9]+)\.keras|\1 &|' \
		| sort -n \
		| tail -n 1 \
		| cut -d' ' -f2-)
elif [ -f "$LOCAL_DIR/models/dqn_best.keras" ]; then
	RESUME_MODEL_SRC="$LOCAL_DIR/models/dqn_best.keras"
elif [ -f "$LOCAL_DIR/models/dqn_final.keras" ]; then
	RESUME_MODEL_SRC="$LOCAL_DIR/models/dqn_final.keras"
fi

if [ "$NO_RESUME" = true ]; then
	echo "--no-resume flag set: skipping checkpoint upload. Training will start fresh."
elif [ -f "$RESUME_STATE_SRC" ] && [ -n "$RESUME_MODEL_SRC" ]; then
	cp "$RESUME_STATE_SRC" "$TMP_UPLOAD_DIR/models/"
	cp "$RESUME_MODEL_SRC" "$TMP_UPLOAD_DIR/models/"
	echo "Including resume files:"
	echo "  - $(basename "$RESUME_STATE_SRC")"
	echo "  - $(basename "$RESUME_MODEL_SRC")"
else
	echo "No complete local resume set found (training_state.json + checkpoint)."
	echo "Training will start fresh on the cluster."
fi

EXTRA_ARGS=""
if [ "$NO_RESUME" = true ]; then
	EXTRA_ARGS="--no-resume"
fi

echo "Authenticating (you will only be prompted once)..."

echo "Uploading + submitting in a single SSH session..."
if ! (
	cd "$TMP_UPLOAD_DIR"
	COPYFILE_DISABLE=1 tar -czf - .
) | ssh -o ServerAliveInterval=60 "${REMOTE_USER}@${REMOTE_HOST}" '
set -euo pipefail

cd ~

echo "Clearing remote home directory (keeping only .ssh)..."
# This only affects files on the cluster, never local files on your Mac.
find "$HOME" -mindepth 1 -maxdepth 1 \
	! -name .ssh \
	-exec rm -rf -- {} +

mkdir -p atariBreakout
cd atariBreakout

echo "Checking remote storage..."
AVAIL_KB=$(df -Pk ~ 2>/dev/null | awk "NR==2 {print \$4}")
echo "Home available space (KB): ${AVAIL_KB}"

mkdir -p logs models

echo "Extracting uploaded project files..."
tar -xzf -

echo "Submitting job to the cluster..."
GPU_CANDIDATES=$(sinfo -h -p instruction -o "%G" 2>/dev/null \
	| tr "," "\n" \
	| grep -Eo "gpu(:[^, ]+)*:[0-9]+" \
	| sed -E "s/.*:([0-9]+)$/\1/" \
	| grep -E "^[1-9][0-9]*$" \
	| sort -n \
	| uniq || true)

if [ -z "$GPU_CANDIDATES" ]; then
	GPU_CANDIDATES="1"
fi

echo "GPU candidates on partition: $(echo "$GPU_CANDIDATES" | tr "\n" " " | sed "s/ $//")"

'"EXTRA_ARGS=\"${EXTRA_ARGS}\""'

for GPU_COUNT in $(echo "$GPU_CANDIDATES" | sort -nr); do
	if [ "$GPU_COUNT" -lt 1 ]; then
		continue
	fi
	echo "Trying submission with ${GPU_COUNT} GPU(s)..."
	if SUBMIT_OUTPUT=$(sbatch --gres=gpu:${GPU_COUNT} submit_training.sh ${EXTRA_ARGS} 2>&1); then
		echo "$SUBMIT_OUTPUT"
		echo "Selected GPU count: ${GPU_COUNT}"
		exit 0
	fi
	echo "$SUBMIT_OUTPUT"
done

echo "All submission attempts failed." >&2
exit 1
'; then
	echo "Upload or job submission failed."
	exit 1
fi

echo ""
echo "Job submitted! You will receive an email when it is complete."
echo "You can close this window. To check job status, SSH to the cluster and run 'squeue -u ${REMOTE_USER}'"
echo "To download your models later:"
echo "  scp -r ${REMOTE_USER}@${REMOTE_HOST}:~/atariBreakout/models ."
echo "  scp -r ${REMOTE_USER}@${REMOTE_HOST}:~/atariBreakout/logs ."
