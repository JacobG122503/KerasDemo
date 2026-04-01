#!/bin/bash
set -euo pipefail

# This script follows the user's proven example for single-login execution.
# It uploads code, runs training, waits, downloads results, and then shows a progress page.

REMOTE_USER="jacobgar"
REMOTE_HOST="nova-login-1.its.iastate.edu"
TENSORBOARD_PORT=6007

echo "Authenticating (you will only be prompted once)..."

echo "Uploading + submitting in a single SSH session..."
if ! tar -czf - \
	--exclude='./models' \
	--exclude='./__pycache__' \
	--exclude='./venv' \
	--exclude='./.venv' \
	--exclude='./logs' \
	--exclude='./cache' \
	--exclude='./training_log_*.out' \
	--exclude='./Flicker8k_Dataset' \
	--exclude='./Flickr8k_Dataset*' \
	--exclude='./Flickr8k_text*' \
	--exclude='./*.zip' \
	--exclude='./__MACOSX' \
	. | ssh -o ServerAliveInterval=60 "${REMOTE_USER}@${REMOTE_HOST}" '
set -euo pipefail

cd ~
mkdir -p image_captioning
cd image_captioning

echo "Checking remote storage and cleaning stale artifacts..."
AVAIL_KB=$(df -Pk ~ 2>/dev/null | awk "NR==2 {print \$4}")
INODE_FREE=$(df -i ~ 2>/dev/null | awk "NR==2 {print \$4}")
echo "Home available space (KB): ${AVAIL_KB}"
echo "Inode free: ${INODE_FREE}"

find . -maxdepth 1 -type f -name ".*.*" -delete 2>/dev/null || true
rm -rf cache __pycache__ 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true

mkdir -p logs models
if [ -d logs ]; then
  ls -1t logs/training_log_*.out 2>/dev/null | awk "NR>5" | xargs rm -f 2>/dev/null || true
  find logs -type f -name "events.out.tfevents.*" 2>/dev/null \
    | xargs ls -1t 2>/dev/null | awk "NR>10" | xargs rm -f 2>/dev/null || true
fi

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

for GPU_COUNT in $(echo "$GPU_CANDIDATES" | sort -nr); do
	if [ "$GPU_COUNT" -lt 1 ]; then
		continue
	fi
	echo "Trying submission with ${GPU_COUNT} GPU(s)..."
	if SUBMIT_OUTPUT=$(sbatch --gres=gpu:${GPU_COUNT} submit_training.sh 2>&1); then
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
echo "To download your models later, you can use a command like:"
echo "scp -r ${REMOTE_USER}@${REMOTE_HOST}:~/image_captioning/models ."
echo "scp -r ${REMOTE_USER}@${REMOTE_HOST}:~/image_captioning/logs ."