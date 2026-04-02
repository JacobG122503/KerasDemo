#!/bin/bash
set -euo pipefail

REMOTE_USER="jacobgar"
REMOTE_HOST="nova-login-1.its.iastate.edu"
REMOTE_DIR="~/atariBreakout"

echo "Downloading models and logs from the cluster..."
echo "This will replace local 'models' and 'logs' with the cluster copies."

TMP_DIR=$(mktemp -d)
cleanup() {
	rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "Authenticating (you will only be prompted once)..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" '
set -euo pipefail
cd ~/atariBreakout
mkdir -p models logs
echo "Latest remote training logs:" >&2
ls -lt logs/training_log_*.out 2>/dev/null | head -10 >&2 || echo "No remote training_log_*.out files found yet." >&2
tar -czf - models logs
' | tar -xzf - -C "${TMP_DIR}"

rm -rf models logs
mv "${TMP_DIR}/models" ./models
mv "${TMP_DIR}/logs" ./logs

echo ""
echo "Latest local training logs after sync:"
ls -lt logs/training_log_*.out 2>/dev/null | head -10 || echo "No local training_log_*.out files found."

echo ""
echo "Download complete!"
echo "The 'models' and 'logs' directories have been updated."
