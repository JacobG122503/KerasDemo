#!/bin/bash

REMOTE_USER="jacobgar"
REMOTE_HOST="nova-login-1.its.iastate.edu"
REMOTE_DIR="~/image_captioning"

echo "Downloading models and logs from the cluster..."
echo "This will sync local 'models' and 'logs' with the cluster copies."

# Set up a control master for SSH so we only have to authenticate once.
CONTROL_PATH=~/.ssh/control-%r@%h:%p
echo "Establishing master SSH connection..."
ssh -M -S "$CONTROL_PATH" -o "ControlPersist=60s" "${REMOTE_USER}@${REMOTE_HOST}" exit 2>/dev/null

# Show remote latest training logs first so we can confirm freshness.
echo ""
echo "Latest remote training logs:"
ssh -S "$CONTROL_PATH" "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && ls -lt logs/training_log_*.out 2>/dev/null | head -10 || echo 'No remote training_log_*.out files found yet.'" 2>/dev/null

# Use the master connection for rsync
mkdir -p models logs
echo "Downloading models..."
rsync -avz --delete -e "ssh -S $CONTROL_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/models/" ./models/
echo "Downloading logs..."
rsync -avz --delete -e "ssh -S $CONTROL_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/logs/" ./logs/

echo ""
echo "Latest local training logs after sync:"
ls -lt logs/training_log_*.out 2>/dev/null | head -10 || echo "No local training_log_*.out files found."

# Close the master connection
echo "Closing master SSH connection..."
ssh -S "$CONTROL_PATH" -O exit "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null

echo ""
echo "Download complete!"
echo "The 'models' and 'logs' directories have been updated."
