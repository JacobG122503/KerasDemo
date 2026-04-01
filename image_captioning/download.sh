#!/bin/bash

REMOTE_USER="jacobgar"
REMOTE_HOST="nova-login-1.its.iastate.edu"

echo "Downloading models and logs from the cluster..."
echo "This will overwrite existing local 'models' and 'logs' directories."

# Set up a control master for SSH so we only have to authenticate once.
CONTROL_PATH=~/.ssh/control-%r@%h:%p
echo "Establishing master SSH connection..."
ssh -M -S "$CONTROL_PATH" -o "ControlPersist=60s" "${REMOTE_USER}@${REMOTE_HOST}" exit

# Use the master connection for scp
echo "Downloading models..."
scp -o "ControlPath=$CONTROL_PATH" -pr "${REMOTE_USER}@${REMOTE_HOST}:~/image_captioning/models" .
echo "Downloading logs..."
scp -o "ControlPath=$CONTROL_PATH" -pr "${REMOTE_USER}@${REMOTE_HOST}:~/image_captioning/logs" .

# Close the master connection
echo "Closing master SSH connection..."
ssh -S "$CONTROL_PATH" -O exit "${REMOTE_USER}@${REMOTE_HOST}"

echo ""
echo "Download complete!"
echo "The 'models' and 'logs' directories have been updated."
