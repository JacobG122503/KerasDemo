#!/bin/bash

# This script follows the user's proven example for single-login execution.
# It uploads code, runs training, waits, downloads results, and then shows a progress page.

REMOTE_USER="jacobgar"
REMOTE_HOST="nova-login-1.its.iastate.edu"
TENSORBOARD_PORT=6007

echo "Authenticating and establishing master connection..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" exit

echo "Uploading files to Nova cluster..."
rsync -avz --exclude 'models' --exclude '__pycache__' --exclude 'venv' --exclude '.venv' --exclude 'training_log_*.out' --exclude 'logs' . "${REMOTE_USER}@${REMOTE_HOST}:~/image_captioning/"

echo "Submitting job and waiting for it to finish..."
echo "(This will block until the training is 100% complete. Do not close your laptop!)"
ssh -o "ServerAliveInterval=60" "${REMOTE_USER}@${REMOTE_HOST}" << 'EOF'
  cd ~/image_captioning
  sbatch --wait submit_training.sh
EOF

echo "Training finished! Downloading models, logs, and TensorBoard data..."
scp -r "${REMOTE_USER}@${REMOTE_HOST}:~/image_captioning/models" "/Users/jacobg/Projects/School/KerasDemo/image_captioning/"
scp -r "${REMOTE_USER}@${REMOTE_HOST}:~/image_captioning/logs" "/Users/jacobg/Projects/School/KerasDemo/image_captioning/"

echo "Success! The models and logs are now in your local folder."