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

echo "Submitting job to the cluster..."
ssh -o "ServerAliveInterval=60" "${REMOTE_USER}@${REMOTE_HOST}" 'cd ~/image_captioning && sbatch submit_training.sh'

echo ""
echo "Job submitted! You will receive an email when it is complete."
echo "You can close this window. To check job status, SSH to the cluster and run 'squeue -u ${REMOTE_USER}'"
echo "To download your models later, you can use a command like:"
echo "scp -r ${REMOTE_USER}@${REMOTE_HOST}:~/image_captioning/models ."
echo "scp -r ${REMOTE_USER}@${REMOTE_HOST}:~/image_captioning/logs ."