#!/bin/bash

# 1. Upload the code to your home directory on the cluster
echo "Uploading files to Nova cluster..."
rsync -avz --exclude 'models' --exclude '__pycache__' --exclude 'venv' --exclude '.venv' --exclude 'training_log_*.out' /Users/jacobg/Projects/School/KerasDemo/sequence_to_sequence_addition jacobgar@nova-login-1.its.iastate.edu:~

# 2. Connect via SSH, move to the work directory, and run the job
echo "Submitting job and waiting for it to finish..."
echo "(This will block until the training is 100% complete. Do not close your laptop!)"

# ServerAliveInterval prevents the SSH connection from timing out during long training
ssh -o ServerAliveInterval=60 jacobgar@nova-login-1.its.iastate.edu << 'EOF'
  cd ~/sequence_to_sequence_addition
  
  # Submit the job and wait for it to finish
  sbatch --wait submit_training.sh
EOF

# 3. Download the trained models and logs back to your Mac
echo "Training finished! Downloading models and logs back to your Mac..."
scp "jacobgar@nova-login-1.its.iastate.edu:~/sequence_to_sequence_addition/models/*.keras" /Users/jacobg/Projects/School/KerasDemo/sequence_to_sequence_addition/models/
scp "jacobgar@nova-login-1.its.iastate.edu:~/sequence_to_sequence_addition/training_log_*.out" /Users/jacobg/Projects/School/KerasDemo/sequence_to_sequence_addition/

echo "Success! The models and training logs are now in your local folder."