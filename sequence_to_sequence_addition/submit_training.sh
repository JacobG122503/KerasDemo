#!/bin/bash
#SBATCH --job-name=keras_addition
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=16G
#SBATCH --output=training_log_%j.out
#SBATCH --account=s2026.se.4390.01
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1

# Load any necessary modules (you may need to adjust this depending on Nova's module names)
module load python/3.10 2>/dev/null || true

# Set up a virtual environment on the cluster if it doesn't exist
if [ ! -d "cluster_venv" ]; then
    echo "Setting up cluster Python environment (this takes a minute on the first run)..."
    python3 -m venv cluster_venv
    source cluster_venv/bin/activate
    pip install --upgrade pip
fi
source cluster_venv/bin/activate
pip install keras tensorflow numpy tqdm

# Run the headless python training script
echo "Starting Keras Training..."
python3 train_headless.py