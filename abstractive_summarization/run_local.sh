#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Activate the existing project venv
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
else
    echo "Creating new venv..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install requirements
pip install -U pip setuptools wheel
pip install -r requirements.txt

# Run for 10 epochs
echo "========================================="
echo "Starting local training (10 epochs)"
echo "========================================="
python3 train.py --epochs 1

# Run for 20 epochs
echo "========================================="
echo "Starting local training (20 epochs)"
echo "========================================="
python3 train.py --epochs 20

echo "Local training completed!"
