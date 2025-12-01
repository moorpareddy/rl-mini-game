#!/bin/bash
set -e

echo "=== RL Mini Game Training Pipeline ==="

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Train the agent
echo "=== Step 1: Training ==="
python src/train.py --env CartPole-v1 --total-timesteps 50000 --seed 42

# Evaluate the trained model
echo "=== Step 2: Evaluation ==="
python src/evaluate.py --model models/best/best_model.zip --env CartPole-v1 --n-episodes 10

# Record a demonstration
echo "=== Step 3: Recording ==="
python src/record.py --model models/best/best_model.zip --env CartPole-v1 --output demos/demo.mp4

echo "=== Pipeline complete! ==="
echo "Check models/ for saved models"
echo "Check logs/ for TensorBoard logs"
echo "Check demos/ for recorded videos"
