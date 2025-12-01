# RL Mini Game - PPO on CartPole

A complete, production-ready reinforcement learning project using Stable-Baselines3 PPO to train an agent on CartPole-v1.

## Features

- PPO agent with evaluation and checkpointing callbacks
- Reproducible training with seeding
- Model evaluation with statistics
- Video recording of trained agent
- Custom environment example
- Docker support
- CI/CD pipeline with GitHub Actions

## Setup

### Local Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train the PPO agent on CartPole-v1:

```bash
python src/train.py --env CartPole-v1 --total-timesteps 50000 --seed 42
```

Quick smoke test (2000 steps):
```bash
python src/train.py --env CartPole-v1 --total-timesteps 2000 --seed 42
```

### Evaluation

Evaluate a trained model:

```bash
python src/evaluate.py --model models/best/best_model.zip --env CartPole-v1 --n-episodes 10
```

### Recording

Record a video demonstration:

```bash
python src/record.py --model models/best/best_model.zip --env CartPole-v1 --output demos/demo.mp4
```

Convert MP4 to GIF:
```bash
ffmpeg -i demos/demo.mp4 -vf "fps=10,scale=320:-1:flags=lanczos" demos/demo.gif
```

### Custom Environment

To use a custom environment:

1. Define your environment in `src/envs/` (see `simple_env.py` example)
2. Register it using `scripts/register_simple_env.py`
3. Train with: `python src/train.py --env SimpleEnv-v0`

## Docker

Build and run with Docker:

```bash
docker build -t rl-mini-game .
docker run rl-mini-game
```

## Testing

Run smoke tests:

```bash
pytest tests/
```

## Project Structure

```
.
├── src/
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   ├── record.py         # Video recording script
│   └── envs/
│       └── simple_env.py # Custom environment example
├── scripts/
│   └── register_simple_env.py
├── tests/
│   └── test_smoke.py     # Smoke tests
├── models/               # Saved models (gitignored)
├── logs/                 # TensorBoard logs (gitignored)
├── demos/                # Recorded videos (gitignored)
├── requirements.txt
├── .gitignore
├── Dockerfile
└── run.sh
```

## Output Directories

- `models/` - Checkpoints and best models
- `models/best/` - Best model based on evaluation
- `logs/` - TensorBoard logs
- `demos/` - Recorded demonstration videos

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

## Tips

- Adjust `--total-timesteps` based on your environment complexity
- Use `--seed` for reproducible results
- Check `models/best/` for the best performing model
- Evaluation callback saves the best model automatically
