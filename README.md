# PixelBrush Agent

RL agent learns to paint pixel art using CLIP as a reward signal.

## Architecture
- **Canvas Resolution**: 32 × 32 pixels
- **Action Space**: 24,576 discrete actions (place, fill, blend)
- **Reward Signal**: CLIP ViT-B/32 cosine similarity
- **Policy**: PPO with CNN encoder conditioned on CLIP text embeddings

## Installation
```bash
pip install gymnasium open_clip_torch torch torchvision pillow pyyaml gradio pytest
```

## Training
To start training:
```bash
python -m pixelbrush.train.trainer
```

## Demo
To run the Gradio demo:
```bash
python -m pixelbrush.demo.visualize
```

## Structure
- `pixelbrush/env/`: Gym environment and action logic.
- `pixelbrush/reward/`: CLIP reward signal wrapper.
- `pixelbrush/agent/`: Policy network and PPO algorithm.
- `pixelbrush/train/`: Training orchestrator and prompt bank.
- `pixelbrush/demo/`: Gradio visualization app.
- `config.yaml`: All hyperparameters.
