# Monocular Depth Estimation Project

This project implements and compares various monocular depth estimation models.

## Project Structure
```
MDE/
├── config/
├── data/
├── models/
├── trainers/
├── utils/
├── visualization/
└── scripts/
```

## Setup
1. Create virtual environment
2. Install requirements
3. Configure data paths
4. Run training

## Usage
- Training: `python scripts/train.py --config config/model_config.yaml`
- Evaluation: `python scripts/evaluate.py --model_path checkpoints/model.pth`
- Visualization: `python scripts/visualize.py --results_dir results/`
