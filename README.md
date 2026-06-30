# ACE – Vicon Tennis Serve Analysis

ACE is a motion capture–based tennis serve analysis system. It compares a user's serve to a reference player and provides quantitative feedback to help improve serve mechanics.

## Pipelines

### Video → 3D skeleton (MotionBERT)

```text
2d_video/*.mp4 → MediaPipe → MotionBERT → SkeletonSequence → normalization → FeatureSequence
```

Entry point: `src/motionbert/run_pipeline.py`

See [docs/motionbert_pipeline.md](docs/motionbert_pipeline.md), [docs/skeleton_pipeline.md](docs/skeleton_pipeline.md), and [docs/feature_pipeline.md](docs/feature_pipeline.md).

Optional full-body mesh output is available through MotionBERT's native `infer_wild_mesh.py` script (see MotionBERT upstream docs).

### Vicon marker segmentation

Entry point: `python -m segmentation` (from `src/` on `PYTHONPATH`, or via project venv)

Plotting tools live under `plotting/`.

## Setup

1. Follow [setup.md](setup.md)
2. Activate the virtual environment
3. Run pipelines from the project root

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-motionbert.txt

# Process videos in 2d_video/ and open the viewer
python src/motionbert/run_pipeline.py --view

# Vicon marker segmentation CLI
PYTHONPATH=src python -m segmentation --v2 --frames firstserve
```

## Repository structure

```text
.
├── 2d_video/              Input serve videos
├── docs/                  Usage documentation
├── dtw/                   Vicon DTW reference motion
├── external/              MotionBERT checkout
├── generated_motionbert/  Pipeline outputs
├── plotting/              Vicon marker data and visualization
├── src/
│   ├── features/          FeatureSequence extraction
│   ├── motionbert/        Video pose pipeline
│   ├── segmentation/      Serve phase segmentation
│   └── skeleton/          SkeletonSequence and normalization
├── tests/
└── webapp/                Flask CSV similarity prototype
```

## Tests

```bash
python -m pytest
```

## Team

Project ACE — Allison Turnbow, Max Gavin, Biplav Adhikari, Devyn Gayle, Maximiliano Barajas, Jaime Favela
