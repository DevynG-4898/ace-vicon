# Project Setup Guide

Follow these steps to set up the ACE development environment.

## 1. Clone the Repository

```bash
git clone https://github.com/allisonturnbow/ace-vicon
cd ace-vicon
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

## 3. Activate the Virtual Environment

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your terminal prompt.

## 4. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-motionbert.txt
```

## 5. Run the Video Pipeline

From the project root:

```bash
python src/motionbert/run_pipeline.py --view
```

Place input videos in `2d_video/` before running.

## Notes

- Always activate the virtual environment before running code.
- Do not commit the `.venv` folder.
- Raw Vicon CSV files go under `plotting/markers/`.
- External model checkout: see `docs/motionbert_pipeline.md`.
