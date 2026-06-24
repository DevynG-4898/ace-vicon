# Project Setup Guide

Follow these steps to set up the ACE development environment.

---

## 1. Clone the Repository

git clone (https://github.com/allisonturnbow/ace-vicon)
cd ace-vicon

---

## 2. Create a Virtual Environment

python -m venv .venv

---

## 3. Activate the Virtual Environment

.venv\Scripts\Activate

You should see:
(.venv) in your terminal

## 4. Install Dependencies

pip install -r requirements.txt

---

## 5. Run the Project

From the project root:
python src/main.py

---

## Notes

- Always activate the virtual environment before running code.
- Do NOT commit the `.venv` folder.
- Raw Vicon CSV files should go in `data/raw/`.



python -m mediapipe_pose.tennis_video_analysis mediapipe_pose/tennis_clip_1.mp4
 python -m mediapipe_pose.tennis_video_analysis mediapipe_pose/tennis_clip_1.mp4

 python -m mediapipe_pose.pose_estimation.py mediapipe_pose/tennis_serve.jpg