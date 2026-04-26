import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow imports from the dtw folder where load_multi_serve lives
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dtw"))
from load_data import load_multi_serve

DATA_DIR = os.path.join(os.path.dirname(__file__), "markers", "unmarked_edited")
csv_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

marker = "right_hand"

plt.figure(figsize=(14, 6))

for csv_path in csv_paths:
    label = os.path.splitext(os.path.basename(csv_path))[0]
    serve = load_multi_serve(csv_path)

    TX = serve[marker]["TX"]
    TY = serve[marker]["TY"]
    TZ = serve[marker]["TZ"]
    frames = serve["frames"]

    dX = np.diff(TX)
    dY = np.diff(TY)
    dZ = np.diff(TZ)
    speed = np.sqrt(dX**2 + dY**2 + dZ**2)
    smoothed_speed = pd.Series(speed).rolling(window=10, center=True).mean().values
    speed_frames = frames[:-1]

    plt.plot(speed_frames, smoothed_speed, linewidth=1.2, label=label)

plt.xlabel("Frame number")
plt.ylabel("Speed (mm / frame)")
plt.title(f"Hand speed over time — all serves  [marker: {marker}]")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()
