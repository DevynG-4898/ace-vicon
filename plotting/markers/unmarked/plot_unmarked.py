import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from load_unmarked import load_unmarked_csv

bones = [
    ("head", "chest"),
    ("chest", "left_shoulder"),
    ("chest", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "left_hand"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
    ("left_hip", "right_hip"),
    ("chest", "left_hip"),
    ("chest", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
    ("right_hip", "right_knee"),
]

csv_path = os.path.join(
    os.path.dirname(__file__), "..", "unmarked_edited", "serve2.csv"
)
data = load_unmarked_csv(csv_path)

marker_names = [k for k in data if k != "frames"]
n_frames = len(data["frames"])
print("Loaded marker names:", marker_names)

# Marker unique colors and CSV columns (0-indexed):
#   col 0 = frame number, col 1 = sub-frame (ignored)
#   each marker occupies 3 consecutive cols: TX, TY, TZ
#
#   #   Marker           Color                  CSV cols (TX / TY / TZ)
#   1.  head             #1f77b4  (blue)        2 / 3 / 4
#   2.  left_shoulder    #ff7f0e  (orange)      5 / 6 / 7
#   3.  right_elbow      #2ca02c  (green)       8 / 9 / 10
#   4.  left_elbow       #d62728  (red)         11 / 12 / 13
#   5.  chest            #9467bd  (purple)      14 / 15 / 16
#   6.  left_hand        #8c564b  (brown)       17 / 18 / 19
#   7.  right_shoulder   #e377c2  (pink)        20 / 21 / 22
#   8.  left_knee        #7f7f7f  (gray)        23 / 24 / 25
#   9.  right_knee       #bcbd22  (olive)       26 / 27 / 28
#   10. right_hand       #17becf  (cyan)        29 / 30 / 31
#   11. right_hip        #f7b733  (gold)        32 / 33 / 34
#   12. left_foot        #00c957  (spring green)35 / 36 / 37
#   13. left_hip         #6a0dad  (indigo)      38 / 39 / 40

MARKER_COLORS = {
    "head": "#1f77b4",
    "left_shoulder": "#ff7f0e",
    "right_elbow": "#2ca02c",
    "left_elbow": "#d62728",
    "chest": "#9467bd",
    "left_hand": "#8c564b",
    "right_shoulder": "#e377c2",
    "left_knee": "#7f7f7f",
    "right_knee": "#bcbd22",
    "right_hand": "#17becf",
    "right_hip": "#f7b733",
    "left_foot": "#00c957",
    "left_hip": "#6a0dad",
}

# Axis limits from all frames across all markers
all_x = np.concatenate([data[m]["TX"] for m in marker_names])
all_y = np.concatenate([data[m]["TY"] for m in marker_names])
all_z = np.concatenate([data[m]["TZ"] for m in marker_names])


def padded_limits(arr, pad=0.08):
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


x_lim = padded_limits(all_x)
y_lim = padded_limits(all_y)
z_lim = padded_limits(all_z)

x_range = x_lim[1] - x_lim[0]
y_range = y_lim[1] - y_lim[0]
z_range = z_lim[1] - z_lim[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


def apply_axes():
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def update(frame_idx):
    ax.cla()
    apply_axes()
    ax.set_title(f'Frame {int(data["frames"][frame_idx])}')

    # Draw markers
    for name in marker_names:
        x = data[name]["TX"][frame_idx]
        y = data[name]["TY"][frame_idx]
        z = data[name]["TZ"][frame_idx]
        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            ax.scatter(x, y, z, s=20, color=MARKER_COLORS.get(name, "#000000"))

    # Draw bones
    for start, end in bones:
        if start not in data or end not in data:
            continue
        x0, y0, z0 = (
            data[start]["TX"][frame_idx],
            data[start]["TY"][frame_idx],
            data[start]["TZ"][frame_idx],
        )
        x1, y1, z1 = (
            data[end]["TX"][frame_idx],
            data[end]["TY"][frame_idx],
            data[end]["TZ"][frame_idx],
        )
        if any(np.isnan(v) for v in [x0, y0, z0, x1, y1, z1]):
            continue
        ax.plot([x0, x1], [y0, y1], [z0, z1], "b-", linewidth=1.5)


apply_axes()
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=33, repeat=True)

plt.tight_layout()
plt.show()
