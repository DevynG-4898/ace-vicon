import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from load_unmarked import load_unmarked_csv

# ── Change this to plot a different marker ──────────────────────────────────
MARKER = "left_shoulder"
# Valid options: head, left_shoulder, right_shoulder, left_elbow, right_elbow,
#               left_hand, right_hand, chest, left_hip, right_hip,
#               left_knee, right_knee, left_foot, right_foot
# ────────────────────────────────────────────────────────────────────────────

csv_path = os.path.join(os.path.dirname(__file__), "multi", "1.csv")
data = load_unmarked_csv(csv_path)

if MARKER not in data:
    raise ValueError(
        f"Marker '{MARKER}' not found. Available: {[k for k in data if k != 'frames']}"
    )

n_frames = len(data["frames"])
marker_data = data[MARKER]

tx = marker_data["TX"]
ty = marker_data["TY"]
tz = marker_data["TZ"]


x_lim = (-1600, 1600)
y_lim = (-1500, 1500)
z_lim = (0, 1800)

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
    ax.set_title(f"{MARKER}  —  Frame {int(data['frames'][frame_idx])}")

    x, y, z = tx[frame_idx], ty[frame_idx], tz[frame_idx]
    if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
        ax.scatter(x, y, z, s=40, color="#1f77b4")


apply_axes()
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=33, repeat=True)

plt.tight_layout()
plt.show()
