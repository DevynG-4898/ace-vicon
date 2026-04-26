import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dtw"))
from constants import MARKER_ORDER

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
    ("right_knee", "right_foot"),
]

# ---------------------------------------------------------------------------
# Load barycenter and reconstruct marker dict
# ---------------------------------------------------------------------------
npy_path = os.path.join(os.path.dirname(__file__), "..", "dtw", "barycenter1.npy")
barycenter = np.load(npy_path)  # shape (n_frames, n_markers * 3)
n_frames = barycenter.shape[0]

# Slice columns back into per-marker TX/TY/TZ arrays
markers = {}
for i, name in enumerate(MARKER_ORDER):
    c = i * 3
    markers[name] = {
        "TX": barycenter[:, c],
        "TY": barycenter[:, c + 1],
        "TZ": barycenter[:, c + 2],
    }

# ---------------------------------------------------------------------------
# Fixed axis limits
# ---------------------------------------------------------------------------
all_x = barycenter[:, 0::3].ravel()
all_y = barycenter[:, 1::3].ravel()
all_z = barycenter[:, 2::3].ravel()


def padded_limits(data, pad=0.08):
    lo, hi = data.min(), data.max()
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


x_lim = padded_limits(all_x)
y_lim = padded_limits(all_y)
z_lim = padded_limits(all_z)

x_range = x_lim[1] - x_lim[0]
y_range = y_lim[1] - y_lim[0]
z_range = z_lim[1] - z_lim[0]

zoom_level = 1.0  # < 1 zooms in, > 1 zooms out; use +/= and - keys

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


def apply_axes():
    cx = (x_lim[0] + x_lim[1]) / 2
    cy = (y_lim[0] + y_lim[1]) / 2
    cz = (z_lim[0] + z_lim[1]) / 2
    hx = x_range / 2 * zoom_level
    hy = y_range / 2 * zoom_level
    hz = z_range / 2 * zoom_level
    ax.set_xlim(cx - hx, cx + hx)
    ax.set_ylim(cy - hy, cy + hy)
    ax.set_zlim(cz - hz, cz + hz)
    ax.set_box_aspect([hx * 2, hy * 2, hz * 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def on_key(event):
    global zoom_level
    if event.key in ("+", "="):
        zoom_level *= 0.8
    elif event.key == "-":
        zoom_level /= 0.8


fig.canvas.mpl_connect("key_press_event", on_key)
apply_axes()


def get_pos(joint, frame_idx):
    m = markers[joint]
    return (
        float(m["TX"][frame_idx]),
        float(m["TY"][frame_idx]),
        float(m["TZ"][frame_idx]),
    )


def update(frame_idx):
    ax.cla()
    apply_axes()
    ax.set_title(f"DTW Barycenter (average serve) — frame {frame_idx + 1} / {n_frames}")

    for joint in markers:
        x, y, z = get_pos(joint, frame_idx)
        if not any(np.isnan(v) for v in [x, y, z]):
            ax.scatter(x, y, z, s=20)

    for start, end in bones:
        if start not in markers or end not in markers:
            continue
        x0, y0, z0 = get_pos(start, frame_idx)
        x1, y1, z1 = get_pos(end, frame_idx)
        if any(np.isnan(v) for v in [x0, y0, z0, x1, y1, z1]):
            continue
        ax.plot([x0, x1], [y0, y1], [z0, z1], "b-", linewidth=1.5)


ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=33, repeat=True)

plt.show()
