import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets
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

csv_path = os.path.join(os.path.dirname(__file__), "multi", "1.csv")
data = load_unmarked_csv(csv_path)

marker_names = [k for k in data if k != "frames"]
n_frames = len(data["frames"])
print("Loaded marker names:", marker_names)

MARKER_COLORS = {
    "head": "#1f77b4",
    "chest": "#ff7f0e",
    "left_shoulder": "#2ca02c",
    "right_shoulder": "#d62728",
    "left_elbow": "#9467bd",
    "right_elbow": "#8c564b",
    "left_hand": "#e377c2",
    "right_hand": "#7f7f7f",
    "left_hip": "#bcbd22",
    "right_hip": "#17becf",
    "left_knee": "#f7b733",
    "right_knee": "#00c957",
    "left_foot": "#6a0dad",
    "right_foot": "#961233",
}

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

legend_handles = [
    plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=name
    )
    for name, color in MARKER_COLORS.items()
]

fig = plt.figure(figsize=(10, 8))
# Reserve bottom space for slider and button
fig.subplots_adjust(bottom=0.2)
ax = fig.add_subplot(111, projection="3d")

# Slider: occupies most of the bottom strip
ax_slider = fig.add_axes([0.15, 0.08, 0.65, 0.03])
frame_slider = widgets.Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1)

# Pause/Play button to the right of the slider
ax_button = fig.add_axes([0.82, 0.05, 0.1, 0.075])
pause_button = widgets.Button(ax_button, "Pause")

is_paused = [False]
current_frame = [0]
slider_dragging = [False]


def apply_axes():
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def draw_frame(frame_idx):
    ax.cla()
    apply_axes()
    ax.set_title(f'Frame {int(data["frames"][frame_idx])}')

    for name in marker_names:
        x = data[name]["TX"][frame_idx]
        y = data[name]["TY"][frame_idx]
        z = data[name]["TZ"][frame_idx]
        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            ax.scatter(x, y, z, s=20, color=MARKER_COLORS.get(name, "#000000"))

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=8,
        framealpha=0.7,
    )

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


def update(frame_idx):
    if slider_dragging[0]:
        return
    current_frame[0] = frame_idx
    draw_frame(frame_idx)
    # Sync slider without triggering on_slider_change
    frame_slider.eventson = False
    frame_slider.set_val(frame_idx)
    frame_slider.eventson = True


def on_pause(_):
    if is_paused[0]:
        ani.event_source.start()
        pause_button.label.set_text("Pause")
        is_paused[0] = False
    else:
        ani.event_source.stop()
        pause_button.label.set_text("Play")
        is_paused[0] = True
    fig.canvas.draw_idle()


def on_slider_change(_):
    frame_idx = int(frame_slider.val)
    current_frame[0] = frame_idx
    draw_frame(frame_idx)
    fig.canvas.draw_idle()


def on_slider_press(_):
    slider_dragging[0] = True


def on_slider_release(_):
    slider_dragging[0] = False
    on_slider_change(frame_slider.val)


pause_button.on_clicked(on_pause)
frame_slider.on_changed(on_slider_change)
frame_slider.ax.figure.canvas.mpl_connect("button_press_event", on_slider_press)
frame_slider.ax.figure.canvas.mpl_connect("button_release_event", on_slider_release)

apply_axes()
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=33, repeat=True)

plt.show()
