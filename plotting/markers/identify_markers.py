"""
identify_markers.py

Plot a CSV file labeling each point as Marker_1, Marker_2, etc. so you can
visually identify which marker is which body part.

Change csv_path at the bottom to whichever file you want to identify.
Use the slider to step through frames. Rotate the 3D view to inspect.
Once you know the order, create a matching <N>_order.py in multi/.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import numpy as np
import pandas as pd
import os

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#f7b733",
    "#00c957",
    "#6a0dad",
    "#961233",
]


def load_csv_raw(filepath):
    """Load CSV using column positions, labeling markers Marker_1..Marker_N."""
    raw = pd.read_csv(filepath, header=None, dtype=str)
    n_markers = (raw.shape[1] - 2) // 3
    marker_names = [f"Marker_{i + 1}" for i in range(n_markers)]

    data_rows = raw.iloc[3:].reset_index(drop=True)
    frames = pd.to_numeric(data_rows.iloc[:, 0], errors="coerce").values

    result = {"frames": frames}
    for i, name in enumerate(marker_names):
        c = 2 + i * 3
        result[name] = {
            "TX": pd.to_numeric(data_rows.iloc[:, c], errors="coerce").values,
            "TY": pd.to_numeric(data_rows.iloc[:, c + 1], errors="coerce").values,
            "TZ": pd.to_numeric(data_rows.iloc[:, c + 2], errors="coerce").values,
        }
    return result


def run(csv_path):
    data = load_csv_raw(csv_path)
    marker_names = [k for k in data if k != "frames"]
    n_frames = len(data["frames"])
    print(
        f"Loaded {len(marker_names)} markers, {n_frames} frames from {os.path.basename(csv_path)}"
    )

    all_x = np.concatenate([data[m]["TX"] for m in marker_names])
    all_y = np.concatenate([data[m]["TY"] for m in marker_names])
    all_z = np.concatenate([data[m]["TZ"] for m in marker_names])

    def padded(arr, pad=0.08):
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        margin = (hi - lo) * pad
        return lo - margin, hi + margin

    x_lim, y_lim, z_lim = padded(all_x), padded(all_y), padded(all_z)
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    z_range = z_lim[1] - z_lim[0]

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[i % len(COLORS)],
            markersize=8,
            label=name,
        )
        for i, name in enumerate(marker_names)
    ]

    fig = plt.figure(figsize=(11, 8))
    fig.suptitle(f"Marker identification — {os.path.basename(csv_path)}", fontsize=11)
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111, projection="3d")

    ax_slider = fig.add_axes([0.15, 0.08, 0.65, 0.03])
    frame_slider = widgets.Slider(
        ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1
    )

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
        ax.set_title(f"Frame {int(data['frames'][frame_idx])}", pad=2)

        for i, name in enumerate(marker_names):
            x = data[name]["TX"][frame_idx]
            y = data[name]["TY"][frame_idx]
            z = data[name]["TZ"][frame_idx]
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue
            color = COLORS[i % len(COLORS)]
            ax.scatter(x, y, z, s=30, color=color, zorder=5)
            ax.text(x, y, z, f" {name}", fontsize=7, color=color, zorder=6)

        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            fontsize=7,
            framealpha=0.7,
        )

    def update(frame_idx):
        if slider_dragging[0]:
            return
        current_frame[0] = frame_idx
        draw_frame(frame_idx)
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
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=33, repeat=True
    )
    plt.show()


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "multi", "6.csv")
    run(csv_path)
