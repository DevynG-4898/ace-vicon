import importlib.util
import os
import pandas as pd


def _load_marker_order(csv_path):
    """Load MARKER_ORDER from the matching <stem>_order.py next to the CSV."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    order_path = os.path.join(os.path.dirname(csv_path), f"{stem}_order.py")
    if not os.path.exists(order_path):
        raise FileNotFoundError(
            f"No order file found for '{os.path.basename(csv_path)}'. "
            f"Expected: {order_path}"
        )
    spec = importlib.util.spec_from_file_location("_order", order_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MARKER_ORDER


def load_unmarked_csv(filepath):
    """
    Load an unmarked marker CSV file (Motive export format).

    Structure:
      Row 0: marker names (every 3 cols starting at col 2)
      Row 1: TX/TY/TZ labels
      Row 2: units (mm) — skipped
      Row 3+: data — col 0 = frame, col 1 = sub frame (ignored),
               then groups of 3 columns = TX, TY, TZ per marker

    Anatomical label order is read from a sidecar file next to the CSV,
    e.g. multi/1.csv → multi/1_order.py (defines MARKER_ORDER list).

    Returns a dict keyed by anatomical body-part name:
      {
        'frames': np.ndarray,
        'head': {'TX': np.ndarray, 'TY': np.ndarray, 'TZ': np.ndarray},
        'chest': {...},
        ...
      }
    """
    anatomical_labels = _load_marker_order(filepath)

    raw = pd.read_csv(filepath, header=None, dtype=str)

    n_cols = raw.shape[1]
    n_markers = (n_cols - 2) // 3
    marker_names = []
    for i in range(n_markers):
        if i < len(anatomical_labels):
            marker_names.append(anatomical_labels[i])
        else:
            marker_names.append(f"unknown_{i + 1}")

    data = raw.iloc[3:].reset_index(drop=True)
    frames = pd.to_numeric(data.iloc[:, 0], errors="coerce").values

    result = {"frames": frames}
    for i, name in enumerate(marker_names):
        c = 2 + i * 3
        result[name] = {
            "TX": pd.to_numeric(data.iloc[:, c], errors="coerce").values,
            "TY": pd.to_numeric(data.iloc[:, c + 1], errors="coerce").values,
            "TZ": pd.to_numeric(data.iloc[:, c + 2], errors="coerce").values,
        }

    return result


if __name__ == "__main__":
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "serve3.csv")
    data = load_unmarked_csv(csv_path)

    print(f"Frames: {len(data['frames'])} ({data['frames'][0]} - {data['frames'][-1]})")
    print(f"Markers ({len(data) - 1}):")
    for name in list(data.keys())[1:]:
        tx = data[name]["TX"]
        print(f"  {name}  TX[0]={tx[0]:.3f}  shape={tx.shape}")
