import pandas as pd

from constants import MARKER_ORDER

FILENAME_TO_MARKER = {
    "chest": "chest",
    "head": "head",
    "leftelbow": "left_elbow",
    "leftfoot": "left_foot",
    "lefthand": "left_hand",
    "lefthip": "left_hip",
    "leftknee": "left_knee",
    "leftshoulder": "left_shoulder",
    "rightelbow": "right_elbow",
    "rightfoot": "right_foot",
    "righthand": "right_hand",
    "righthip": "right_hip",
    "rightknee": "right_knee",
    "rightshoulder": "right_shoulder",
}


def load_single_serve(serve_dict):
    """Load a serve exported as one CSV file per marker.
    (like the data in plotting/markers/serve1)

    Args:
        serve_dict: dict mapping anatomical marker name -> filepath (str or Path)
                    e.g. {'head': 'serve1/forehead.csv', 'chest': 'serve1/chest.csv', ...}

    Returns:
        {
            'frames': np.ndarray,
            'head':   {'TX': np.ndarray, 'TY': np.ndarray, 'TZ': np.ndarray},
            ...
        }
    """
    result = {}
    frames = None

    for marker_name, filepath in serve_dict.items():
        raw = pd.read_csv(filepath, header=None, dtype=str)
        # Row 0: track name  Row 1: TX/TY/TZ labels  Row 2: units
        # Row 3+: Frame, SubFrame, TX, TY, TZ
        data = raw.iloc[3:].reset_index(drop=True)

        if frames is None:
            frames = pd.to_numeric(data.iloc[:, 0], errors="coerce").values

        result[marker_name] = {
            "TX": pd.to_numeric(data.iloc[:, 2], errors="coerce").values,
            "TY": pd.to_numeric(data.iloc[:, 3], errors="coerce").values,
            "TZ": pd.to_numeric(data.iloc[:, 4], errors="coerce").values,
        }

    result["frames"] = frames
    return result


def load_multi_serve(filepath):
    """Load a serve from a single CSV file containing all markers.
    (like data in plotting/markers/unmarked_edited/serve2.csv)

    Markers are assigned anatomical names from MARKER_ORDER by position.

    Args:
        filepath: path (str or Path) to a multi-marker Vicon CSV

    Returns:
        {
            'frames': np.ndarray,
            'head':   {'TX': np.ndarray, 'TY': np.ndarray, 'TZ': np.ndarray},
            ...
        }
    """
    raw = pd.read_csv(filepath, header=None, dtype=str)

    # Row 0: track names  Row 1: TX/TY/TZ labels  Row 2: units  Row 3+: data
    n_markers = (raw.shape[1] - 2) // 3
    marker_names = [
        MARKER_ORDER[i] if i < len(MARKER_ORDER) else f"Marker_{i + 1}"
        for i in range(n_markers)
    ]

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
