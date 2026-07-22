"""
reference_format_writer.py
Extracts the 14 "nthserve"-style points (head, chest, left/right
shoulder/elbow/hand/hip/knee/foot) from MediaPipe pose_world_landmarks
and writes them out in the same CSV format as the mocap reference
dataset (2_labeled.csv):

    Row 1: point names           (head, , , chest, , , leftshoulder, ...)
    Row 2: axis labels           (TX, TY, TZ, TX, TY, TZ, ...)
    Row 3: units                 (mm, mm, mm, ...)
    Data : Frame, Sub Frame, <42 tx/ty/tz values>

Point mapping (MediaPipe world-landmark index -> reference point):
    head          <- 0  (nose)
    chest         <- midpoint(11, 12)  (left_shoulder, right_shoulder)
    leftshoulder  <- 11
    rightshoulder <- 12
    leftelbow     <- 13
    rightelbow    <- 14
    lefthand      <- 15  (left_wrist)
    righthand     <- 16  (right_wrist)
    lefthip       <- 23
    righthip      <- 24
    leftknee      <- 25
    rightknee     <- 26
    leftfoot      <- 27  (left_ankle)
    rightfoot     <- 28  (right_ankle)

NOTE on units/frame: MediaPipe world landmarks are in meters, hip-centered.
The reference mocap data is in millimeters, in the mocap system's own lab
frame. `scale=1000` fixes units (m -> mm) but does NOT re-align the
coordinate origin/orientation between the two systems — absolute tx/ty/tz
won't numerically match the reference even after scaling. Relative
motion, distances, and joint angles remain valid to compare.

Landmarks below `visibility_threshold` are written as blank cells,
mirroring how the reference CSV leaves a point blank in frames where
it wasn't tracked.
"""

import csv

REFERENCE_POINT_ORDER = [
    "head", "chest", "leftshoulder", "rightshoulder",
    "leftelbow", "rightelbow", "lefthand", "righthand",
    "lefthip", "righthip", "leftknee", "rightknee",
    "leftfoot", "rightfoot",
]

# Direct point -> single landmark index
_DIRECT_IDX = {
    "head":          0,
    "leftshoulder":  11,
    "rightshoulder": 12,
    "leftelbow":     13,
    "rightelbow":    14,
    "lefthand":      15,
    "righthand":     16,
    "lefthip":       23,
    "righthip":      24,
    "leftknee":      25,
    "rightknee":     26,
    "leftfoot":      27,
    "rightfoot":     28,
}

# Point -> (landmark index a, landmark index b), averaged
_MIDPOINT_IDX = {
    "chest": (11, 12),
}


def extract_reference_point_values(world_landmarks, visibility_threshold: float = 0.3,
                                    scale: float = 1000.0) -> dict:
    """
    Build a dict {point_name: (tx, ty, tz) or None} for one frame of
    MediaPipe pose_world_landmarks.

    A value of None means the point should be written blank (occluded /
    below visibility_threshold), matching how the reference CSV handles
    untracked points.
    """
    values = {}

    for point, idx in _DIRECT_IDX.items():
        lm = world_landmarks[idx]
        if lm.visibility < visibility_threshold:
            values[point] = None
        else:
            values[point] = (lm.x * scale, lm.y * scale, lm.z * scale)

    for point, (idx_a, idx_b) in _MIDPOINT_IDX.items():
        lm_a = world_landmarks[idx_a]
        lm_b = world_landmarks[idx_b]
        if min(lm_a.visibility, lm_b.visibility) < visibility_threshold:
            values[point] = None
        else:
            tx = (lm_a.x + lm_b.x) / 2.0 * scale
            ty = (lm_a.y + lm_b.y) / 2.0 * scale
            tz = (lm_a.z + lm_b.z) / 2.0 * scale
            values[point] = (tx, ty, tz)

    return values


class ReferenceFormatWriter:
    """
    Accumulates per-frame point values and writes them out in the
    mocap-reference CSV format when done.

    Usage:
        writer = ReferenceFormatWriter()
        # per frame:
        writer.add_frame(frame_num, sub_frame, point_values_dict)  # or writer.add_blank_frame(frame_num, sub_frame)
        # at the end:
        writer.write(output_path)
    """

    def __init__(self):
        self._rows = []  # list of (frame, sub_frame, {point: (tx,ty,tz) or None})

    def add_frame(self, frame_num: int, sub_frame: int, point_values: dict):
        self._rows.append((frame_num, sub_frame, point_values))

    def add_blank_frame(self, frame_num: int, sub_frame: int = 0):
        self._rows.append((frame_num, sub_frame, {p: None for p in REFERENCE_POINT_ORDER}))

    def write(self, output_path: str):
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Row 1: point names, each spanning 3 columns
            row1 = ["Frame", "Sub Frame"]
            for point in REFERENCE_POINT_ORDER:
                row1 += [point, "", ""]
            writer.writerow(row1)

            # Row 2: axis labels
            row2 = ["", ""]
            for _ in REFERENCE_POINT_ORDER:
                row2 += ["TX", "TY", "TZ"]
            writer.writerow(row2)

            # Row 3: units
            row3 = ["Frames", "Frames"]
            for _ in REFERENCE_POINT_ORDER:
                row3 += ["mm", "mm", "mm"]
            writer.writerow(row3)

            # Data rows
            for frame_num, sub_frame, point_values in self._rows:
                row = [frame_num, sub_frame]
                for point in REFERENCE_POINT_ORDER:
                    val = point_values.get(point)
                    if val is None:
                        row += ["", "", ""]
                    else:
                        tx, ty, tz = val
                        row += [f"{tx:.3f}", f"{ty:.3f}", f"{tz:.3f}"]
                writer.writerow(row)

        print(f"Reference-format CSV written: {output_path} "
              f"({len(self._rows)} frames, {len(REFERENCE_POINT_ORDER)} points)")