"""
serve_angles.py
Pure angle-calculation logic for tennis serve biomechanics.
No drawing, no video I/O — just math.

Updated to use 3D (x, y, z) coordinates for more accurate angle calculations,
especially when the serving arm goes behind the body (occlusion in 2D).
Note: MediaPipe z is relative depth — less accurate than x/y but still
improves results significantly over pure 2D for occluded joints.
"""

import math
import numpy as np


class TennisServeAnalyzer:
    """
    Calculates 6 key joint angles from a MediaPipe pose for tennis serve analysis.

    Supports right-handed (default) and left-handed serves via the `hand` parameter.
    Uses 3D (x, y, z) coordinates to handle occlusion from camera angle.
    """

    # MediaPipe landmark indices
    _LANDMARKS = {
        'right': {
            'opp_shoulder': 11,   # left shoulder
            'shoulder':     12,   # right shoulder
            'elbow':        14,
            'wrist':        16,
            'index':        20,
            'hip':          24,
            'knee':         26,
            'ankle':        28,
        },
        'left': {
            'opp_shoulder': 12,   # right shoulder
            'shoulder':     11,   # left shoulder
            'elbow':        13,
            'wrist':        15,
            'index':        19,
            'hip':          23,
            'knee':         25,
            'ankle':        27,
        },
    }

    def __init__(self, pose_landmarks, hand: str = 'right'):
        """
        Args:
            pose_landmarks: MediaPipe pose landmark list for one person.
            hand:           'right' or 'left' — which arm is serving.
        """
        if hand not in ('right', 'left'):
            raise ValueError(f"hand must be 'right' or 'left', got '{hand}'")

        self.lm   = pose_landmarks
        self._idx = self._LANDMARKS[hand]

        self.prev_elbow_angle    = None
        self.prev_shoulder_angle = None
        self.prev_wrist_angle    = None
        self.prev_time           = None

        self.prev_elbow_vel      = None
        self.prev_shoulder_vel   = None
        self.prev_wrist_vel      = None

    # ------------------------------------------------------------------
    # Internal math helpers
    # ------------------------------------------------------------------

    def _pt3d(self, landmark) -> np.ndarray:
        """Extract (x, y, z) from a MediaPipe landmark as a numpy array."""
        return np.array([landmark.x, landmark.y, landmark.z])

    def _angle(self, p1, p2, p3) -> float:
        """
        Angle at p2 formed by p1-p2-p3 in 3D space (degrees).
        Uses x, y, z coordinates — improves accuracy when joints are
        occluded or behind the body relative to the camera.
        """
        a = self._pt3d(p1)
        b = self._pt3d(p2)
        c = self._pt3d(p3)

        v1, v2 = a - b, c - b

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cos = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return math.degrees(math.acos(cos))

    def _vertical_angle(self, p1, p2) -> float:
        """
        Deviation of the p1→p2 line from vertical in 3D (degrees).
        Uses x, y, z — accounts for forward/backward lean in depth.
        """
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z

        horizontal = math.sqrt(dx**2 + dz**2)
        return math.degrees(math.atan2(horizontal, abs(dy)))

    # ------------------------------------------------------------------
    # Individual angle accessors
    # ------------------------------------------------------------------

    def get_shoulder_angle(self) -> float:
        """Angle between torso line and upper arm (serving side) in 3D."""
        i = self._idx
        return self._angle(self.lm[i['opp_shoulder']],
                           self.lm[i['shoulder']],
                           self.lm[i['elbow']])

    def get_elbow_angle(self) -> float:
        """Flexion angle at the serving elbow in 3D."""
        i = self._idx
        return self._angle(self.lm[i['shoulder']],
                           self.lm[i['elbow']],
                           self.lm[i['wrist']])

    def get_wrist_angle(self) -> float:
        """Extension angle at the serving wrist in 3D."""
        i = self._idx
        return self._angle(self.lm[i['elbow']],
                           self.lm[i['wrist']],
                           self.lm[i['index']])

    def get_hip_rotation_angle(self) -> float:
        """
        Angle between shoulder line and hip line in 3D.
        Including z makes this much more accurate for hip rotation
        since rotation happens primarily in the horizontal plane.
        """
        l11, l12 = self.lm[11], self.lm[12]
        l23, l24 = self.lm[23], self.lm[24]

        sv = np.array([l12.x - l11.x, l12.y - l11.y, l12.z - l11.z])
        hv = np.array([l24.x - l23.x, l24.y - l23.y, l24.z - l23.z])

        norm_sv = np.linalg.norm(sv)
        norm_hv = np.linalg.norm(hv)

        if norm_sv < 1e-6 or norm_hv < 1e-6:
            return 0.0

        cos = np.clip(np.dot(sv, hv) / (norm_sv * norm_hv), -1.0, 1.0)
        return math.degrees(math.acos(cos))

    def get_knee_angle(self) -> float:
        """Flexion angle at the serving-side knee in 3D."""
        i = self._idx
        return self._angle(self.lm[i['hip']],
                           self.lm[i['knee']],
                           self.lm[i['ankle']])

    def get_trunk_lean_angle(self) -> float:
        """
        Forward lean of the torso from vertical in 3D.
        Now accounts for depth (z) so forward lean into the serve
        is captured, not just side-to-side lean.
        """
        mid_sh_x = (self.lm[11].x + self.lm[12].x) / 2
        mid_sh_y = (self.lm[11].y + self.lm[12].y) / 2
        mid_sh_z = (self.lm[11].z + self.lm[12].z) / 2

        mid_hp_x = (self.lm[23].x + self.lm[24].x) / 2
        mid_hp_y = (self.lm[23].y + self.lm[24].y) / 2
        mid_hp_z = (self.lm[23].z + self.lm[24].z) / 2

        dx = mid_sh_x - mid_hp_x
        dy = mid_sh_y - mid_hp_y
        dz = mid_sh_z - mid_hp_z

        horizontal = math.sqrt(dx**2 + dz**2)
        return math.degrees(math.atan2(horizontal, abs(dy)))

    # ------------------------------------------------------------------
    # Aggregate — angles
    # ------------------------------------------------------------------

    def get_all_angles(self) -> dict:
        """Return all 6 angles as a dictionary."""
        return {
            'shoulder_angle': self.get_shoulder_angle(),
            'elbow_angle':    self.get_elbow_angle(),
            'wrist_angle':    self.get_wrist_angle(),
            'hip_rotation':   self.get_hip_rotation_angle(),
            'knee_angle':     self.get_knee_angle(),
            'trunk_lean':     self.get_trunk_lean_angle(),
        }

    # ------------------------------------------------------------------
    # Aggregate — raw coordinates
    # ------------------------------------------------------------------

    def get_all_coordinates(self) -> dict:
        """
        Return raw (x, y, z) coordinates for all key landmarks
        plus visibility scores.
        Coordinates are normalized (0-1) as provided by MediaPipe.
        """
        lm = self.lm
        return {
            # ── Right side (serving arm) ──────────────────────────────
            'r_shoulder_x': lm[12].x, 'r_shoulder_y': lm[12].y, 'r_shoulder_z': lm[12].z,
            'r_elbow_x':    lm[14].x, 'r_elbow_y':    lm[14].y, 'r_elbow_z':    lm[14].z,
            'r_wrist_x':    lm[16].x, 'r_wrist_y':    lm[16].y, 'r_wrist_z':    lm[16].z,
            'r_hip_x':      lm[24].x, 'r_hip_y':      lm[24].y, 'r_hip_z':      lm[24].z,
            'r_knee_x':     lm[26].x, 'r_knee_y':     lm[26].y, 'r_knee_z':     lm[26].z,
            'r_ankle_x':    lm[28].x, 'r_ankle_y':    lm[28].y, 'r_ankle_z':    lm[28].z,

            # ── Left side (toss arm) ──────────────────────────────────
            'l_shoulder_x': lm[11].x, 'l_shoulder_y': lm[11].y, 'l_shoulder_z': lm[11].z,
            'l_elbow_x':    lm[13].x, 'l_elbow_y':    lm[13].y, 'l_elbow_z':    lm[13].z,
            'l_wrist_x':    lm[15].x, 'l_wrist_y':    lm[15].y, 'l_wrist_z':    lm[15].z,
            'l_hip_x':      lm[23].x, 'l_hip_y':      lm[23].y, 'l_hip_z':      lm[23].z,
            'l_knee_x':     lm[25].x, 'l_knee_y':     lm[25].y, 'l_knee_z':     lm[25].z,
            'l_ankle_x':    lm[27].x, 'l_ankle_y':    lm[27].y, 'l_ankle_z':    lm[27].z,

            # ── Visibility scores ─────────────────────────────────────
            'r_shoulder_vis': lm[12].visibility,
            'r_elbow_vis':    lm[14].visibility,
            'r_wrist_vis':    lm[16].visibility,
            'l_shoulder_vis': lm[11].visibility,
            'l_elbow_vis':    lm[13].visibility,
            'l_wrist_vis':    lm[15].visibility,
        }

    def print_analysis(self) -> dict:
        """Print a formatted coaching report and return the angle dict."""
        angles = self.get_all_angles()
        print("\n" + "=" * 50)
        print("TENNIS SERVE ANALYSIS (3D)")
        print("=" * 50)
        rows = [
            ("Shoulder (serving arm)", angles['shoulder_angle'], "80–120°  arm extension"),
            ("Elbow (serving arm)",    angles['elbow_angle'],    "90–120°  trophy position"),
            ("Wrist",                  angles['wrist_angle'],    "150–180° slight extension"),
            ("Hip Rotation",           angles['hip_rotation'],   "20–45°   good rotation"),
            ("Knee (back leg)",        angles['knee_angle'],     "140–170° slight bend"),
            ("Trunk Lean",             angles['trunk_lean'],     "5–20°    forward lean"),
        ]
        for i, (label, value, ideal) in enumerate(rows, 1):
            print(f"\n{i}. {label}: {value:.1f}°")
            print(f"   └─ Ideal: {ideal}")
        print("\n" + "=" * 50)
        return angles