"""
serve_angles.py
Pure angle-calculation logic for tennis serve biomechanics.
No drawing, no video I/O — just math.

Uses 3D (x, y, z) coordinates for angle calculations. Callers should pass
in MediaPipe *world* landmarks (pose_world_landmarks), which are metric-scale
and hip-centered, rather than the normalized pose_landmarks — normalized
x/y are image-relative and z is only a rough depth proxy, which distorts
angle math as camera distance/zoom changes across frames.

UPDATED: Occlusion handling via landmark.visibility.
- Each landmark used in an angle calculation is checked against
  `visibility_threshold` before being trusted.
- If any landmark feeding an angle is below threshold, that angle is
  returned as float('nan') rather than a silently-wrong value computed
  from a guessed coordinate.
- `get_all_angles()` also exposes the minimum visibility among the
  landmarks used for each angle, so downstream code (CSV export,
  interpolation, accuracy comparison) can distinguish confident angles
  from occlusion-derived ones and report interpolated-vs-not accuracy
  separately.
"""

import math
import numpy as np


class TennisServeAnalyzer:
    """
    Calculates 6 key joint angles from a MediaPipe pose for tennis serve analysis.

    Supports right-handed (default) and left-handed serves via the `hand` parameter.
    Uses 3D (x, y, z) coordinates to handle occlusion from camera angle, and
    gates each angle on landmark visibility to avoid silently trusting
    guessed/occluded coordinates.
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

    NAN = float('nan')

    def __init__(self, pose_landmarks, hand: str = 'right',
                 visibility_threshold: float = 0.3):
        """
        Args:
            pose_landmarks:        MediaPipe pose landmark list for one person.
                                    Expected to be *world* landmarks (metric-scale)
                                    for accurate angle math, not normalized
                                    image-space landmarks.
            hand:                  'right' or 'left' — which arm is serving.
            visibility_threshold:  Minimum landmark.visibility required to
                                    trust a landmark in angle computation.
                                    Landmarks below this are treated as
                                    missing (occluded).
        """
        if hand not in ('right', 'left'):
            raise ValueError(f"hand must be 'right' or 'left', got '{hand}'")

        self.lm                    = pose_landmarks
        self._idx                  = self._LANDMARKS[hand]
        self.visibility_threshold  = visibility_threshold

    # ------------------------------------------------------------------
    # Internal math helpers
    # ------------------------------------------------------------------

    def _pt3d(self, landmark):
        """
        Extract (x, y, z) from a MediaPipe landmark as a numpy array,
        or None if the landmark's visibility is below threshold.
        """
        if landmark.visibility < self.visibility_threshold:
            return None
        return np.array([landmark.x, landmark.y, landmark.z])

    def _min_visibility(self, *landmarks) -> float:
        """Minimum visibility across a set of landmarks."""
        return min(l.visibility for l in landmarks)

    def _angle(self, p1, p2, p3):
        """
        Angle at p2 formed by p1-p2-p3 in 3D space (degrees).
        Returns float('nan') if any of the three points is below the
        visibility threshold, rather than computing from a guessed value.
        """
        a = self._pt3d(p1)
        b = self._pt3d(p2)
        c = self._pt3d(p3)

        if a is None or b is None or c is None:
            return self.NAN

        v1, v2 = a - b, c - b

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return self.NAN

        cos = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return math.degrees(math.acos(cos))

    def _vertical_angle(self, p1, p2):
        """
        Deviation of the p1→p2 line from vertical in 3D (degrees).
        Returns float('nan') if either landmark is below the visibility
        threshold.
        """
        if p1.visibility < self.visibility_threshold or \
           p2.visibility < self.visibility_threshold:
            return self.NAN

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

        if min(l11.visibility, l12.visibility, l23.visibility, l24.visibility) \
                < self.visibility_threshold:
            return self.NAN

        sv = np.array([l12.x - l11.x, l12.y - l11.y, l12.z - l11.z])
        hv = np.array([l24.x - l23.x, l24.y - l23.y, l24.z - l23.z])

        norm_sv = np.linalg.norm(sv)
        norm_hv = np.linalg.norm(hv)

        if norm_sv < 1e-6 or norm_hv < 1e-6:
            return self.NAN

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
        Accounts for depth (z) so forward lean into the serve is
        captured, not just side-to-side lean.
        """
        l11, l12 = self.lm[11], self.lm[12]
        l23, l24 = self.lm[23], self.lm[24]

        if min(l11.visibility, l12.visibility, l23.visibility, l24.visibility) \
                < self.visibility_threshold:
            return self.NAN

        mid_sh_x = (l11.x + l12.x) / 2
        mid_sh_y = (l11.y + l12.y) / 2
        mid_sh_z = (l11.z + l12.z) / 2

        mid_hp_x = (l23.x + l24.x) / 2
        mid_hp_y = (l23.y + l24.y) / 2
        mid_hp_z = (l23.z + l24.z) / 2

        dx = mid_sh_x - mid_hp_x
        dy = mid_sh_y - mid_hp_y
        dz = mid_sh_z - mid_hp_z

        horizontal = math.sqrt(dx**2 + dz**2)
        return math.degrees(math.atan2(horizontal, abs(dy)))

    # ------------------------------------------------------------------
    # Confidence (min visibility used per angle)
    # ------------------------------------------------------------------

    def get_angle_confidences(self) -> dict:
        """
        Return the minimum landmark visibility used to compute each angle.
        Useful downstream to flag which angles were interpolated / how
        trustworthy a given (possibly non-NaN) angle actually is.
        """
        i = self._idx
        l11, l12, l23, l24 = self.lm[11], self.lm[12], self.lm[23], self.lm[24]
        return {
            'shoulder_angle': self._min_visibility(self.lm[i['opp_shoulder']],
                                                     self.lm[i['shoulder']],
                                                     self.lm[i['elbow']]),
            'elbow_angle':    self._min_visibility(self.lm[i['shoulder']],
                                                     self.lm[i['elbow']],
                                                     self.lm[i['wrist']]),
            'wrist_angle':    self._min_visibility(self.lm[i['elbow']],
                                                     self.lm[i['wrist']],
                                                     self.lm[i['index']]),
            'hip_rotation':   self._min_visibility(l11, l12, l23, l24),
            'knee_angle':     self._min_visibility(self.lm[i['hip']],
                                                     self.lm[i['knee']],
                                                     self.lm[i['ankle']]),
            'trunk_lean':     self._min_visibility(l11, l12, l23, l24),
        }

    # ------------------------------------------------------------------
    # Aggregate — angles
    # ------------------------------------------------------------------

    def get_all_angles(self) -> dict:
        """Return all 6 angles as a dictionary. Values are float('nan')
        wherever the required landmarks were below the visibility
        threshold (occluded)."""
        return {
            'shoulder_angle': self.get_shoulder_angle(),
            'elbow_angle':    self.get_elbow_angle(),
            'wrist_angle':    self.get_wrist_angle(),
            'hip_rotation':   self.get_hip_rotation_angle(),
            'knee_angle':     self.get_knee_angle(),
            'trunk_lean':     self.get_trunk_lean_angle(),
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_analysis(self, angles: dict = None) -> dict:
        """
        Print a human-readable summary of the computed angles.

        Args:
            angles: Pre-computed angle dict (from get_all_angles()). If not
                    provided, computed on the fly.

        Returns:
            The angle dict that was printed.
        """
        if angles is None:
            angles = self.get_all_angles()

        confidences = self.get_angle_confidences()

        print("\nTENNIS SERVE ANGLE ANALYSIS")
        print("=" * 50)
        for name, value in angles.items():
            if value != value:  # NaN check
                print(f"  {name:16s}: N/A (occluded, min visibility "
                      f"{confidences[name]:.2f})")
            else:
                print(f"  {name:16s}: {value:6.2f}°  "
                      f"(min visibility {confidences[name]:.2f})")

        return angles