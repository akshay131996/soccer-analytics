"""Pitch homography: broadcast pixels -> 2D tactical map (105m x 68m pitch).

A homography is a 3x3 matrix mapping one plane to another. The pitch IS a plane, so
4+ point correspondences (corner flags, penalty-box corners, center spot...) let us
project every player's feet position onto a bird's-eye minimap. No deep learning —
this is the classical geometry that makes the deep learning useful.

The thing this module now insists on is knowing *how wrong it is*. A homography always
returns a matrix; it never tells you the matrix is nonsense. See ground_truth.html §04.
"""
import cv2
import numpy as np

PITCH_W, PITCH_H = 105.0, 68.0  # meters (FIFA standard-ish)
SCALE = 8  # pixels per meter on the minimap

# Above this mean reprojection error the mapping is not usable for metre-denominated
# stats. A player is ~0.5 m wide; being 3 m out means distances and possession are
# fiction. Chosen to be permissive -- it is a "this is broken" line, not a quality bar.
RESIDUAL_WARN_M = 3.0


class PitchMapper:
    """Maps image pixels to pitch metres, and reports its own reprojection error.

    `residual_m` is the mean distance, in metres, between where each supplied pitch
    point actually is and where the fitted homography puts its image counterpart.

    Read the caveat on `is_measurable` before trusting it.
    """

    def __init__(self, image_points, pitch_points, ransac_threshold_m=1.5):
        src = np.array(image_points, dtype=np.float32)
        dst = np.array(pitch_points, dtype=np.float32)
        if len(src) != len(dst):
            raise ValueError(f"got {len(src)} image points but {len(dst)} pitch points")
        if len(src) < 4:
            raise ValueError(f"need at least 4 correspondences, got {len(src)}")

        # With exactly 4 points the system is exactly determined: least squares fits
        # them perfectly whatever they are, so RANSAC has nothing to reject and the
        # residual is structurally zero. With 5+ the fit is over-determined, outliers
        # can be identified, and the residual finally carries information.
        self.n_points = len(src)
        self.is_measurable = self.n_points >= 5

        if self.is_measurable:
            self.H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=ransac_threshold_m)
            self.inliers = int(mask.sum()) if mask is not None else self.n_points
        else:
            self.H, _ = cv2.findHomography(src, dst)
            self.inliers = self.n_points

        if self.H is None:
            raise ValueError(
                "homography failed. The usual cause is degenerate correspondences: "
                "three or more points on a line, or all four bunched into a small "
                "region of the frame. Spread them as widely as the pitch allows."
            )

        # Reprojection error, in metres, because that is the unit a human can judge.
        projected = self.to_pitch(src)
        errors = np.linalg.norm(projected - dst, axis=1)
        self.residual_m = float(errors.mean())
        self.max_residual_m = float(errors.max())

        if self.is_measurable and self.residual_m > RESIDUAL_WARN_M:
            print(f"  WARNING: homography reprojection error is "
                  f"{self.residual_m:.2f} m (max {self.max_residual_m:.2f} m) across "
                  f"{self.inliers}/{self.n_points} inliers. Distances and possession "
                  f"derived from this are not trustworthy — recheck the correspondences.")

    def to_pitch(self, points_px):
        """Nx2 image points (use players' BOTTOM_CENTER = feet) -> Nx2 pitch meters."""
        pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def quality(self) -> dict:
        """Serialisable summary, for the stats dict and the API response."""
        return {
            "correspondences": self.n_points,
            "inliers": self.inliers,
            "mean_reprojection_error_m": round(self.residual_m, 3),
            "max_reprojection_error_m": round(self.max_residual_m, 3),
            # The honest caveat, carried alongside the number rather than in a docstring
            # nobody reads: with 4 points this error is zero by construction.
            "error_is_meaningful": self.is_measurable,
        }


# The pitch background never changes, but draw_minimap() used to rebuild it from
# scratch for every frame -- a fresh 840x544x3 allocation plus ~10 draw calls, 25
# times a second, for a static image.
_PITCH_CACHE = None


def draw_pitch():
    global _PITCH_CACHE
    if _PITCH_CACHE is not None:
        return _PITCH_CACHE.copy()

    w, h = int(PITCH_W * SCALE), int(PITCH_H * SCALE)
    img = np.full((h, w, 3), (60, 140, 60), dtype=np.uint8)
    white = (255, 255, 255)

    def line(p0, p1):
        cv2.line(img, tuple(int(v * SCALE) for v in p0), tuple(int(v * SCALE) for v in p1), white, 2)

    line((0, 0), (PITCH_W, 0)); line((0, PITCH_H), (PITCH_W, PITCH_H))
    line((0, 0), (0, PITCH_H)); line((PITCH_W, 0), (PITCH_W, PITCH_H))
    line((PITCH_W / 2, 0), (PITCH_W / 2, PITCH_H))
    cv2.circle(img, (int(PITCH_W / 2 * SCALE), int(PITCH_H / 2 * SCALE)),
               int(9.15 * SCALE), white, 2)
    for x0 in (0, PITCH_W - 16.5):  # penalty boxes
        y0 = (PITCH_H - 40.3) / 2
        cv2.rectangle(img, (int(x0 * SCALE), int(y0 * SCALE)),
                      (int((x0 + 16.5) * SCALE), int((y0 + 40.3) * SCALE)), white, 2)

    _PITCH_CACHE = img
    return img.copy()


def draw_minimap(team0_m, team1_m, ball_m=None):
    """Positions in meters -> rendered tactical map (BGR image)."""
    img = draw_pitch()
    for pts, color in ((team0_m, (255, 80, 80)), (team1_m, (80, 80, 255))):
        for x, y in np.asarray(pts).reshape(-1, 2):
            if 0 <= x <= PITCH_W and 0 <= y <= PITCH_H:
                cv2.circle(img, (int(x * SCALE), int(y * SCALE)), 7, color, -1)
                cv2.circle(img, (int(x * SCALE), int(y * SCALE)), 7, (0, 0, 0), 1)
    if ball_m is not None:
        x, y = np.asarray(ball_m).flatten()[:2]
        # A ball off the plane (in flight) maps outside the pitch; clamping would draw
        # a confident lie, so skip it instead.
        if 0 <= x <= PITCH_W and 0 <= y <= PITCH_H:
            cv2.circle(img, (int(x * SCALE), int(y * SCALE)), 5, (255, 255, 255), -1)
            cv2.circle(img, (int(x * SCALE), int(y * SCALE)), 5, (0, 0, 0), 1)
    return img
