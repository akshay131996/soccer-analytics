"""Pitch homography: broadcast pixels -> 2D tactical map (105m x 68m pitch).

A homography is a 3x3 matrix mapping one plane to another. The pitch IS a plane, so
4+ point correspondences (corner flags, penalty-box corners, center spot...) let us
project every player's feet position onto a bird's-eye minimap. No deep learning —
this is the classical geometry that makes the deep learning useful.
"""
import cv2
import numpy as np

PITCH_W, PITCH_H = 105.0, 68.0  # meters (FIFA standard-ish)
SCALE = 8  # pixels per meter on the minimap


class PitchMapper:
    def __init__(self, image_points, pitch_points):
        """image_points: Nx2 pixels; pitch_points: same N in meters (0,0 = top-left corner)."""
        src = np.array(image_points, dtype=np.float32)
        dst = np.array(pitch_points, dtype=np.float32)
        self.H, _ = cv2.findHomography(src, dst)
        if self.H is None:
            raise ValueError("homography failed — need >=4 non-collinear correspondences")

    def to_pitch(self, points_px):
        """Nx2 image points (use players' BOTTOM_CENTER = feet) -> Nx2 pitch meters."""
        pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)


def draw_pitch():
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
    return img


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
        cv2.circle(img, (int(x * SCALE), int(y * SCALE)), 5, (255, 255, 255), -1)
    return img
