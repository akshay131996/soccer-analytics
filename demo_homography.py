"""Standalone homography demo — no video needed. Fabricates a camera view of a pitch,
maps synthetic player positions to the tactical map, saves a figure. Run me first:

    python demo_homography.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from homography import PitchMapper, draw_minimap  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

# Pretend we clicked 4 landmarks in a broadcast frame (pixels)...
image_points = [[240, 210], [1680, 200], [1900, 950], [40, 970]]
# ...and these are the same 4 spots on the real pitch (meters): the four corners.
pitch_points = [[0, 0], [105, 0], [105, 68], [0, 68]]

mapper = PitchMapper(image_points, pitch_points)

# Synthetic 'detections' in image space: two teams of 5, one ball.
rng = np.random.default_rng(7)
team0_px = rng.uniform([300, 300], [900, 900], size=(5, 2))
team1_px = rng.uniform([900, 250], [1700, 900], size=(5, 2))
ball_px = np.array([[960, 540]])

minimap = draw_minimap(mapper.to_pitch(team0_px), mapper.to_pitch(team1_px),
                       mapper.to_pitch(ball_px)[0])
cv2.imwrite(str(OUT / "minimap_demo.png"), minimap)

check = mapper.to_pitch([[240, 210]])[0]
print(f"corner sanity check: image (240,210) -> pitch ({check[0]:.1f}, {check[1]:.1f}) m "
      "(should be ~0,0)")
print(f"saved {OUT / 'minimap_demo.png'} — this is the core of your tactical map.")
