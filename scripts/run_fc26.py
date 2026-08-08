import json
import sys

import cv2

sys.path.insert(0, "src")

SRC = "fc26_match.mp4"
SEG = "fc26_segment.mp4"
START, COUNT = 1200, 1500          # ~25 s of gameplay at 59.94 fps

# ---- trim to a gameplay window ------------------------------------------------
# The clip opens on a pre-match splash screen. Fitting team clusters on a title card
# is exactly the failure collect_crops warns about, so cut to real play first.
cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_POS_FRAMES, START)

out = cv2.VideoWriter(SEG, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
written = 0
while written < COUNT:
    ok, frame = cap.read()
    if not ok:
        break
    out.write(frame)
    written += 1
cap.release()
out.release()
print("segment: {} frames, {}x{} @ {:.2f} fps".format(written, w, h, fps))

# ---- run the pipeline ---------------------------------------------------------
from pipeline import process_video

stats = process_video(
    source=SEG,
    output_path="outputs/fc26_annotated.mp4",
    weights="yolo26n.pt",
    keypoints=None,          # camera pans with play -- see note in the report
    imgsz=1280,
    conf=0.3,
    fit_frames=30,           # 180 frames scanned for the clustering fit
    verbose=True,
)
print()
print(json.dumps(stats, indent=2))
