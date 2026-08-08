"""Measure the three defects rather than guessing at them."""
import sys

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

sys.path.insert(0, "src")
from pipeline import class_ids_for, crops_from, PLAYER_NAMES  # noqa: E402
from team_cluster import HistogramEmbedder  # noqa: E402

SEG = "fc26_segment.mp4"
model = YOLO("yolo26n.pt")
pid = class_ids_for(model, PLAYER_NAMES)

# ---------------------------------------------------------------- 1. duplicates
print("=" * 62)
print("1. DUPLICATE DETECTIONS  (pairwise IoU within a single frame)")
print("=" * 62)


def iou_matrix(b):
    n = len(b)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            xa = max(b[i][0], b[j][0]); ya = max(b[i][1], b[j][1])
            xb = min(b[i][2], b[j][2]); yb = min(b[i][3], b[j][3])
            inter = max(0, xb - xa) * max(0, yb - ya)
            a1 = (b[i][2] - b[i][0]) * (b[i][3] - b[i][1])
            a2 = (b[j][2] - b[j][0]) * (b[j][3] - b[j][1])
            m[i, j] = inter / (a1 + a2 - inter + 1e-9)
    return m


dup_total = det_total = 0
for fi, frame in enumerate(sv.get_video_frames_generator(SEG)):
    if fi >= 200:
        break
    d = sv.Detections.from_ultralytics(model(frame, conf=0.3, imgsz=1280, verbose=False)[0])
    d = d[np.isin(d.class_id, pid)]
    det_total += len(d)
    if len(d) > 1:
        dup_total += int((iou_matrix(d.xyxy) > 0.6).sum())
print(f"  detections over 200 frames : {det_total}")
print(f"  pairs with IoU > 0.6       : {dup_total}   <- should be ~0 for NMS-free")
print(f"  duplicate rate             : {100*dup_total/max(det_total,1):.2f}%")

# ---------------------------------------------------------------- 2. crop content
print()
print("=" * 62)
print("2. WHAT IS ACTUALLY IN THE TORSO CROPS")
print("=" * 62)
crops = []
for fi, frame in enumerate(sv.get_video_frames_generator(SEG)):
    if fi >= 180 or len(crops) >= 300:
        break
    d = sv.Detections.from_ultralytics(model(frame, conf=0.3, imgsz=1280, verbose=False)[0])
    crops += crops_from(frame, d[np.isin(d.class_id, pid)])
print(f"  crops collected: {len(crops)}")

sizes = np.array([[c.shape[1], c.shape[0]] for c in crops])
print(f"  crop size  median {np.median(sizes,0).astype(int)}  min {sizes.min(0)}  max {sizes.max(0)}")

green_frac = []
for c in crops:
    h, w = c.shape[:2]
    torso = c[int(0.1 * h):int(0.55 * h), int(0.2 * w):int(0.8 * w)]
    if torso.size == 0:
        torso = c
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    # pitch green in OpenCV HSV: hue ~35-85, with some saturation
    g = ((hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 60)).mean()
    green_frac.append(g)
green_frac = np.array(green_frac)
print(f"  GRASS fraction of torso region: mean {green_frac.mean():.1%}  "
      f"median {np.median(green_frac):.1%}  >30% in {100*(green_frac>0.3).mean():.0f}% of crops")

# ---------------------------------------------------------------- 3. separability
print()
print("=" * 62)
print("3. DOES THE EMBEDDING ACTUALLY SEPARATE THE TWO KITS?")
print("=" * 62)
emb = HistogramEmbedder()
X = emb(crops)
from sklearn.cluster import KMeans  # noqa: E402
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
lab = km.labels_
print(f"  cluster sizes: {np.bincount(lab)}  (a healthy split is roughly balanced)")

# silhouette tells us whether the split is real or arbitrary
from sklearn.metrics import silhouette_score  # noqa: E402
print(f"  silhouette score: {silhouette_score(X, lab):.3f}   "
      f"(>0.5 strong, 0.25-0.5 weak, <0.25 essentially arbitrary)")

# ground-truth-ish check: mean hue of each cluster's torso, grass masked out
def kit_hue(c):
    h, w = c.shape[:2]
    t = c[int(0.1 * h):int(0.55 * h), int(0.2 * w):int(0.8 * w)]
    if t.size == 0:
        t = c
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    m = ~((hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 60))
    return np.median(hsv[:, :, 0][m]) if m.sum() > 10 else np.nan

hues = np.array([kit_hue(c) for c in crops])
for k in (0, 1):
    hk = hues[lab == k]
    hk = hk[~np.isnan(hk)]
    print(f"  cluster {k}: n={len(hk):3d}  median non-grass hue = {np.median(hk):.0f}"
          f"   (orange ~5-25, navy ~105-125)")

# what a grass-masked embedding would give instead
def masked_embed(cs):
    out = []
    for c in cs:
        h, w = c.shape[:2]
        t = c[int(0.1 * h):int(0.55 * h), int(0.2 * w):int(0.8 * w)]
        if t.size == 0:
            t = c
        hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
        mask = (~((hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 60))).astype(np.uint8) * 255
        hist = cv2.calcHist([hsv], [0, 1], mask, [18, 6], [0, 180, 0, 256])
        out.append(cv2.normalize(hist, None).flatten())
    return np.array(out)

Xm = masked_embed(crops)
kmm = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xm)
print()
print(f"  WITH GRASS MASKED -> cluster sizes {np.bincount(kmm.labels_)}, "
      f"silhouette {silhouette_score(Xm, kmm.labels_):.3f}")
for k in (0, 1):
    hk = hues[kmm.labels_ == k]
    hk = hk[~np.isnan(hk)]
    print(f"    cluster {k}: n={len(hk):3d}  median hue = {np.median(hk):.0f}")
