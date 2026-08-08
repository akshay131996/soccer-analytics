# Soccer Analytics from Broadcast Video ⚽

Broadcast clip in → tracked players coloured by team, a live tactical minimap,
per-player distance covered, and possession percentages out.

Detection (YOLO26) → tracking (ByteTrack) → unsupervised team assignment → pitch
homography → derived statistics. The interpretation layer is classical geometry, not
learning: a pitch is a plane, so one 3×3 matrix converts broadcast pixels into metres.

## Start here

**📖 [Read the write-ups online](https://akshay131996.github.io/soccer-analytics/)** — the HTML pages
below render with all their interactive figures working.

| Document | What it's for |
|---|---|
| **[`run_all.ipynb`](run_all.ipynb)** | The build. Every stage opens with an explanation of *why* before the code that does it — run it top to bottom on a GPU. |
| **[`walkthrough.html`](https://akshay131996.github.io/soccer-analytics/walkthrough.html)** | Architecture and design document: data contracts, failure modes, trade-offs, with interactive homography and tracking figures. Written *before* the code. |
| **[`label_assignment.html`](https://akshay131996.github.io/soccer-analytics/label_assignment.html)** | Detector internals — why anchors and NMS both disappeared, and how one-to-one matching actually works. |
| **[`concepts.html`](https://akshay131996.github.io/soccer-analytics/concepts.html)** | The four techniques doing the real work — tracking, projective geometry, embeddings, detection metrics — each taken apart, then composed in an interactive concept map. |

All three HTML pages are self-contained: they work from the link above, or clone the repo
and open them in any browser. Viewing them on github.com shows raw source, not the page.

## Run it

```bash
pip install -r requirements.txt
jupyter lab run_all.ipynb          # or upload to Colab
```

Set `SOURCE` in §1 to your own clip. Without one it falls back to a public basketball
clip — a genuine stand-in, since it's a two-team sport from an elevated fixed camera, so
detection, tracking and team clustering all exercise properly. Only pitch geometry differs.

Command line equivalent, once you have keypoints:

```bash
python src/pipeline.py --source match.mp4 --keypoints keypoints.json --imgsz 1280
```

## How it works

| Stage | Approach | The interesting part |
|---|---|---|
| **Detection** | YOLO26, classes resolved **by name** | Index 0 is `person` in COCO but `ball` in a soccer checkpoint — see below |
| **Tracking** | ByteTrack (Kalman + Hungarian on IoU) | Keeps *low-confidence* boxes for a second matching pass, which is what rescues tracks through occlusion |
| **Teams** | Crop → embed → k-means, k=2 | Fully unsupervised. No labels, no training, milliseconds |
| **Pitch** | Homography from 4+ point correspondences | A pitch is 105 × 68 m by regulation — the ground truth is free |
| **Stats** | Distance, speed, possession | Every one of them is easy to compute meaninglessly; see the caveats |

### Two bugs fixed before they could bite

**Class indices.** The original scaffold hardcoded `person_class = 0`. Correct for COCO,
silently catastrophic with a fine-tuned soccer model where index 0 is the *ball* — it
would track the ball as a squad of players and still emit a full set of plausible
statistics. Classes are now resolved by name in `class_ids_for()`, which fails loudly
rather than quietly detecting the wrong thing. This exact mistake invalidated the
baseline in [traffic-lens](https://github.com/akshay131996/traffic-lens).

**Per-frame re-embedding.** Team was being recomputed for every player on every frame.
A player's team doesn't change, so it's now classified once per `tracker_id` and cached —
which removes the dominant cost when using the SigLIP embedder *and* stops the assignment
flickering between frames.

## Reading the numbers honestly

- **Distance** discards per-frame steps over 2 m (180 km/h at 25 fps — physically
  impossible, so it's an ID-switch teleport rather than motion). That makes reported
  distance a **lower bound**, not a measurement.
- **Possession** requires a 60% majority over a half-second window; raw nearest-player
  flickers every frame when two players contest the ball.
- **Calibration** is only as good as the landmarks you clicked. Map a point you *didn't*
  use for fitting and measure the error before trusting any metric quantity.
- **Sanity anchor from outside the code:** a footballer covers ~10–12 km in 90 minutes.
  The notebook extrapolates your clip and flags implausible values.

## Results

First end-to-end runs, on 25 s (1500 frames) of EA FC 26 gameplay at `imgsz=1280`,
RTX 4000 Ada. Scripts in [`scripts/`](scripts/); raw numbers in `outputs/`.

### The detector

YOLO26n fine-tuned on [Roboflow football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
v20 — 4 classes, 298 train / 49 val images, 50 epochs at 1280 px, 8.5 min:

| class | AP50 | AP50-95 |
|---|---|---|
| player | **0.958** | 0.687 |
| referee | 0.853 | 0.524 |
| goalkeeper | 0.786 | 0.564 |
| **ball** | **0.499** | **0.222** |
| *mean* | *0.774* | *0.499* |

**Read the per-class column, not the mean.** 0.774 looks respectable and is carried
almost entirely by `player` at 0.958. The ball — the object every possession metric
depends on — scores half that. A single averaged number hides exactly the failure that
matters most.

### Fixing what the first run exposed

The first annotated video looked convincing and was wrong. Measuring before changing
anything ([`scripts/diagnose_fc26.py`](scripts/diagnose_fc26.py)):

```
duplicate detection pairs (IoU>0.6)   8.55% of 3744 detections
grass inside the torso crop           mean 31%, >30% in 42% of crops
k-means silhouette                    0.386  (weak)
cluster median hues                   108 (navy) and 11 (orange)
```

That last line corrected a wrong diagnosis: the clustering *was* separating the kits
correctly. The real fault was that team was decided from **one crop at track birth** —
with a silhouette of 0.386 that is close to a coin flip per track, and it then stuck.

| | v1 | v2 (fixes) | v3 (+ fine-tune) |
|---|---|---|---|
| tracks (~20 players present) | 308 | 208 | **182** |
| mean track length (frames) | 83.0 | **120.3** | 99.4 |
| team labels | mixed across both kits | consistent | consistent |
| keepers / referees | forced into a team | forced into a team | **own roles** |

Changes: mask pitch green out of the histogram (silhouette 0.386 → 0.475), vote over 15
observations instead of deciding once, NMS at IoU 0.7, and a tracker buffer that scales
with frame rate (ByteTrack's default 30 frames is 0.5 s at 60 fps).

v3 `roles_tracked`: team 0 **75**, team 1 **99**, goalkeeper **3**, referee **5** —
keepers and referees no longer polluting the clustering.

### What these numbers do not tell you

**182 tracks for ~20 players is still wrong, and this is not a measurement of tracking
quality.** ByteTrack matches on IoU with no camera-motion compensation, and FC 26's
camera pans with play, so every box moves between frames. Some of the 182 is legitimate
re-entry as players leave and rejoin frame; without ground-truth tracks there is no way
to separate the two. That needs SoccerNet and HOTA — see
[`ground_truth.html`](https://akshay131996.github.io/soccer-analytics/ground_truth.html).

Note also that v3's mean track length *fell* versus v2. The fine-tuned detector finds
more marginal, distant players, which creates additional short tracks. Better detection,
worse-looking tracking metric — which is why the two must be read together.

No calibration was supplied, so there is no minimap, no distances and no possession:
FC 26's camera moves, and the homography is fitted once and reused, so a single
calibration would be silently wrong for most of the clip.

## Status

- [x] Pipeline, team clustering, homography, stats — written and documented
- [x] Architecture, detector-internals, concepts and ground-truth explainers
- [x] Class-resolution and team-caching bugs fixed
- [x] **Run end-to-end** — 1500 frames of FC 26, annotated video out
- [x] **Fine-tuned player/ball/goalkeeper/referee detector** — per-class AP above
- [x] **Goalkeepers and referees excluded from clustering** (structural, not tuning)
- [x] Homography reports its own reprojection error in metres, with RANSAC at 5+ points
- [ ] Ball tracking — AP50 0.499 is not good enough to trust possession
- [ ] Camera-motion compensation, or re-ID, to stop track fragmentation
- [ ] HOTA against SoccerNet ground truth, replacing the track-length proxy
- [ ] Learned pitch keypoints, so calibration works without clicking
- [ ] Demo video + writeup
