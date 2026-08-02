# Soccer Analytics from Broadcast Video ⚽

Broadcast clip in → tracked players coloured by team, a live tactical minimap,
per-player distance covered, and possession percentages out.

Detection (YOLO26) → tracking (ByteTrack) → unsupervised team assignment → pitch
homography → derived statistics. The interpretation layer is classical geometry, not
learning: a pitch is a plane, so one 3×3 matrix converts broadcast pixels into metres.

## Start here

| Document | What it's for |
|---|---|
| **[`run_all.ipynb`](run_all.ipynb)** | The build. Every stage opens with an explanation of *why* before the code that does it — run it top to bottom on a GPU. |
| **[`walkthrough.html`](walkthrough.html)** | Architecture and design document: data contracts, failure modes, trade-offs, with interactive homography and tracking figures. Written *before* the code. |
| **[`label_assignment.html`](label_assignment.html)** | Detector internals — why anchors and NMS both disappeared, and how one-to-one matching actually works. |

The two HTML pages are self-contained; open them in any browser.

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

## Status

- [x] Pipeline, team clustering, homography, stats — written and documented
- [x] Architecture + detector-internals explainers
- [x] Class-resolution and team-caching bugs fixed
- [ ] Run end-to-end on real soccer footage
- [ ] Fine-tuned player/ball/referee detector (Roboflow football dataset)
- [ ] Ball tracking — expect this to be the hard part (~10 px, blurred, occluded)
- [ ] Learned pitch keypoints, so calibration works on any broadcast without clicking
- [ ] Demo video + writeup
