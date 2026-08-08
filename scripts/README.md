# scripts/

Everything actually run on the GPU pod, so results are reproducible rather than
reconstructed from memory. These were previously piped over SSH stdin and existed
nowhere — which is the sort of thing that makes a result unrepeatable.

Run from the repo root on the pod:

```bash
cd /workspace/soccer-analytics
export YOLO_CONFIG_DIR=/tmp/Ultralytics      # else every run prints a config warning
python3 scripts/<name>.py
```

| Script | What it does |
|---|---|
| `pod_smoke.py` | Import and API checks — versions, `sv.ByteTrack`, `np.cross` on 2-D, the `PitchMapper` RANSAC path and the 4-point residual trap, `load_model` failing loudly. Run this first after any pod restart. |
| `run_fc26.py` | Trims the FC26 capture to a gameplay window (the clip opens on a splash screen) and runs `process_video` over it. |
| `diagnose_fc26.py` | Measures the three defects rather than guessing: duplicate-detection rate, how much grass is in the torso crops, and whether the embedding separates the kits (silhouette + per-cluster hue), with and without grass masking. |
| `finetune_pod.py` | Pod-side equivalent of `finetune_detector.ipynb`. Downloads the Roboflow 4-class set, asserts the classes resolve by name, trains at 1280, and writes per-class AP to `outputs/finetune_results.json`. |

## Secrets

`finetune_pod.py` reads the Roboflow key from **`/root/.roboflow_key`** — outside the
repo, mode 600, never committed. No script here takes a key as an argument or contains
one. If you run these elsewhere, create that file rather than editing the script.

## What these measured

`diagnose_fc26.py` on a 25 s FC26 segment, before any fixes:

```
duplicate pairs (IoU>0.6)   8.55% of 3744 detections
grass in torso crop         mean 31%, >30% in 42% of crops
k-means silhouette          0.386  (weak)
cluster median hues         108 (navy) and 11 (orange)   <- correctly separated
with grass masked           silhouette 0.475
```

That last pair is why the embedder now masks pitch green: a 23% improvement, measured
rather than assumed. The cluster hues also corrected an earlier wrong diagnosis — the
clustering was separating the kits correctly all along; the fault was that team was
decided from a single crop at track birth.
