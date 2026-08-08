# Soccer Analytics — context for a fresh session

Broadcast clip in; tracked players coloured by team, a tactical minimap, per-player
distance and possession out. Detection (YOLO26) -> tracking (ByteTrack) -> unsupervised
team assignment -> pitch homography -> derived statistics.

Read this before changing anything. It records decisions and traps that are not
recoverable from the code.

## Current state — read this first

**The pipeline has never completed an end-to-end run.** Everything below is written and
syntax-checked; almost none of it is verified against real output. Treat any claim about
behaviour as unconfirmed unless there is an artifact in `outputs/` backing it.

**Possession has never produced a number, but not for the reason you might assume.**
Verified on the pod against `yolo26n.pt`:

```
players : [0]  ['person']
ball    : [32] ['sports ball']     <- COCO HAS a ball class
referee : []                       <- genuinely absent
```

So `ball_ids` is *not* empty. The real chain is: possession sits inside
`if mapper is not None`, `mapper` is only built when `keypoints` are passed, and
**keypoints have never been supplied** because they are hand-authored JSON with no
picker. The blocker is calibration, not the ball class.

What COCO genuinely lacks is `referee` and `goalkeeper` — the latter absorbed into
`person`, which is why k=2 misassigns them. And COCO's `sports ball` was trained on
large, centred, unblurred balls, so it will rarely fire on a ~10 px motion-blurred
football. The fine-tune is still worth doing; it is an accuracy and class-coverage win,
not the resurrection of dead code.

The unblock is `finetune_detector.ipynb`: Roboflow's four-class set
(`ball / player / goalkeeper / referee`). Run it on Colab (free T4) rather than a paid
pod. It needs a Roboflow API key from Colab Secrets as `ROBOFLOW_API_KEY`, and the
dataset `VERSION` number should be checked on the Universe page before running.

## Hard rules

**Never commit a secret.** No API keys, tokens or credentials in files, notebooks,
commit messages or published pages. The repo is public. Secrets come from Colab
`userdata`, GitHub Actions secrets, `wrangler secret put`, or env vars — referenced by
name, never by value. A key in git history is only fixed by rotating the credential.

**Resolve classes by NAME, never by index.** `class_ids_for()` exists because index 0 is
`person` in COCO and `ball` in a soccer checkpoint. Hardcoding `class_id == 0` tracks the
ball as a squad of players and still emits plausible statistics. This exact mistake
invalidated the baseline in the sibling
[traffic-lens](https://github.com/akshay131996/traffic-lens) project — see its
`LEARNINGS.md`.

**No GPU on the author's machine.** Never run or benchmark pipeline code locally. Static
checks (`ast.parse`, linting) are fine. Real execution goes to Colab or a GPU pod, and
artifacts come back to the repo.

## Layout

```
src/pipeline.py        process_video() — the core, imported by everything else
src/homography.py      PitchMapper (pixels -> metres), minimap rendering
src/team_cluster.py    HSV/SigLIP embedder + k-means k=2
finetune_detector.ipynb  the unblock: 4-class detector on Roboflow
run_all.ipynb          teaching notebook, explanation before each stage
service/               RunPod Serverless worker + Cloudflare control plane
*.html                 five explainer pages, served via GitHub Pages
```

## Design decisions worth not re-litigating

**Two passes over the video.** k-means cannot predict before it is fitted, and the kit
colours are unknown until players have been observed. Hence `collect_crops()` then the
render loop. It is why the pipeline cannot run on a live camera feed.

**`imgsz=1280`, not 640.** A football is ~10 px in a 1080p frame and effectively
disappears at 640. This is the single most expensive setting and the reason the app was
moved server-side.

**Team is cached per `tracker_id`.** Classifying once per track rather than per frame is
essential with the SigLIP embedder and also stops the assignment flickering.

**The distance guard discards ID-switch teleports**, which makes reported distance a
LOWER BOUND rather than a measurement. It is derived from `fps` (15 m/s ceiling) — it was
previously a fixed 2.0 m per frame, which meant 180 km/h at 25 fps but 432 km/h at 60 fps.

**Inference is server-side, not on-device.** Investigated and rejected: YOLO26's NMS-free
head is rejected by Android's LiteRT GPU delegate, 1280 px is ~4x the FLOPs of 640, and
sustained throttling costs a further ~44%. See `android/DEPENDENCIES.md`.

## Known-fragile areas

- **Calibration keypoints are hand-authored JSON.** No picker, no auto-detection. Without
  them `has_pitch_mapping` is false and there is no minimap, no distances, no possession.
  The pipeline degrades gracefully by design.
- **The homography is fitted once and reused for the whole clip.** Fine for a fixed
  broadcast camera; silently wrong for handheld footage.
- **With exactly 4 correspondences the reprojection residual is structurally zero** and
  carries no information. `PitchMapper.quality()["error_is_meaningful"]` flags this.
  Supply 5+ to get RANSAC and a real error estimate.
- **`k=2` cannot represent goalkeepers and referees.** This is structural, not a tuning
  problem. The fix is detecting them as their own classes so they never reach the
  clustering — which is what the fine-tune provides.
- **`supervision` is pinned `>=0.29,<0.30`** because ByteTrack was removed in 0.30, and
  **`numpy<2.4`** because 2.4 removed `np.cross` for 2-D vectors, which `LineZone` calls.

## Evaluation

Tracking quality is currently a **proxy** — unique IDs and mean track length. That cannot
distinguish "the tracker fragmented one player into three" from "more real players were
detected". Real metrics need ground-truth tracks: SoccerNet-Tracking, evaluated with
`TrackEval`, reporting HOTA with its DetA/AssA split. The reasoning is in
`ground_truth.html`; the plan is in `change-proposal.html`.

## Next steps, in order

1. Run `finetune_detector.ipynb` on Colab (~1 GPU hour).
2. Run the pipeline end-to-end with those weights — the first real output this project
   will have produced.
3. Commit artifacts, update the README with real numbers.
4. Then choose: SoccerNet evaluation, deploying `service/`, or the parked Android app.

## Conventions

- Explanation before code in notebooks — these double as teaching material.
- Document failures rather than quietly fixing them; the write-ups are the portfolio.
- Default branch is `master`, not `main`.
