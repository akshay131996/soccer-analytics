# Service layer — phone in, annotated video out

The Android app never runs inference. It uploads a clip, a GPU worker on RunPod
processes it, and the app collects the result later.

```
 app ──POST /jobs──────────────► control plane        (Cloudflare Worker)
 app ──PUT  upload_url─────────► R2                   (streamed, EU bucket)
 app ──POST /jobs/:id/start────► control plane ──/run──► RunPod Serverless
                                                          │ downloads input
                                                          │ process_video()
                                                          │ uploads output
 app ──GET  /jobs/:id──────────► control plane ──/status──┘
 app ──GET  result_url─────────► R2
```

## Why it's shaped this way

**RunPod Serverless is already the job queue.** `POST /run` returns a job id
immediately and `GET /status/{id}` reports progress, so there is no queue, no worker
pool and no job table to build. The control plane stores one small KV record per job,
purely so the app never sees a RunPod id.

**The control plane exists only because the app cannot hold secrets.** The RunPod API
key and the R2 credentials stay server-side; the app receives short-lived capability
URLs signed with HMAC. The token binds key + method + expiry, so a URL minted for
reading cannot be replayed to write.

**Serverless rather than a always-on pod.** A pod billed hourly costs the same whether
anyone uses the app or not. Scale-to-zero means an idle app costs nothing, which for a
portfolio project is the difference between shipping it and turning it off.

## Deploy

### 1. The worker image

Build from the **repo root** — the image needs `src/`:

```bash
docker build -t YOUR_DOCKERHUB_USER/soccer-worker:v1 -f service/Dockerfile .
docker push YOUR_DOCKERHUB_USER/soccer-worker:v1
```

Then in the RunPod console: **Serverless → New Endpoint**, point it at that image, pick
a **24 GB GPU (L4/A10)**, and set:

| Setting | Value | Why |
|---|---|---|
| Max workers | **2** | The only hard spend ceiling that actually exists. Billing alerts are alerts, not brakes. |
| Idle timeout | 5 s | You pay for idle. |
| Execution timeout | 300 s | Bounds a runaway job. |
| FlashBoot | on | Cuts cold start materially. |

### 2. Storage and control plane

```bash
cd service/control-plane
wrangler r2 bucket create soccer-videos --jurisdiction eu
wrangler kv namespace create JOBS          # paste the id into wrangler.toml
wrangler secret put RUNPOD_API_KEY
wrangler secret put SIGNING_SECRET         # any long random string
wrangler deploy
```

Then set a **24-hour lifecycle rule** on the bucket. It is simultaneously the cost
control and the data-retention policy — the two things you would otherwise have to
remember separately.

## API

```
POST /jobs                    -> { job_id, upload_url }
PUT  <upload_url>             (raw video body)
POST /jobs/:id/start          { keypoints?, imgsz?, conf? } -> { status }
GET  /jobs/:id                -> { status, progress, stats?, result_url?, code?, error? }
```

`status` is RunPod's: `IN_QUEUE`, `IN_PROGRESS`, `COMPLETED`, `FAILED`, `TIMED_OUT`.

Failures carry a `code` so the app can say something specific:

| Code | Meaning |
|---|---|
| `TOO_LONG` / `TOO_LARGE` | Clip exceeds the limit — trim on device first |
| `UNREADABLE_VIDEO` | Truncated, or HEVC that OpenCV wasn't built for |
| `PIPELINE` | No players found, or too few crops to fit two teams |
| `DOWNLOAD_FAILED` / `UPLOAD_FAILED` | Storage round-trip failed |
| `INTERNAL` | Unhandled — check the RunPod logs |

## Known gaps

These are real and deliberate, not oversights:

- **No auth.** Anyone with the URL can create jobs and spend your GPU budget. Before
  this is public it needs Firebase App Check or equivalent, plus a per-user quota.
  `Max workers = 2` is the only thing currently bounding the damage.
- **No calibration UI.** Without `keypoints`, `has_pitch_mapping` is false and you get
  tracking and team colours but no minimap, distances or possession. The pipeline
  degrades gracefully by design; the app needs a tap-four-corners screen to close it.
- **Possession needs a fine-tuned detector.** On COCO weights there is no `ball` class,
  so `ball_xy` is always `None` and possession never accumulates. The Roboflow
  `player / goalkeeper / referee / ball` fine-tune is the unblock.
- **Homography assumes a fixed camera.** It is fitted once and reused for the clip.
  Handheld phone footage violates that silently.
