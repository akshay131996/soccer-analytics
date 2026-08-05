"""RunPod Serverless handler: video in, annotated video + stats out.

The phone never talks to this directly. Flow:

    app -> control plane -> presigned R2 URLs -> app uploads video to R2
    app -> control plane -> POST /run {video_url, upload_url}   (this handler)
    app -> control plane -> GET /status/{id}  until COMPLETED
    app -> downloads the annotated video from R2

RunPod Serverless already provides the async job machinery -- `/run` returns a job id
immediately and `/status/{id}` reports progress -- so there is deliberately no queue,
no database and no job table here. That is the whole reason this file is short.

Design notes worth keeping:

* Every job gets its OWN temp directory. The Gradio app this replaces wrote to a fixed
  filename in the system temp dir, which under concurrency meant two users overwriting
  each other's video -- and one user being handed another user's footage. That is a
  privacy bug, not just a race.
* Duration is validated with ffprobe against the DECODED metadata, not the file size.
  A heavily compressed 20-minute clip can be 40 MB and cost 40x the expected GPU time.
* Errors are returned as structured codes, not raised. A raised exception becomes an
  opaque failure the app can only render as "something went wrong".
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import requests
import runpod

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import process_video, load_model  # noqa: E402

# ---------------------------------------------------------------------------
# Cold-start work. Weights are baked into the image (see Dockerfile) so this is a
# local load, never a download -- a download here would run on every cold container
# and be billed as GPU time.
# ---------------------------------------------------------------------------
WEIGHTS = os.environ.get("WEIGHTS", "yolo26n.pt")
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", "60"))
MAX_BYTES = int(os.environ.get("MAX_BYTES", str(300 * 1024 * 1024)))

print(f"[init] loading {WEIGHTS}")
MODEL = load_model(WEIGHTS)
print(f"[init] model ready, classes: {MODEL.names}")


class JobError(Exception):
    """Carries a machine-readable code the app can branch on."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def probe_duration(path: str) -> float:
    """Seconds of actual video, from the container metadata.

    Size is not a proxy for duration and must not be used as one.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        raise JobError("UNREADABLE_VIDEO",
                       "Could not read the video's duration. It may be truncated or "
                       "in an unsupported container.")


def download(url: str, dest: str) -> None:
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise JobError("TOO_LARGE",
                                       f"Video exceeds {MAX_BYTES // (1024*1024)} MB.")
                    f.write(chunk)
    except JobError:
        raise
    except Exception as e:
        raise JobError("DOWNLOAD_FAILED", f"Could not fetch the uploaded video: {e}")


def upload(url: str, path: str) -> None:
    try:
        with open(path, "rb") as f:
            r = requests.put(url, data=f, timeout=300,
                             headers={"Content-Type": "video/mp4"})
        r.raise_for_status()
    except Exception as e:
        raise JobError("UPLOAD_FAILED", f"Could not store the annotated video: {e}")


def run(job):
    """RunPod entrypoint. `job["input"]` is whatever the control plane POSTed."""
    inp = job.get("input") or {}
    workdir = tempfile.mkdtemp(prefix="soccer_")      # per-job, never shared
    try:
        video_url = inp.get("video_url")
        upload_url = inp.get("upload_url")
        if not video_url or not upload_url:
            raise JobError("BAD_INPUT", "video_url and upload_url are both required.")

        src = os.path.join(workdir, "input.mp4")
        out = os.path.join(workdir, "annotated.mp4")

        runpod.serverless.progress_update(job, "downloading")
        download(video_url, src)

        seconds = probe_duration(src)
        if seconds > MAX_SECONDS:
            raise JobError(
                "TOO_LONG",
                f"Clip is {seconds:.0f}s; the limit is {MAX_SECONDS}s. "
                "Trim it on the device before uploading.")

        def on_progress(stage, frac):
            runpod.serverless.progress_update(job, f"{stage}:{frac:.2f}")

        stats = process_video(
            source=src,
            output_path=out,
            model=MODEL,                       # reused across warm invocations
            keypoints=inp.get("keypoints"),    # None -> no minimap, no metres
            embedder=inp.get("embedder", "histogram"),
            conf=float(inp.get("conf", 0.3)),
            imgsz=int(inp.get("imgsz", 1280)),
            max_frames=int(inp.get("max_frames", 0)),
            verbose=True,
            on_progress=on_progress,
        )

        runpod.serverless.progress_update(job, "uploading")
        upload(upload_url, out)

        stats.pop("output_path", None)         # a server path means nothing to the app
        stats["duration_seconds"] = round(seconds, 1)
        stats["weights"] = os.path.basename(WEIGHTS)   # never leave this implicit
        return {"ok": True, "stats": stats}

    except JobError as e:
        print(f"[job-error] {e.code}: {e.message}")
        return {"ok": False, "code": e.code, "message": e.message}
    except ValueError as e:
        # pipeline.py and team_cluster.py raise these with genuinely useful text --
        # no players found, no readable frames, too few crops to fit two teams.
        print(f"[pipeline-error] {e}")
        return {"ok": False, "code": "PIPELINE", "message": str(e)}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "code": "INTERNAL", "message": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)     # nothing user-supplied survives


runpod.serverless.start({"handler": run})
