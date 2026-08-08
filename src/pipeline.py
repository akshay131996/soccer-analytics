"""Soccer analytics pipeline: video -> detect -> track -> teams -> pitch -> stats.

    python src/pipeline.py --source match.mp4 --keypoints keypoints.json

Works today with a COCO-pretrained model (people only); pass --weights with a
fine-tuned soccer checkpoint to get ball / goalkeeper / referee classes as well.

The core is `process_video()`, imported by the notebook so both run identical code.
Design rationale for every non-obvious choice is in walkthrough.html.
"""
import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from homography import PitchMapper, draw_minimap
from team_cluster import TeamClassifier

OUT = Path(__file__).parent.parent / "outputs"
TEAM_COLORS = sv.ColorPalette.from_hex(["#ff5050", "#5050ff", "#ffd700"])

# Resolve classes BY NAME, never by hardcoded index. A COCO checkpoint puts `person`
# at 0; a fine-tuned soccer checkpoint typically orders classes alphabetically, which
# puts `ball` at 0. Hardcoding `class_id == 0` therefore tracks the ball as if it were
# a squad of players and still emits a full set of plausible-looking statistics.
# This is the same mistake that invalidated Traffic Lens's baseline -- see LEARNINGS
# in that repo, and section 1 of walkthrough.html here.
PLAYER_NAMES = {"player", "person", "goalkeeper", "goalkeepers"}
BALL_NAMES = {"ball", "sports ball", "football", "soccer ball"}
REFEREE_NAMES = {"referee", "refere"}


def class_ids_for(model, names: set) -> list:
    """Indices in *this* model's label space whose name matches `names`."""
    return [i for i, n in model.names.items() if str(n).lower() in names]


def load_model(name: str) -> YOLO:
    """Load weights, or fail loudly.

    This used to swallow *any* exception and quietly substitute yolo11n.pt. That is
    the worst possible behaviour for a pipeline that reports metrics: every downstream
    number would be attributed to a model that never ran, and the only evidence was a
    print() nobody reads in a batch job. A missing checkpoint is a deployment error,
    not something to paper over.
    """
    try:
        return YOLO(name)
    except Exception as e:
        raise ValueError(
            f"could not load weights '{name}': {e}. Check the path, or pass "
            f"--weights with a checkpoint that exists. Results are attributed to "
            f"specific weights, so no substitute is loaded automatically."
        ) from e


def crops_from(frame, det):
    return [sv.crop_image(frame, xyxy) for xyxy in det.xyxy.astype(int)]


def collect_crops(model, source, player_ids, target=200, max_scan=120,
                  imgsz=1280, conf=0.3, verbose=True):
    """Gather player crops for fitting the team clusters.

    Scans until it has `target` crops rather than reading a fixed number of frames.
    A fixed count is fragile: a clip can easily open on a crowd shot, a replay or an
    empty pitch, and you end up trying to fit k-means on nothing. Scanning until the
    quota is met -- or the budget runs out -- works on any footage.
    """
    crops = []
    i = -1                       # bound before the loop: a video that yields zero
                                 # frames (unreadable codec, truncated upload) would
                                 # otherwise raise UnboundLocalError below, reporting
                                 # a problem that has nothing to do with the real one.
    for i, frame in enumerate(sv.get_video_frames_generator(source)):
        if i >= max_scan or len(crops) >= target:
            break
        det = sv.Detections.from_ultralytics(model(frame, conf=conf, imgsz=imgsz, verbose=False)[0])
        crops += crops_from(frame, det[np.isin(det.class_id, player_ids)])
    if i < 0:
        raise ValueError(
            "the video yielded no frames at all. It is unreadable, truncated, or in a "
            "codec OpenCV was not built for (HEVC/H.265 from a phone camera is the "
            "usual culprit). Re-encode to H.264 before processing."
        )
    if verbose:
        print(f"collected {len(crops)} player crops from {min(i + 1, max_scan)} frames")
    return crops


def process_video(
    source: str,
    output_path: str,
    model=None,
    weights: str = "yolo26n.pt",
    keypoints: dict | None = None,
    embedder: str = "histogram",
    conf: float = 0.3,
    imgsz: int = 1280,
    max_frames: int = 0,
    fit_frames: int = 20,
    verbose: bool = True,
    on_progress=None,
    nms_iou: float = 0.7,
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int | None = None,
    minimum_matching_threshold: float = 0.8,
    team_votes_target: int = 15,
) -> dict:
    """Run the full pipeline over one clip and write an annotated video.

    `keypoints` is {"image_points": [[px,py] x4+], "pitch_points": [[m,m] x4+]} and
    unlocks the minimap plus every metric expressed in metres. A pitch is 105 x 68 m
    by regulation, so unlike most calibration problems the real-world coordinates are
    known exactly rather than estimated.

    `imgsz` defaults to 1280 rather than the usual 640: a football is ~10 px in a
    1080p frame and effectively disappears when the frame is downscaled to 640.
    """
    if model is None:
        model = load_model(weights)

    player_ids = class_ids_for(model, PLAYER_NAMES)
    ball_ids = class_ids_for(model, BALL_NAMES)
    ref_ids = class_ids_for(model, REFEREE_NAMES)
    if not player_ids:
        raise ValueError(f"no player-like class in this model's labels: {model.names}")
    if verbose:
        print(f"players -> {[model.names[i] for i in player_ids]}"
              f"  ball -> {[model.names[i] for i in ball_ids] or 'not in this model'}"
              f"  refs -> {[model.names[i] for i in ref_ids] or 'not in this model'}")

    info = sv.VideoInfo.from_video_path(source)
    fps = info.fps
    # ByteTrack's defaults assume ~25-30 fps and a broadly static camera. On 60 fps
    # footage with a camera that pans with play, a 30-frame buffer is half a second,
    # so a player briefly occluded or briefly off-frame comes back as a new identity.
    # Measured on FC26: 308 tracks for ~20 players over 25 s. Scale the buffer with
    # frame rate and hold tracks for ~2 seconds instead.
    tracker = sv.ByteTrack(
        frame_rate=fps,
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer if lost_track_buffer is not None else int(fps * 2),
        minimum_matching_threshold=minimum_matching_threshold,
    )

    # 15 m/s sits comfortably above a world-class sprint (~12.4 m/s) while still
    # rejecting the metre-scale jumps an ID switch produces. Derived from fps so the
    # threshold means the same thing on 25, 30 and 60 fps footage.
    MAX_PLAYER_SPEED_MS = 15.0
    max_step_m = MAX_PLAYER_SPEED_MS / fps

    ellipse_ann = sv.EllipseAnnotator(color=TEAM_COLORS, thickness=2)
    label_ann = sv.LabelAnnotator(color=TEAM_COLORS, text_scale=0.5,
                                  text_position=sv.Position.BOTTOM_CENTER)

    mapper = PitchMapper(keypoints["image_points"], keypoints["pitch_points"]) if keypoints else None

    # ---- pass 1: fit the team clusters -------------------------------------------
    # k-means cannot predict before it has been fitted, and the two kit colours are
    # not known until players have been observed -- hence reading the clip twice.
    if on_progress:
        on_progress("fitting", 0.0)
    teams = TeamClassifier(embedder)
    # `fit_frames` is the number of frames worth scanning per ~6 expected players.
    # It used to be floored at 120, which silently made every value <= 20 -- including
    # the default -- behave identically, so the parameter did nothing at all.
    fit_crops = collect_crops(model, source, player_ids, target=200,
                              max_scan=max(fit_frames, 1) * 6,
                              imgsz=imgsz, conf=conf, verbose=verbose)
    teams.fit(fit_crops)          # raises a descriptive error if it found too few
    if verbose:
        print(f"team clusters fitted on {len(fit_crops)} crops")

    # ---- pass 2: track, assign, map, render --------------------------------------
    team_of = {}                       # tracker_id -> current majority team
    team_votes = defaultdict(lambda: defaultdict(int))   # tracker_id -> {team: votes}
    distance_m = defaultdict(float)
    last_pos_m, speed_win = {}, defaultdict(lambda: deque(maxlen=int(fps)))
    top_speed_ms = defaultdict(float)
    possession = defaultdict(int)
    poss_hist = deque(maxlen=int(fps // 2) or 1)
    seen_frames = defaultdict(int)
    n_frames = 0
    # total_frames can be None or 0 for containers without a reliable index, so every
    # use of this is guarded rather than assumed.
    total_target = max_frames or (info.total_frames or 0)

    with sv.VideoSink(output_path, video_info=info) as sink:
        for i, frame in enumerate(sv.get_video_frames_generator(source)):
            if max_frames and i >= max_frames:
                break
            n_frames += 1

            result = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            all_det = sv.Detections.from_ultralytics(result)

            det = all_det[np.isin(all_det.class_id, player_ids)]
            # YOLO26 is NMS-free and should not emit overlapping boxes for one object,
            # but measured on FC26 footage 8.6% of detection pairs overlapped at
            # IoU > 0.6. Each spurious box becomes its own track, which is a direct
            # contributor to the ID count.
            if nms_iou and len(det) > 1:
                det = det.with_nms(threshold=nms_iou)
            det = tracker.update_with_detections(det)

            ball_xy = None
            if ball_ids:
                bd = all_det[np.isin(all_det.class_id, ball_ids)]
                if len(bd) > 0:
                    best = int(np.argmax(bd.confidence))
                    ball_xy = bd.get_anchors_coordinates(sv.Position.CENTER)[best]

            # Team is a property of the player, not of the frame: classify each
            # tracker_id once and reuse. Cheap with the histogram embedder, essential
            # with SigLIP (one transformer pass per player per frame otherwise), and
            # it also stops the assignment flickering between frames.
            # Team is a property of the player, not of the frame -- but deciding it
            # from ONE crop at track birth is a single roll of a weak classifier.
            # Measured on FC26: the k-means silhouette is 0.386, so a large minority
            # of crops sit near the boundary, and a player occluded or motion-blurred
            # at the moment their track started got a coin flip that then stuck.
            # Vote instead: keep classifying until a track has `team_votes_target`
            # observations, then freeze the majority. Costs a few extra histograms
            # per track, which is nothing next to the detector.
            voting = [j for j, tid in enumerate(det.tracker_id)
                      if sum(team_votes[tid].values()) < team_votes_target]
            if voting:
                crops = crops_from(frame, det[np.array(voting)])
                for j, t in zip(voting, teams.predict(crops)):
                    team_votes[det.tracker_id[j]][int(t)] += 1

            for tid in det.tracker_id:
                v = team_votes[tid]
                team_of[tid] = max(v, key=v.get) if v else 0
            team_ids = np.array([team_of.get(t, 0) for t in det.tracker_id], dtype=int)

            for tid in det.tracker_id:
                seen_frames[tid] += 1

            labels = [f"#{tid}" for tid in det.tracker_id]
            annotated = ellipse_ann.annotate(frame.copy(), det, custom_color_lookup=team_ids)
            annotated = label_ann.annotate(annotated, det, labels=labels,
                                           custom_color_lookup=team_ids)
            if ball_xy is not None:
                cv2.circle(annotated, (int(ball_xy[0]), int(ball_xy[1])), 8, (255, 255, 255), -1)
                cv2.circle(annotated, (int(ball_xy[0]), int(ball_xy[1])), 8, (0, 0, 0), 2)

            if mapper is not None and len(det) > 0:
                feet = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_pos = mapper.to_pitch(feet)

                for tid, pos in zip(det.tracker_id, pitch_pos):
                    if tid in last_pos_m:
                        step = float(np.linalg.norm(pos - last_pos_m[tid]))
                        # A teleport guard has to be a SPEED, not a per-frame distance.
                        # The old fixed 2.0 m/frame meant 180 km/h at 25 fps but 432
                        # km/h at 60 fps -- so it filtered almost nothing on
                        # high-frame-rate phone footage and over-filtered downsampled
                        # clips. Discarding these still makes distance a LOWER BOUND.
                        if step < max_step_m:
                            distance_m[tid] += step
                            speed_win[tid].append(step)
                            # speed_win used to be written every frame and never read.
                            # A full window is exactly one second of motion, so its
                            # mean step x fps is metres per second.
                            w = speed_win[tid]
                            if len(w) == w.maxlen:
                                v = sum(w) / len(w) * fps
                                if v > top_speed_ms[tid]:
                                    top_speed_ms[tid] = v
                    last_pos_m[tid] = pos

                # possession = team of the player nearest the ball, with hysteresis
                # so contested balls don't make the stat flicker every frame
                if ball_xy is not None:
                    ball_m = mapper.to_pitch(np.array([ball_xy]))[0]
                    d = np.linalg.norm(pitch_pos - ball_m, axis=1)
                    poss_hist.append(int(team_ids[int(np.argmin(d))]))
                    if len(poss_hist) == poss_hist.maxlen:
                        vals = list(poss_hist)
                        winner = max(set(vals), key=vals.count)
                        if vals.count(winner) > len(vals) * 0.6:
                            possession[winner] += 1

                mini = draw_minimap(pitch_pos[team_ids == 0], pitch_pos[team_ids == 1],
                                    mapper.to_pitch(np.array([ball_xy]))[0] if ball_xy is not None else None)
                # The minimap keeps a 105:68 aspect, so its width is ~1.54x its height.
                # Sizing it purely off frame height overflowed the frame width on any
                # source narrower than ~0.39x its height -- portrait phone video sits
                # close to that line. Fit to BOTH dimensions.
                mh = info.height // 4
                mw = int(mini.shape[1] * mh / mini.shape[0])
                if mw > info.width:
                    mw = info.width
                    mh = int(mini.shape[0] * mw / mini.shape[1])
                mini = cv2.resize(mini, (mw, mh))
                annotated[-mh:, :mw] = mini

            sink.write_frame(annotated)
            if on_progress and i % 10 == 0 and total_target:
                on_progress("processing", min(1.0, (i + 1) / total_target))
            if verbose and i % 25 == 0:
                print(f"  frame {i}: {len(det)} players"
                      + ("" if ball_xy is None else ", ball found"))

    # ---- summarise ----------------------------------------------------------------
    top = sorted(distance_m.items(), key=lambda kv: -kv[1])[:5]
    total_poss = sum(possession.values())
    stats = {
        "frames": n_frames,
        "players_tracked": len(seen_frames),
        "mean_track_length_frames": round(float(np.mean(list(seen_frames.values()))), 1) if seen_frames else 0,
        "has_pitch_mapping": mapper is not None,
        # Carry the calibration's own error estimate next to every metric derived from
        # it. A distance in metres is only as good as the homography that produced it,
        # and that quality was previously invisible.
        "calibration": mapper.quality() if mapper is not None else None,
        "ball_class_available": bool(ball_ids),
        "weights": getattr(getattr(model, "ckpt_path", None), "name", None) or str(weights),
        "classes_detected": {
            "players": [model.names[i] for i in player_ids],
            "ball": [model.names[i] for i in ball_ids],
            "referee": [model.names[i] for i in ref_ids],
        },
        "top_distance_m": [{"id": int(t), "metres": round(d, 1)} for t, d in top],
        "top_speed_kmh": [
            {"id": int(t), "kmh": round(v * 3.6, 1)}
            for t, v in sorted(top_speed_ms.items(), key=lambda kv: -kv[1])[:5]
        ] or None,
        "possession_pct": ({str(k): round(100 * v / total_poss, 1) for k, v in possession.items()}
                           if total_poss else None),
        "output_path": output_path,
        "imgsz": imgsz,
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--embedder", default="histogram", choices=["histogram", "siglip"])
    ap.add_argument("--keypoints", default=None, help="JSON of pitch correspondences -> minimap + metres")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fit-frames", type=int, default=20)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    kp = json.loads(Path(args.keypoints).read_text()) if args.keypoints else None
    stats = process_video(
        source=args.source, output_path=str(OUT / "flagship_annotated.mp4"),
        weights=args.weights, keypoints=kp, embedder=args.embedder,
        conf=args.conf, imgsz=args.imgsz, max_frames=args.max_frames,
        fit_frames=args.fit_frames,
    )
    print("\n" + json.dumps(stats, indent=2))
    if not stats["has_pitch_mapping"]:
        print("\nno --keypoints given -> no minimap, distances or possession. See README.")


if __name__ == "__main__":
    main()
