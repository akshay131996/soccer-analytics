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
    try:
        return YOLO(name)
    except Exception as e:
        print(f"could not load {name} ({e}); falling back to yolo11n.pt")
        return YOLO("yolo11n.pt")


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
    for i, frame in enumerate(sv.get_video_frames_generator(source)):
        if i >= max_scan or len(crops) >= target:
            break
        det = sv.Detections.from_ultralytics(model(frame, conf=conf, imgsz=imgsz, verbose=False)[0])
        crops += crops_from(frame, det[np.isin(det.class_id, player_ids)])
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
    tracker = sv.ByteTrack(frame_rate=fps)

    ellipse_ann = sv.EllipseAnnotator(color=TEAM_COLORS, thickness=2)
    label_ann = sv.LabelAnnotator(color=TEAM_COLORS, text_scale=0.5,
                                  text_position=sv.Position.BOTTOM_CENTER)

    mapper = PitchMapper(keypoints["image_points"], keypoints["pitch_points"]) if keypoints else None

    # ---- pass 1: fit the team clusters -------------------------------------------
    # k-means cannot predict before it has been fitted, and the two kit colours are
    # not known until players have been observed -- hence reading the clip twice.
    teams = TeamClassifier(embedder)
    fit_crops = collect_crops(model, source, player_ids, target=200,
                              max_scan=max(fit_frames * 6, 120),
                              imgsz=imgsz, conf=conf, verbose=verbose)
    teams.fit(fit_crops)          # raises a descriptive error if it found too few
    if verbose:
        print(f"team clusters fitted on {len(fit_crops)} crops")

    # ---- pass 2: track, assign, map, render --------------------------------------
    team_of = {}                       # tracker_id -> team, decided once and cached
    distance_m = defaultdict(float)
    last_pos_m, speed_win = {}, defaultdict(lambda: deque(maxlen=int(fps)))
    possession = defaultdict(int)
    poss_hist = deque(maxlen=int(fps // 2) or 1)
    seen_frames = defaultdict(int)
    n_frames = 0

    with sv.VideoSink(output_path, video_info=info) as sink:
        for i, frame in enumerate(sv.get_video_frames_generator(source)):
            if max_frames and i >= max_frames:
                break
            n_frames += 1

            result = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            all_det = sv.Detections.from_ultralytics(result)

            det = all_det[np.isin(all_det.class_id, player_ids)]
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
            new_idx = [j for j, tid in enumerate(det.tracker_id) if tid not in team_of]
            if new_idx:
                crops = crops_from(frame, det[np.array(new_idx)])
                for j, t in zip(new_idx, teams.predict(crops)):
                    team_of[det.tracker_id[j]] = int(t)
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
                        # >2 m between frames at 25fps is 180 km/h -- physically
                        # impossible, so it's an ID switch teleport, not motion.
                        # Discarding it makes distance a LOWER BOUND, not a measurement.
                        if step < 2.0:
                            distance_m[tid] += step
                            speed_win[tid].append(step)
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
                mh = info.height // 4
                mini = cv2.resize(mini, (int(mini.shape[1] * mh / mini.shape[0]), mh))
                annotated[-mh:, :mini.shape[1]] = mini

            sink.write_frame(annotated)
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
        "ball_class_available": bool(ball_ids),
        "top_distance_m": [{"id": int(t), "metres": round(d, 1)} for t, d in top],
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
