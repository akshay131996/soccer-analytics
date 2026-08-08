"""Fine-tune a 4-class soccer detector on the pod.

Pod-side equivalent of finetune_detector.ipynb: reads the Roboflow key from
/root/.roboflow_key (outside the repo, never committed) instead of Colab Secrets,
and leaves artifacts on disk instead of calling files.download().
"""
import json
import os
import pathlib
import sys

KEY = pathlib.Path("/root/.roboflow_key").read_text().strip()

# ---------------------------------------------------------------- dataset
from roboflow import Roboflow

rf = Roboflow(api_key=KEY)
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")

versions = [v.version for v in project.versions()]
print("available versions:", versions)
latest = sorted(int(str(v).split("/")[-1]) for v in versions)[-1]
print("using version:", latest)

dataset = project.version(latest).download("yolov8")
DATA = os.path.join(dataset.location, "data.yaml")
print("data.yaml ->", DATA)

# ---------------------------------------------------------------- verify classes
import yaml

cfg = yaml.safe_load(open(DATA))
names = cfg["names"]
names = {i: n for i, n in enumerate(names)} if isinstance(names, list) else names
print("\nclass index -> name")
for i, n in sorted(names.items()):
    print("  {}: {}".format(i, n))

sys.path.insert(0, "src")
from pipeline import PLAYER_NAMES, BALL_NAMES, REFEREE_NAMES  # noqa: E402

for label, wanted in (("players", PLAYER_NAMES), ("ball", BALL_NAMES), ("referee", REFEREE_NAMES)):
    got = [i for i, n in names.items() if str(n).lower() in wanted]
    print("  {:8s} -> {} {}".format(label, got, [names[i] for i in got]))
    assert got, "no class matched {} -- the pipeline would not find it either".format(label)

for split in ("train", "valid", "test"):
    d = os.path.join(dataset.location, split, "images")
    if os.path.isdir(d):
        print("  {:6s}: {} images".format(split, len(os.listdir(d))))

# ---------------------------------------------------------------- train
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.train(
    data=DATA,
    epochs=50,
    imgsz=1280,        # the ball is ~10 px; 640 loses it before the model sees it
    batch=8,
    patience=15,
    project="runs",
    name="soccer_4class_1280",
    exist_ok=True,
    plots=True,
    verbose=True,
)
best = "{}/weights/best.pt".format(results.save_dir)
print("\nweights ->", best)

# ---------------------------------------------------------------- per-class metrics
m = YOLO(best)
metrics = m.val(data=DATA, imgsz=1280, split="val")

rows = []
print("\n{:<14}{:>9}{:>10}".format("class", "AP50", "AP50-95"))
print("-" * 33)
for i, c in enumerate(metrics.box.ap_class_index):
    name = m.names[int(c)]
    ap50 = float(metrics.box.ap50[i])
    ap = float(metrics.box.ap[i])
    rows.append({"class": name, "AP50": round(ap50, 4), "AP50_95": round(ap, 4)})
    print("{:<14}{:>9.3f}{:>10.3f}".format(name, ap50, ap))
print("-" * 33)
print("{:<14}{:>9.3f}{:>10.3f}".format("MEAN", metrics.box.map50, metrics.box.map))

out = {
    "weights": best,
    "dataset_version": latest,
    "per_class": rows,
    "mAP50": round(float(metrics.box.map50), 4),
    "mAP50_95": round(float(metrics.box.map), 4),
}
pathlib.Path("outputs").mkdir(exist_ok=True)
pathlib.Path("outputs/finetune_results.json").write_text(json.dumps(out, indent=2))
print("\nwrote outputs/finetune_results.json")
