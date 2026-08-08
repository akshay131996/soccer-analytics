import sys
sys.path.insert(0, "src")

ok = True


def chk(label, fn):
    global ok
    try:
        r = fn()
        print("  OK   {}{}".format(label, " -> {}".format(r) if r is not None else ""))
    except Exception as e:
        ok = False
        print("  FAIL {}: {}: {}".format(label, type(e).__name__, e))


import numpy as np

chk("numpy", lambda: np.__version__)
chk("cv2", lambda: __import__("cv2").__version__)
chk("torch cuda", lambda: __import__("torch").cuda.is_available())
chk("ultralytics", lambda: __import__("ultralytics").__version__)
chk("supervision", lambda: __import__("supervision").__version__)
chk("sklearn", lambda: __import__("sklearn").__version__)


def bytetrack():
    import supervision as sv
    return type(sv.ByteTrack(frame_rate=25)).__name__


chk("sv.ByteTrack exists", bytetrack)


def npcross():
    # the exact call numpy 2.4 removed, which is why the pin exists
    return float(np.cross(np.array([1.0, 2.0]), np.array([3.0, 4.0])))


chk("np.cross on 2-D (LineZone needs it)", npcross)

chk("import pipeline", lambda: __import__("pipeline").__name__)
chk("import homography", lambda: __import__("homography").__name__)
chk("import team_cluster", lambda: __import__("team_cluster").__name__)


def hom5():
    from homography import PitchMapper
    img = [[100, 100], [500, 100], [520, 400], [80, 400], [300, 250]]
    pitch = [[0, 0], [105, 0], [105, 68], [0, 68], [52.5, 34]]
    q = PitchMapper(img, pitch).quality()
    return "residual={}m meaningful={} inliers={}/{}".format(
        q["mean_reprojection_error_m"], q["error_is_meaningful"],
        q["inliers"], q["correspondences"])


chk("PitchMapper 5-pt + RANSAC", hom5)


def hom4():
    from homography import PitchMapper
    q = PitchMapper([[100, 100], [500, 100], [520, 400], [80, 400]],
                    [[0, 0], [105, 0], [105, 68], [0, 68]]).quality()
    return "error_is_meaningful={} (must be False), residual={}".format(
        q["error_is_meaningful"], q["mean_reprojection_error_m"])


chk("PitchMapper 4-pt trap flagged", hom4)


def pitch_cache():
    from homography import draw_pitch
    a, b = draw_pitch(), draw_pitch()
    return "shape={} returns_copies={}".format(a.shape, a is not b)


chk("draw_pitch memoised", pitch_cache)


def loadfail():
    from pipeline import load_model
    try:
        load_model("definitely_not_a_real_checkpoint_xyz.pt")
        return "NO RAISE -- silent substitution is back"
    except ValueError as e:
        return "raises ValueError as intended"


chk("load_model fails loudly", loadfail)

print()
print("ALL GOOD" if ok else "SOMETHING FAILED")
