"""Team assignment: cluster player crops into two teams, no labels needed.

The trick: kit colors dominate player crops, so embeddings of crops separate cleanly
into two clusters. Default embedder is an HSV histogram (fast, CPU, surprisingly good).
Upgrade path is SigLIP embeddings (--embedder siglip) — better under weird lighting,
handles keepers/refs as outliers more gracefully.
"""
import numpy as np


# Pitch green in OpenCV's HSV (hue 0-180). Measured on FC26 footage: even after
# cropping to the torso strip, grass was a mean 31% of the pixels and over 30% in
# 42% of crops -- so the histogram was partly describing the pitch, not the kit.
# Masking it lifted the k-means silhouette from 0.386 to 0.475.
GRASS_HUE_LO, GRASS_HUE_HI, GRASS_SAT_MIN = 30, 90, 60


class HistogramEmbedder:
    """HSV colour histogram of the torso region, with pitch green masked out.

    The torso strip (upper-middle of the crop) is mostly shirt, but for distant
    players the box is only ~30x65 px and the gaps around the body are all grass.
    Excluding those pixels is what makes the two kits separable.
    """

    def __call__(self, crops):
        import cv2
        feats = []
        for crop in crops:
            h, w = crop.shape[:2]
            torso = crop[int(0.1 * h): int(0.55 * h), int(0.2 * w): int(0.8 * w)]
            if torso.size == 0:
                torso = crop
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            grass = ((hsv[:, :, 0] > GRASS_HUE_LO) & (hsv[:, :, 0] < GRASS_HUE_HI)
                     & (hsv[:, :, 1] > GRASS_SAT_MIN))
            mask = (~grass).astype(np.uint8) * 255
            # If a crop is essentially all grass there is nothing to describe; fall
            # back to the unmasked histogram rather than emitting an all-zero vector,
            # which would cluster with every other degenerate crop.
            if mask.sum() < 255 * 10:
                mask = None
            hist = cv2.calcHist([hsv], [0, 1], mask, [18, 6], [0, 180, 0, 256])
            feats.append(cv2.normalize(hist, None).flatten())
        return np.array(feats)


class SiglipEmbedder:
    """Semantic embeddings from SigLIP's vision tower. Heavier but more robust."""

    def __init__(self, model_id="google/siglip-base-patch16-224"):
        import torch
        from transformers import AutoProcessor, SiglipVisionModel
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = SiglipVisionModel.from_pretrained(model_id).eval()

    def __call__(self, crops):
        from PIL import Image
        images = [Image.fromarray(c[:, :, ::-1]) for c in crops]  # BGR -> RGB
        inputs = self.processor(images=images, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(**inputs).pooler_output
        return out.numpy()


class TeamClassifier:
    """fit() on a few hundred crops from early frames, then predict() forever after."""

    def __init__(self, embedder="histogram"):
        self.embedder = SiglipEmbedder() if embedder == "siglip" else HistogramEmbedder()
        self.kmeans = None

    def fit(self, crops):
        from sklearn.cluster import KMeans
        # Fail with a message that says what to do. Handing an empty list to sklearn
        # produces "Expected 2D array, got 1D array instead: array=[]", which tells you
        # nothing about the actual problem -- that the detector found no players.
        if len(crops) < 2:
            raise ValueError(
                f"need at least 2 player crops to fit two teams, got {len(crops)}. "
                "The detector found (almost) nothing: check that the clip actually "
                "contains people, that class resolution matched (print model.names), "
                "and that imgsz is high enough for the object size in your footage."
            )
        feats = self.embedder(crops)
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feats)
        return self

    def predict(self, crops):
        if not crops:
            return np.array([], dtype=int)
        return self.kmeans.predict(self.embedder(crops))
