"""
MedScan AI — Chest X-Ray Inference Service

Backed by TorchXRayVision (densenet121-res224-all), which is a DenseNet-121
pretrained on the union of 7 large public chest X-ray datasets
(NIH ChestX-ray14, CheXpert, MIMIC-CXR, PadChest, OpenI, RSNA, Google).

The model produces calibrated probabilities for 18 thoracic findings.
Inference is multi-label (sigmoid), not single-class softmax.
"""
from __future__ import annotations

import io
import logging
import os
import time

import numpy as np
import torch
import torchxrayvision as xrv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from skimage.transform import resize

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("medscan-ai")

# ── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

STATIC_FOLDER = "static"
app.config["STATIC_FOLDER"] = STATIC_FOLDER
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ── Model bootstrap (load ONCE at startup, not per request) ───────────────────
MODEL_NAME = "densenet121-res224-all"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

log.info("Loading model %s on %s ...", MODEL_NAME, DEVICE)
_t0 = time.time()
MODEL = xrv.models.DenseNet(weights=MODEL_NAME)
MODEL.eval()
MODEL.to(DEVICE)
log.info("Model loaded in %.2fs", time.time() - _t0)

PATHOLOGIES: list[str] = MODEL.pathologies          # 18 labels (some may be "")
ACTIVE_LABELS = [p for p in PATHOLOGIES if p]       # drop empty placeholders
log.info("Active pathologies (%d): %s", len(ACTIVE_LABELS), ACTIVE_LABELS)

# Confidence threshold above which a finding is considered "present"
POSITIVE_THRESHOLD = 0.5

# Test-Time Augmentation toggle (averages original + horizontal flip)
USE_TTA = True


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    """Convert raw bytes → (1, 1, 224, 224) tensor-ready numpy array.

    TorchXRayVision expects:
      - grayscale (1 channel)
      - 224 × 224
      - normalized to roughly [-1024, 1024] (HU-like range)
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Strip alpha + handle RGBA / palette → grayscale
    if img.mode != "L":
        img = img.convert("L")

    arr = np.asarray(img, dtype=np.float32)

    # Normalize to TorchXRayVision's expected [-1024, 1024] range
    arr = xrv.datasets.normalize(arr, 255)

    # Channels first: (1, H, W)
    arr = arr[None, :, :]

    # Center-crop to a square (preserve aspect ratio)
    arr = xrv.datasets.XRayCenterCrop()(arr)

    # Resize to 224×224
    arr = xrv.datasets.XRayResizer(224)(arr)

    # Add batch dim → (1, 1, 224, 224)
    return arr[None, :, :, :]


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(image_bytes: bytes) -> dict:
    """Return per-pathology probabilities + top finding for one image."""
    t0 = time.time()

    arr = preprocess(image_bytes)
    tensor = torch.from_numpy(arr).float().to(DEVICE)

    # Forward pass
    logits = MODEL(tensor)
    probs = logits.cpu().numpy()[0]  # shape (N,)

    # Test-Time Augmentation: predict on horizontally-flipped image, average
    if USE_TTA:
        flipped = torch.flip(tensor, dims=[-1])
        probs_flipped = MODEL(flipped).cpu().numpy()[0]
        probs = (probs + probs_flipped) / 2.0

    # Build {label: prob} dict, skipping empty placeholder labels
    by_label = {
        label: float(prob)
        for label, prob in zip(PATHOLOGIES, probs)
        if label  # filter empty strings
    }

    # Sort findings descending; mark anything ≥ threshold as "positive"
    sorted_findings = sorted(by_label.items(), key=lambda kv: kv[1], reverse=True)
    top_findings = [
        {"label": label, "probability": round(prob, 4)}
        for label, prob in sorted_findings
        if prob >= POSITIVE_THRESHOLD
    ]

    # Pick the headline prediction
    top_label, top_prob = sorted_findings[0]
    if top_prob < POSITIVE_THRESHOLD:
        # Nothing crossed the threshold → call it "No Finding"
        predicted_class = "No Finding"
        confidence = float((1.0 - top_prob) * 100)
    else:
        predicted_class = top_label
        confidence = float(top_prob * 100)

    inference_ms = round((time.time() - t0) * 1000, 1)

    return {
        # ── Backwards-compatible fields (consumed by FastAPI worker) ──
        "Predicted class": predicted_class,
        "confidence": round(confidence, 2),
        # ── Enriched fields (saved to DB, available to UI) ──
        "model": MODEL_NAME,
        "model_version": "TorchXRayVision-1.2.4",
        "device": DEVICE,
        "tta": USE_TTA,
        "threshold": POSITIVE_THRESHOLD,
        "pathologies": {k: round(v, 4) for k, v in by_label.items()},
        "top_findings": top_findings,
        "inference_time_ms": inference_ms,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def welcome():
    if os.path.exists(os.path.join(STATIC_FOLDER, "index.html")):
        return send_from_directory(app.config["STATIC_FOLDER"], "index.html")
    return jsonify({
        "service": "medscan-ai",
        "model": MODEL_NAME,
        "device": DEVICE,
        "labels": ACTIVE_LABELS,
        "status": "ready",
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": DEVICE})


@app.route("/", methods=["POST"])
def predict():
    if "image_data" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image_data"]
    image_bytes = image_file.read()

    if not image_bytes:
        return jsonify({"error": "Empty image file"}), 400

    try:
        result = run_inference(image_bytes)
    except Exception as exc:
        log.exception("Inference failed")
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    log.info(
        "Prediction: %s (%.2f%%) in %sms",
        result["Predicted class"],
        result["confidence"],
        result["inference_time_ms"],
    )
    return jsonify(result)


@app.route("/<path:filename>")
def serve_static(filename: str):
    return send_from_directory(app.config["STATIC_FOLDER"], filename)


if __name__ == "__main__":
    # Dev only — production uses gunicorn (see Dockerfile)
    app.run(host="0.0.0.0", port=5000)
