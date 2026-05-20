# MedScan AI — Inference Service

Stateless chest X-ray classifier backed by **TorchXRayVision** (DenseNet-121, pretrained on the union of 7 large public chest X-ray datasets — NIH ChestX-ray14, CheXpert, MIMIC-CXR, PadChest, OpenI, RSNA, Google). Returns calibrated probabilities for **18 thoracic findings** in a single forward pass.

---

## What it predicts

| Group | Labels |
|-------|--------|
| Infectious / inflammatory | Pneumonia, Consolidation, Infiltration |
| Mechanical / structural | Atelectasis, Pneumothorax, Effusion, Pleural Thickening, Fracture, Hernia |
| Cardiac | Cardiomegaly, Enlarged Cardiomediastinum |
| Lesional | Nodule, Mass, Lung Lesion, Lung Opacity |
| Diffuse | Edema, Fibrosis, Emphysema |

Predictions are **multi-label** (sigmoid, not softmax) — a single X-ray can flag multiple findings simultaneously.

---

## API

### `POST /`

**Body (multipart/form-data):**

| Field | Type | Description |
|-------|------|-------------|
| `image_data` | file | X-ray image (PNG / JPG / DICOM-converted) |

**Response (200):**

```json
{
  "Predicted class": "Pneumonia",
  "confidence": 87.32,
  "model": "densenet121-res224-all",
  "model_version": "TorchXRayVision-1.2.4",
  "device": "cpu",
  "tta": true,
  "threshold": 0.5,
  "pathologies": {
    "Atelectasis": 0.1142,
    "Cardiomegaly": 0.0531,
    "Pneumonia": 0.8732,
    "Lung Opacity": 0.6121,
    "...": "...",
    "Hernia": 0.0029
  },
  "top_findings": [
    {"label": "Pneumonia", "probability": 0.8732},
    {"label": "Lung Opacity", "probability": 0.6121}
  ],
  "inference_time_ms": 240.5
}
```

> **Backward compatibility:** the legacy `Predicted class` + `confidence` fields are preserved, so the existing FastAPI worker keeps working without any changes. The enriched fields are written to the prediction's JSON result for future use by the UI.

### `GET /health`

```json
{ "status": "ok", "model": "densenet121-res224-all", "device": "cpu" }
```

---

## Why this is better than the previous model

| | Previous (EfficientNet .h5) | Now (TorchXRayVision) |
|---|---|---|
| **Classes** | 2 (Normal / Pneumonia) | 18 thoracic findings |
| **Training data** | Single dataset | Union of 7 datasets, ~700k images |
| **Preprocessing** | Broken (BGR fed as RGB, no normalization) | Correct grayscale + center-crop + xrv normalize |
| **Output type** | Argmax (forced single class) | Calibrated per-pathology probabilities |
| **Model loading** | On every request (3-5s overhead) | Once at startup, cached in memory |
| **TTA** | No | Yes (horizontal flip averaging) |
| **Server** | Flask dev server | gunicorn (4 threads) |
| **Image size** | ~2 GB (TensorFlow base) | ~1.2 GB (Python slim + torch CPU) |

---

## Local development

```bash
# Run as part of the full stack (from medical-diagnosis-api/):
docker compose up -d ai-model

# Or standalone:
docker compose up --build -d
```

Service listens on `http://localhost:5000/`.

---

## Notes

- **First build downloads ~150 MB** (PyTorch CPU + torchxrayvision + model weights). Weights are baked into the image so first request is instant.
- **Memory footprint**: ~600 MB at runtime (PyTorch + model).
- **No GPU required** — DenseNet-121 inference on CPU takes ~200-400 ms per image.
- To enable GPU, swap the `torch==2.2.2` CPU wheel in `requirements.txt` for the CUDA variant and use an `nvidia/cuda` base image instead of `python:3.10-slim`.
