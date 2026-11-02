"""
backend/main.py
────────────────
FastAPI server — the hospital-side dashboard for TrustEdge-CXR.

Uses the TrustEdge student when artifacts/student/ holds student.keras,
head.npz and calibration.json; otherwise falls back to the Review 1 model
so the demo always runs.

Endpoints:
  GET  /api/info           which model is loaded, threshold and τ
  POST /api/predict        upload an X-ray → NORMAL / PNEUMONIA / REFER,
                           calibrated probability, uncertainty, heatmap
  GET  /api/metrics        test-set comparison (artifacts/results) or the
                           Review 1 validation metrics
  GET  /api/assets/{name}  result figures
  POST /api/edge/report    results pushed by edge devices (edge/edge_infer.py)
  GET  /api/edge/reports   latest edge results for the dashboard
  /                        static frontend (frontend/)

Run:
    source venv/bin/activate
    uvicorn backend.main:app --reload --port 8000
"""

import base64
import io
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from edge.trustedge_runtime import TrustEdgeHead, overlay_heatmap, preprocess_xray  # noqa: E402

MODELS_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"
ARTIFACTS = Path(os.environ.get("TRUSTEDGE_ARTIFACTS", ROOT / "artifacts"))
STUDENT_DIR = ARTIFACTS / "student"
RESULTS_DIR = ARTIFACTS / "results"
EDGE_TOKEN = os.environ.get("TRUSTEDGE_EDGE_TOKEN")

IMG_SIZE = (224, 224)
LEGACY_MODEL_PATH = MODELS_DIR / "pneumonia_mobilenetv2_final.keras"
LEGACY_BASE_LAYER = "mobilenetv2_1.00_224"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

app = FastAPI(title="TrustEdge-CXR API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def img_to_data_url(arr_or_img) -> str:
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ── TrustEdge student ─────────────────────────────────────────────────────────
class TrustEdgePredictor:
    name = "TrustEdge-CXR student (MobileNetV2 + KD, calibrated)"
    explainer = "Grad-CAM++"

    def __init__(self):
        from trustedge.model import feature_model, load_student
        self.features = feature_model(load_student(STUDENT_DIR / "student.keras"))
        self.head = TrustEdgeHead.load(STUDENT_DIR / "head.npz", STUDENT_DIR / "calibration.json")

    def info(self):
        h = self.head
        return {"mode": "trustedge", "model": self.name, "explainer": self.explainer,
                "threshold": h.threshold, "tau": h.tau, "temperature": h.temperature,
                "mc_passes": h.mc_passes}

    def analyze(self, pil_img: Image.Image) -> dict:
        img = preprocess_xray(pil_img)
        fmap = self.features(img[None].astype(np.float32), training=False).numpy()
        pred = self.head.predict(fmap)
        prob, unc = float(pred["prob"][0]), float(pred["uncertainty"][0])
        label = CLASS_NAMES[int(pred["label"][0])]
        return {
            **self.info(),
            "label": label,
            "decision": str(pred["decision"][0]),
            "prob_pneumonia": prob,
            "confidence": prob if label == "PNEUMONIA" else 1 - prob,
            "uncertainty": unc,
            "original": img_to_data_url(pil_img.convert("RGB").resize(IMG_SIZE)),
            "heatmap_overlay": img_to_data_url(overlay_heatmap(img, self.head.cam(fmap)[0])),
        }


# ── Review 1 model (fallback) ─────────────────────────────────────────────────
class LegacyPredictor:
    name = "Review 1 MobileNetV2 (uncalibrated, threshold 0.5)"
    explainer = "Grad-CAM"

    def __init__(self):
        if not LEGACY_MODEL_PATH.exists():
            raise RuntimeError(f"No model found. Train TrustEdge-CXR (artifacts/student/) "
                               f"or put the Review 1 model at {LEGACY_MODEL_PATH}.")
        self.model = tf.keras.models.load_model(LEGACY_MODEL_PATH, compile=False)

    def info(self):
        return {"mode": "legacy", "model": self.name, "explainer": self.explainer,
                "threshold": 0.5, "tau": None}

    def gradcam(self, img_array: np.ndarray) -> np.ndarray:
        """
        Replays the forward pass eagerly because Keras 3 won't let you reach
        into a nested sub-model's output and reconnect it to the outer inputs.
        """
        m = self.model
        with tf.GradientTape() as tape:
            x = tf.keras.applications.mobilenet_v2.preprocess_input(tf.convert_to_tensor(img_array))
            conv = m.get_layer(LEGACY_BASE_LAYER)(x, training=False)
            tape.watch(conv)
            y = m.get_layer("gap")(conv)
            y = m.get_layer("dropout")(y, training=False)
            y = m.get_layer("dense")(y)
            y = m.get_layer("dropout_1")(y, training=False)
            score = m.get_layer("output")(y)[:, 0]
        grads = tape.gradient(score, conv)
        heatmap = tf.squeeze(conv[0] @ tf.reduce_mean(grads, axis=(0, 1, 2))[..., tf.newaxis])
        return (tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)).numpy()

    def analyze(self, pil_img: Image.Image) -> dict:
        rgb = np.asarray(pil_img.convert("RGB").resize(IMG_SIZE), np.float32)[None]
        prob = float(self.model.predict(rgb, verbose=0)[0][0])
        label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
        return {
            **self.info(),
            "label": label,
            "decision": label,
            "prob_pneumonia": prob,
            "confidence": prob if prob > 0.5 else 1 - prob,
            "uncertainty": None,
            "original": img_to_data_url(rgb[0].astype(np.uint8)),
            "heatmap_overlay": img_to_data_url(overlay_heatmap(rgb[0].astype(np.uint8), self.gradcam(rgb))),
        }


_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        ready = all((STUDENT_DIR / f).exists() for f in ("student.keras", "head.npz", "calibration.json"))
        _predictor = TrustEdgePredictor() if ready else LegacyPredictor()
    return _predictor


@app.get("/api/info")
def info():
    try:
        return get_predictor().info()
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "File must be an image (JPG/PNG).")
    raw = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw))
        pil_img.load()
    except Exception:
        raise HTTPException(400, "Could not read image file.")
    try:
        return get_predictor().analyze(pil_img)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


# ── Metrics ───────────────────────────────────────────────────────────────────
def _row(name, m, note=""):
    keys = ("threshold", "accuracy", "sensitivity", "specificity", "f1", "auc", "ece")
    return {"name": name, **{k: m.get(k) for k in keys}, "note": note}


def trustedge_metrics(report: dict) -> dict:
    rows = []
    if "baseline" in report:
        rows.append(_row("Review 1 baseline", report["baseline"]["@0.5"], "image-level split, threshold 0.5"))
    if "teacher" in report:
        rows.append(_row("RAD-DINO linear probe", report["teacher"]["linear_probe"]["test@tuned"], "86.6 M params"))
    if "student_nokd" in report:
        rows.append(_row("Student without KD", report["student_nokd"]["calibrated@tuned"], "ablation"))
    te = report["trustedge"]
    rows.append(_row("TrustEdge student", te["calibrated@tuned"], "KD + calibration, all cases"))
    if "edge_int8" in report:
        e = report["edge_int8"]
        rows.append(_row("TrustEdge INT8 (edge)", e["int8"]["test@tuned"], f"{e['int8_mb']:.1f} MB"))
    for key in sorted(k for k in report if k.startswith("federated")):
        dp = report[key].get("dp")
        rows.append(_row(f"Federated ({report[key]['config']['clients']} sites)", report[key]["test@tuned"],
                         "with DP" if dp else "no DP"))

    extras = {"referral": te["referral"]}
    for name in ("leakage.json", "explanations.json"):
        path = RESULTS_DIR / name
        if path.exists():
            extras[name.removesuffix(".json")] = json.loads(path.read_text())
    figures = [f for f in ("confusion_matrices.png", "roc_curves.png", "reliability.png", "heatmap_examples.png")
               if (RESULTS_DIR / f).exists()]
    return {"mode": "trustedge", "rows": rows, "figures": figures, **extras}


@app.get("/api/metrics")
def metrics():
    report_path = RESULTS_DIR / "test_metrics.json"
    if report_path.exists():
        return trustedge_metrics(json.loads(report_path.read_text()))

    h1_path = MODELS_DIR / "history_phase1.csv"
    h2_path = MODELS_DIR / "history_phase2.csv"
    if not (h1_path.exists() and h2_path.exists()):
        raise HTTPException(404, "No metrics found (run trustedge.evaluate or add the Review 1 histories).")
    h1, h2 = pd.read_csv(h1_path), pd.read_csv(h2_path)
    final = h2.iloc[-1]
    return {
        "mode": "legacy",
        "val_accuracy": float(final["val_accuracy"]),
        "val_auc": float(final["val_auc"]),
        "val_precision": float(final["val_precision"]),
        "val_recall": float(final["val_recall"]),
        "phase1_epochs": int(len(h1)),
        "phase2_epochs": int(len(h2)),
    }


ASSETS = {
    "confusion_matrix.png": MODELS_DIR, "training_curves.png": MODELS_DIR,
    "confusion_matrices.png": RESULTS_DIR, "roc_curves.png": RESULTS_DIR,
    "reliability.png": RESULTS_DIR, "heatmap_examples.png": RESULTS_DIR,
}


@app.get("/api/assets/{name}")
def asset(name: str):
    if name not in ASSETS or not (ASSETS[name] / name).exists():
        raise HTTPException(404, "Not found.")
    return FileResponse(ASSETS[name] / name)


# ── Edge devices (IoT) ────────────────────────────────────────────────────────
class EdgeReport(BaseModel):
    device_id: str = "edge"
    file: str
    decision: str
    prob_pneumonia: float
    uncertainty: float | None = None
    threshold: float | None = None
    tau: float | None = None
    latency_ms: float | None = None
    timestamp: float | None = None
    overlay_png: str | None = None   # base64 PNG, only when the device sends it


_edge_reports: deque = deque(maxlen=200)
EDGE_LOG = ARTIFACTS / "edge_reports.jsonl"


@app.post("/api/edge/report")
def edge_report(report: EdgeReport, x_edge_token: str | None = Header(default=None)):
    if EDGE_TOKEN and x_edge_token != EDGE_TOKEN:
        raise HTTPException(401, "Invalid edge token.")
    entry = report.model_dump()
    entry["received_at"] = time.time()
    _edge_reports.appendleft(entry)
    EDGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EDGE_LOG.open("a") as f:
        f.write(json.dumps({k: v for k, v in entry.items() if k != "overlay_png"}) + "\n")
    return {"ok": True}


@app.get("/api/edge/reports")
def edge_reports(limit: int = 50):
    return {"reports": list(_edge_reports)[:limit]}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
