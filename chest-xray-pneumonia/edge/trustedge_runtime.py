"""
trustedge_runtime.py
────────────────────
NumPy-only inference logic for TrustEdge-CXR. The same file runs on the
Raspberry Pi, inside the FastAPI server and in the evaluation scripts, so
every place reaches its decision in exactly the same way.

  preprocess_xray()   grayscale → resize 224 → CLAHE → 3-channel uint8
  TrustEdgeHead       classifier head applied to the backbone feature map
                        • MC-dropout uncertainty (backbone runs once, only
                          the small head is sampled → ~1 % extra compute)
                        • temperature-scaled probability
                        • decision rule: REFER / PNEUMONIA / NORMAL
                        • exact Grad-CAM and Grad-CAM++ (no autodiff needed)
  TFLiteBackbone      INT8 MobileNetV2 feature extractor for edge devices
  overlay_heatmap()   jet-coloured heatmap blended over the X-ray

Only numpy and Pillow are required; OpenCV is needed for CLAHE and a
TFLite interpreter (ai-edge-litert, tflite-runtime or tensorflow) for
TFLiteBackbone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 224
CLAHE_CLIP = 2.0
CLAHE_GRID = 8
CLASS_NAMES = ("NORMAL", "PNEUMONIA")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on a uint8 image."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("CLAHE needs OpenCV: pip install opencv-python-headless") from e
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    return clahe.apply(gray)


def preprocess_xray(img, size: int = IMG_SIZE, clahe: bool = True) -> np.ndarray:
    """
    PIL image, file path or array → (size, size, 3) uint8.

    Resizing happens before CLAHE so the equalisation tiles cover the same
    anatomy whatever the source resolution (Kermany ≈1–2k px, RSNA 1024 px).
    """
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    gray = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.uint8)
    if clahe:
        gray = apply_clahe(gray)
    return np.repeat(gray[..., None], 3, axis=-1)


# ── Classifier head ───────────────────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class TrustEdgeHead:
    """
    GAP → Dropout → Dense(ReLU) → Dropout → Dense(1) on a (N, H, W, C)
    MobileNetV2 feature map, plus the calibration learned on validation data.
    """

    def __init__(self, w1, b1, w2, b2, dropout1=0.3, dropout2=0.2,
                 temperature=1.0, threshold=0.5, tau=None, mc_passes=20):
        self.w1 = np.asarray(w1, np.float32)
        self.b1 = np.asarray(b1, np.float32)
        self.w2 = np.asarray(w2, np.float32)
        self.b2 = np.asarray(b2, np.float32)
        self.dropout1 = float(dropout1)
        self.dropout2 = float(dropout2)
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.tau = None if tau is None else float(tau)
        self.mc_passes = int(mc_passes)

    @classmethod
    def load(cls, head_path, calibration_path=None) -> "TrustEdgeHead":
        h = np.load(head_path)
        cal = {}
        if calibration_path and Path(calibration_path).exists():
            cal = json.loads(Path(calibration_path).read_text())
        return cls(
            h["w1"], h["b1"], h["w2"], h["b2"],
            dropout1=float(h["dropout1"]), dropout2=float(h["dropout2"]),
            temperature=cal.get("temperature", 1.0),
            threshold=cal.get("threshold", 0.5),
            tau=cal.get("tau"),
            mc_passes=cal.get("mc_passes", 20),
        )

    @staticmethod
    def _pool(fmap):
        fmap = np.asarray(fmap, np.float32)
        n, c = fmap.shape[0], fmap.shape[-1]
        return fmap.reshape(n, -1, c).mean(axis=1)

    def logits(self, fmap, mc_passes: int = 0, seed: int = 0):
        """
        Deterministic logits (N,) and, when mc_passes > 0, Monte-Carlo
        dropout logits (P, N) computed with inverted-dropout masks exactly
        as Keras applies them during training.
        """
        g = self._pool(fmap)
        z = (np.maximum(g @ self.w1 + self.b1, 0.0) @ self.w2 + self.b2)[:, 0]
        if not mc_passes:
            return z, None

        rng = np.random.default_rng(seed)
        p, (n, c) = mc_passes, g.shape
        keep1, keep2 = 1.0 - self.dropout1, 1.0 - self.dropout2
        m1 = (rng.random((p, n, c)) < keep1).astype(np.float32) / keep1
        h = np.maximum((g[None] * m1) @ self.w1 + self.b1, 0.0)
        m2 = (rng.random(h.shape) < keep2).astype(np.float32) / keep2
        z_mc = ((h * m2) @ self.w2 + self.b2)[..., 0]
        return z, z_mc

    def predict(self, fmap, mc_passes: int | None = None, seed: int = 0) -> dict:
        """
        Returns arrays (one entry per image):
          logit        deterministic logit
          prob         calibrated P(pneumonia) (MC mean when sampling)
          uncertainty  std of the MC probabilities (0 without sampling)
          label        0/1 at the tuned threshold
          refer        True when uncertainty > tau
          decision     "REFER" | "PNEUMONIA" | "NORMAL"
        """
        passes = self.mc_passes if mc_passes is None else mc_passes
        z, z_mc = self.logits(fmap, passes, seed)
        if z_mc is not None:
            p_mc = _sigmoid(z_mc / self.temperature)
            prob, unc = p_mc.mean(axis=0), p_mc.std(axis=0)
        else:
            prob, unc = _sigmoid(z / self.temperature), np.zeros_like(z)
        label = (prob >= self.threshold).astype(int)
        refer = unc > self.tau if self.tau is not None else np.zeros_like(label, bool)
        decision = np.where(refer, "REFER", np.where(label == 1, CLASS_NAMES[1], CLASS_NAMES[0]))
        return {"logit": z, "prob": prob, "uncertainty": unc,
                "label": label, "refer": refer, "decision": decision}

    def cam(self, fmap, method: str = "gradcam++") -> np.ndarray:
        """
        Heatmap (N, H, W) in [0, 1] for the pneumonia logit S.

        The head is GAP → MLP, so dS/dA is the same at every location:
        dS/dA_k = (dS/dg_k) / (H·W), with dS/dg computed in closed form.
        Grad-CAM++ then follows Chattopadhay et al. (2018) with the usual
        exponential-score simplification.
        """
        fmap = np.asarray(fmap, np.float32)
        n, hh, ww, c = fmap.shape
        g = self._pool(fmap)
        active = (g @ self.w1 + self.b1) > 0
        ds_dg = (active * self.w2[:, 0]) @ self.w1.T             # (N, C)
        grads = ds_dg / (hh * ww)                                  # dS/dA_k

        if method == "gradcam":
            weights = grads
        elif method == "gradcam++":
            g2, g3 = grads ** 2, grads ** 3
            denom = 2.0 * g2 + fmap.sum(axis=(1, 2)) * g3
            alpha = np.where(grads != 0, g2 / (denom + 1e-12), 0.0)
            weights = alpha * np.maximum(grads, 0.0) * (hh * ww)   # Σ over locations
        else:
            raise ValueError(f"unknown method {method!r}")

        cam = np.maximum(np.einsum("nhwc,nc->nhw", fmap, weights), 0.0)
        return cam / (cam.max(axis=(1, 2), keepdims=True) + 1e-8)


# ── Edge backbone (TFLite) ────────────────────────────────────────────────────
def _find_interpreter():
    for module in ("ai_edge_litert.interpreter",        # LiteRT (current)
                   "tflite_runtime.interpreter",        # older Raspberry Pi wheels
                   "tensorflow.lite.python.interpreter"):
        try:
            return __import__(module, fromlist=["Interpreter"]).Interpreter
        except (ImportError, AttributeError):
            continue
    raise ImportError("Install a TFLite interpreter: pip install ai-edge-litert "
                      "(or tflite-runtime / tensorflow).")


class TFLiteBackbone:
    """Runs the exported MobileNetV2 feature extractor one image at a time."""

    def __init__(self, model_path, num_threads: int = 4):
        Interpreter = _find_interpreter()
        self.interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def __call__(self, images: np.ndarray) -> np.ndarray:
        images = np.asarray(images)
        if images.ndim == 3:
            images = images[None]
        outs = []
        for img in images:
            x = img[None]
            dtype = self.inp["dtype"]
            if dtype in (np.uint8, np.int8) and x.dtype != dtype:
                scale, zero = self.inp["quantization"]
                x = np.round(x.astype(np.float32) / scale + zero) if scale else x
                info = np.iinfo(dtype)
                x = np.clip(x, info.min, info.max)
            self.interp.set_tensor(self.inp["index"], x.astype(dtype))
            self.interp.invoke()
            y = self.interp.get_tensor(self.out["index"])
            if self.out["dtype"] in (np.uint8, np.int8):
                scale, zero = self.out["quantization"]
                y = (y.astype(np.float32) - zero) * scale
            outs.append(y[0].astype(np.float32))
        return np.stack(outs)


# ── Visualisation ─────────────────────────────────────────────────────────────
def _jet(x: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def upsample_cam(cam: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """(h, w) heatmap in [0, 1] → (size, size) with bicubic interpolation."""
    img = Image.fromarray(np.float32(cam), mode="F").resize((size, size), Image.BICUBIC)
    return np.clip(np.asarray(img), 0.0, 1.0)


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet-coloured heatmap over an (H, W, 3) uint8 image."""
    heat = upsample_cam(cam, rgb.shape[0])
    colour = (_jet(heat) * 255.0).astype(np.float32)
    out = (1.0 - alpha) * rgb.astype(np.float32) + alpha * colour
    return np.clip(out, 0, 255).astype(np.uint8)
