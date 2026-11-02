"""
edge_infer.py
─────────────
TrustEdge-CXR on a Raspberry Pi / any small Linux box (IoT edge node).

For each chest X-ray it runs the INT8 TFLite backbone once, then the numpy
head 20× with MC dropout, and prints one JSON line:

    {"file": ..., "decision": "PNEUMONIA" | "NORMAL" | "REFER",
     "prob_pneumonia": ..., "uncertainty": ..., "latency_ms": ...}

Optionally saves a Grad-CAM++ overlay and reports the result to the
hospital server (FastAPI backend, POST /api/edge/report). Images never
leave the device unless --send-image is given.

Usage (inside the exported edge_bundle/ folder):
    python edge_infer.py xray1.jpeg xray2.png
    python edge_infer.py --watch /path/to/scanner_output --server http://192.168.1.20:8000
"""

import argparse
import base64
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

try:  # inside the project
    from edge.trustedge_runtime import TFLiteBackbone, TrustEdgeHead, overlay_heatmap, preprocess_xray
except ImportError:  # inside the copied edge_bundle/
    from trustedge_runtime import TFLiteBackbone, TrustEdgeHead, overlay_heatmap, preprocess_xray

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class EdgeNode:
    def __init__(self, bundle: Path, threads: int):
        self.backbone = TFLiteBackbone(bundle / "backbone_int8.tflite", num_threads=threads)
        self.head = TrustEdgeHead.load(bundle / "head.npz", bundle / "calibration.json")

    def analyze(self, path: Path) -> tuple[dict, np.ndarray]:
        t0 = time.perf_counter()
        img = preprocess_xray(path)
        fmap = self.backbone(img)
        pred = self.head.predict(fmap)
        cam = self.head.cam(fmap)[0]
        result = {
            "file": path.name,
            "decision": str(pred["decision"][0]),
            "prob_pneumonia": round(float(pred["prob"][0]), 4),
            "uncertainty": round(float(pred["uncertainty"][0]), 4),
            "threshold": self.head.threshold,
            "tau": self.head.tau,
            "latency_ms": round(1000 * (time.perf_counter() - t0), 1),
        }
        return result, overlay_heatmap(img, cam)


def png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def report(server: str, payload: dict, token: str | None):
    req = urllib.request.Request(
        server.rstrip("/") + "/api/edge/report", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **({"X-Edge-Token": token} if token else {})})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status


def process(node, path, args):
    result, overlay = node.analyze(path)
    if args.save_overlay:
        out = Path(args.save_overlay)
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(out / f"{path.stem}_gradcampp.png")
    if args.server:
        payload = {**result, "device_id": args.device_id, "timestamp": time.time(),
                   **({"overlay_png": png_b64(overlay)} if args.send_image else {})}
        try:
            report(args.server, payload, args.token)
            result["reported"] = True
        except OSError as e:
            result["reported"] = f"failed: {e}"
    print(json.dumps(result), flush=True)


def main():
    here = Path(__file__).resolve().parent
    default_bundle = here if (here / "backbone_int8.tflite").exists() else here.parent / "artifacts" / "edge_bundle"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--bundle", type=Path, default=default_bundle)
    parser.add_argument("--watch", type=Path, help="process new X-rays dropped into this folder")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--server", help="hospital server URL, e.g. http://192.168.1.20:8000")
    parser.add_argument("--token", default=os.environ.get("TRUSTEDGE_EDGE_TOKEN"))
    parser.add_argument("--device-id", default=os.uname().nodename if hasattr(os, "uname") else "edge")
    parser.add_argument("--send-image", action="store_true", help="include the heatmap overlay in the report")
    parser.add_argument("--save-overlay", help="folder for Grad-CAM++ overlays")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    node = EdgeNode(args.bundle, args.threads)
    for path in args.images:
        process(node, path, args)

    if args.watch:
        print(f"Watching {args.watch} for new X-rays (Ctrl+C to stop) …", flush=True)
        seen = {p for p in args.watch.iterdir()}
        while True:
            for path in sorted(args.watch.iterdir()):
                if path not in seen and path.suffix.lower() in IMAGE_EXTS:
                    seen.add(path)
                    time.sleep(0.2)  # let the writer finish the file
                    process(node, path, args)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
