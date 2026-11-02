"""
export_tflite.py — Step 7
─────────────────────────
Builds the bundle that runs on a Raspberry Pi or Android phone:

  backbone_int8.tflite   MobileNetV2 feature extractor, full-integer INT8
                         post-training quantization (300 calibration images
                         from the training split), takes raw uint8 pixels
  backbone_fp32.tflite   same network in float32, for comparison
  head.npz, calibration.json, trustedge_runtime.py, edge_infer.py

Then checks the INT8 pipeline against the Keras model on the test set and
records size, accuracy and latency in edge_metrics.json.

Run:
    python -m trustedge.export_tflite
"""

import argparse
import json
import shutil
import sys
import time

import keras
import numpy as np
import tensorflow as tf

from .config import ROOT, Paths, add_common_args, banner, seed_everything
from .metrics import binary_metrics
from .model import compute_fmaps, feature_model, load_student
from .splits import load_splits
from .tfdata import load_images

sys.path.insert(0, str(ROOT))
from edge.trustedge_runtime import TFLiteBackbone, TrustEdgeHead  # noqa: E402


def export_saved_model(fmodel, path):
    """Keras 3 → SavedModel with a fixed 1×224×224×3 input (frozen weights)."""
    archive = keras.export.ExportArchive()
    archive.track(fmodel)
    archive.add_endpoint(name="serve", fn=lambda x: fmodel(x, training=False),
                         input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32)])
    archive.write_out(str(path), verbose=False)


def convert(saved_model_dir, rep_images=None) -> bytes:
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if rep_images is not None:
        def representative():
            for img in rep_images:
                yield [img[None].astype(np.float32)]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8
    return conv.convert()


def latency_ms(fn, x, runs=30) -> float:
    fn(x)  # warm-up
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(x)
    return 1000.0 * (time.perf_counter() - t0) / runs


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--student", default="student")
    parser.add_argument("--calibration-images", type=int, default=300)
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    P = Paths(args.artifacts)
    B = P.edge_bundle
    P.ensure(B)
    sdir = P.root / args.student

    banner("STEP 7 — Edge export (TFLite INT8)")
    df = load_splits(P.splits, args.limit, args.seed)
    model = load_student(sdir / "student.keras")
    fmodel = feature_model(model)

    train = df[df.split == "train"]
    rep = train.sample(min(args.calibration_images, len(train)), random_state=args.seed)
    files = {"fp32": B / "backbone_fp32.tflite", "int8": B / "backbone_int8.tflite"}
    saved = P.root / "student_savedmodel"
    export_saved_model(fmodel, saved)
    files["fp32"].write_bytes(convert(saved))
    files["int8"].write_bytes(convert(saved, load_images(rep, P.cache_clahe)))
    for name in ("head.npz", "calibration.json"):
        shutil.copy(sdir / name, B / name)
    edge_src = ROOT / "edge"
    for name in ("trustedge_runtime.py", "edge_infer.py", "requirements-edge.txt", "README.md"):
        shutil.copy(edge_src / name, B / name)

    head = TrustEdgeHead.load(B / "head.npz", B / "calibration.json")
    test = df[df.split == "test"]
    y = test["label"].to_numpy()
    images = load_images(test, P.cache_clahe)

    print("Running the test set through Keras FP32 and TFLite INT8 …")
    pred_keras = head.predict(compute_fmaps(fmodel, images), seed=args.seed)
    int8 = TFLiteBackbone(files["int8"])
    pred_int8 = head.predict(int8(images), seed=args.seed)

    fp32 = TFLiteBackbone(files["fp32"])
    report = {
        "fp32_mb": files["fp32"].stat().st_size / 1e6,
        "int8_mb": files["int8"].stat().st_size / 1e6,
        "head_kb": (B / "head.npz").stat().st_size / 1e3,
        "keras": {"test@tuned": binary_metrics(y, pred_keras["prob"], head.threshold)},
        "int8": {"test@tuned": binary_metrics(y, pred_int8["prob"], head.threshold)},
        "decision_agreement": float((pred_keras["decision"] == pred_int8["decision"]).mean()),
        "mean_abs_prob_diff": float(np.abs(pred_keras["prob"] - pred_int8["prob"]).mean()),
        "latency_ms_on_this_machine": {
            "fp32_backbone": latency_ms(fp32, images[:1]),
            "int8_backbone": latency_ms(int8, images[:1]),
            "head_mc20_plus_gradcam": latency_ms(
                lambda x: (head.predict(x), head.cam(x)), int8(images[:1])),
        },
    }
    (B / "edge_metrics.json").write_text(json.dumps(report, indent=2))

    k, q = report["keras"]["test@tuned"], report["int8"]["test@tuned"]
    print(f"  size      FP32 {report['fp32_mb']:.2f} MB → INT8 {report['int8_mb']:.2f} MB "
          f"(+ {report['head_kb']:.0f} kB head)")
    print(f"  accuracy  Keras {k['accuracy']:.4f} → INT8 {q['accuracy']:.4f}  ·  "
          f"AUC {k['auc']:.4f} → {q['auc']:.4f}")
    print(f"  decisions identical on {report['decision_agreement']:.1%} of test images")
    lat = report["latency_ms_on_this_machine"]
    print(f"  latency   FP32 {lat['fp32_backbone']:.1f} ms · INT8 {lat['int8_backbone']:.1f} ms · "
          f"head {lat['head_mc20_plus_gradcam']:.1f} ms (this machine, 1 image)")
    print(f"\n✅ edge bundle → {B}")


if __name__ == "__main__":
    main()
