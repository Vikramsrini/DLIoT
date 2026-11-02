"""
calibrate.py — Step 4
─────────────────────
Fits three numbers on the patient-level validation set:

  temperature T   minimises validation NLL → calibrated probabilities
  threshold t*    highest specificity with sensitivity ≥ 95 %
                  (replaces Review 1's fixed 0.5)
  tau τ           MC-dropout uncertainty above which a case is referred
                  to a radiologist (default: the most uncertain 10 %)

Writes calibration.json next to the student model.

Run:
    python -m trustedge.calibrate
"""

import argparse
import json
import sys

import numpy as np

from .config import ROOT, Paths, add_common_args, banner, seed_everything
from .metrics import binary_metrics, fit_temperature, pick_threshold, sigmoid
from .model import compute_fmaps, feature_model, load_student
from .splits import load_splits
from .tfdata import load_images

sys.path.insert(0, str(ROOT))
from edge.trustedge_runtime import TrustEdgeHead  # noqa: E402


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--student", default="student", help="sub-folder of artifacts/")
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--referral-rate", type=float, default=0.10)
    parser.add_argument("--mc-passes", type=int, default=20)
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    P = Paths(args.artifacts)
    sdir = P.root / args.student

    banner("STEP 4 — Calibration, decision threshold, referral rule")
    df = load_splits(P.splits, args.limit, args.seed)
    val = df[df.split == "val"]
    y = val["label"].to_numpy()

    model = load_student(sdir / "student.keras")
    fmaps = compute_fmaps(feature_model(model), load_images(val, P.cache_clahe))
    head = TrustEdgeHead.load(sdir / "head.npz")

    z, _ = head.logits(fmaps)
    head.temperature = fit_temperature(y, z)
    pred = head.predict(fmaps, mc_passes=args.mc_passes, seed=args.seed)
    threshold = pick_threshold(y, pred["prob"], args.target_sensitivity)
    tau = float(np.quantile(pred["uncertainty"], 1.0 - args.referral_rate))

    before = binary_metrics(y, sigmoid(z), 0.5)
    after = binary_metrics(y, pred["prob"], threshold)
    print(f"temperature T = {head.temperature:.3f}")
    print(f"threshold  t* = {threshold:.4f}   (val sensitivity {after['sensitivity']:.3f}, "
          f"specificity {after['specificity']:.3f})")
    print(f"tau         τ = {tau:.4f}   (refers {args.referral_rate:.0%} of validation cases)")
    print(f"val ECE {before['ece']:.4f} → {after['ece']:.4f}")

    calibration = {
        "temperature": head.temperature, "threshold": threshold, "tau": tau,
        "mc_passes": args.mc_passes, "target_sensitivity": args.target_sensitivity,
        "referral_rate": args.referral_rate,
        "val_uncalibrated@0.5": before, "val_calibrated@tuned": after,
    }
    (sdir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    print(f"\n✅ calibration → {sdir / 'calibration.json'}")


if __name__ == "__main__":
    main()
