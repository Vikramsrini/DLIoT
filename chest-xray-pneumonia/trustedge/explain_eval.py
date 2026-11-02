"""
explain_eval.py — Step 6
────────────────────────
Tests the heatmaps instead of only showing them (gap G4, Zhang et al. 2024).

  PSC-style test   N test images × K small perturbations (noise, brightness,
                   contrast, 6-px shifts). Correlates how much the prediction
                   moves with how much the heatmap moves (Spearman ρ). A
                   trustworthy map changes when the prediction changes and
                   stays put when it does not. This is a simplified,
                   random-perturbation version of the Prediction–Saliency
                   Correlation metric, not the paper's exact protocol.
  Pointing game    On RSNA pneumonia images, does the heatmap peak fall
                   inside a radiologist's box? (needs `trustedge.rsna`)

Grad-CAM and Grad-CAM++ are compared; an example grid is saved.

Run:
    python -m trustedge.explain_eval
"""

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from .config import ROOT, Paths, add_common_args, banner, seed_everything  # noqa: E402
from .metrics import sigmoid  # noqa: E402
from .model import compute_fmaps, feature_model, load_student  # noqa: E402
from .splits import load_splits  # noqa: E402
from .tfdata import load_images  # noqa: E402

sys.path.insert(0, str(ROOT))
from edge.trustedge_runtime import TrustEdgeHead, overlay_heatmap, upsample_cam  # noqa: E402

METHODS = ("gradcam", "gradcam++")


def perturb(img, rng):
    x = img.astype(np.float32)
    kind = rng.choice(["noise", "brightness", "contrast", "shift"])
    if kind == "noise":
        x = x + rng.normal(0.0, rng.uniform(2.0, 8.0), x.shape[:2])[..., None]
    elif kind == "brightness":
        x = x + rng.uniform(-15.0, 15.0)
    elif kind == "contrast":
        x = (x - x.mean()) * rng.uniform(0.85, 1.15) + x.mean()
    else:
        dy, dx = rng.integers(-6, 7, size=2)
        x = np.roll(x, (int(dy), int(dx)), axis=(0, 1))
    return np.clip(x, 0, 255).astype(np.uint8)


def map_change(a, b):
    """1 − Pearson correlation between two flattened heatmaps."""
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0 if np.allclose(a, b) else 1.0
    return float(1.0 - np.corrcoef(a, b)[0, 1])


def psc_test(fmodel, head, images, rng, k):
    base_f = compute_fmaps(fmodel, images)
    pert = np.stack([perturb(img, rng) for img in images for _ in range(k)])
    pert_f = compute_fmaps(fmodel, pert)
    p0 = sigmoid(head.logits(base_f)[0] / head.temperature)
    p1 = sigmoid(head.logits(pert_f)[0] / head.temperature)
    dp = np.abs(p1 - np.repeat(p0, k))
    out = {"n_images": int(len(images)), "perturbations_per_image": k}
    for method in METHODS:
        c0 = np.repeat(head.cam(base_f, method), k, axis=0)
        c1 = head.cam(pert_f, method)
        ds = np.array([map_change(a, b) for a, b in zip(c0, c1)])
        stable, moved = dp < 0.02, dp >= 0.10
        out[method] = {
            "psc_spearman": float(spearmanr(dp, ds).correlation),
            "map_change_when_prediction_stable": float(ds[stable].mean()) if stable.any() else None,
            "map_change_when_prediction_moves": float(ds[moved].mean()) if moved.any() else None,
        }
    return out


def pointing_game(fmodel, head, P):
    ext = pd.read_csv(P.rsna_manifest)
    pos = ext[(ext.label == 1) & ext.boxes.notna()].reset_index(drop=True)
    if pos.empty:
        return None
    images = load_images(pos.rename(columns={"clahe_relpath": "cache_relpath"}), P.rsna)
    fmaps = compute_fmaps(fmodel, images)
    yy, xx = np.mgrid[0:images.shape[1], 0:images.shape[2]]
    out = {"n_images": int(len(pos))}
    masks = []
    for boxes in pos.boxes:
        m = np.zeros(images.shape[1:3], bool)
        for x, y, w, h in json.loads(boxes):
            m |= (xx >= x) & (xx < x + w) & (yy >= y) & (yy < y + h)
        masks.append(m)
    out["chance_hit_rate"] = float(np.mean([m.mean() for m in masks]))
    for method in METHODS:
        cams = head.cam(fmaps, method)
        hits, energy = [], []
        for cam, m in zip(cams, masks):
            up = upsample_cam(cam, images.shape[1])
            r, c = np.unravel_index(np.argmax(up), up.shape)
            hits.append(bool(m[r, c]))
            energy.append(float(up[m].sum() / (up.sum() + 1e-8)))
        out[method] = {"hit_rate": float(np.mean(hits)), "energy_in_boxes": float(np.mean(energy))}
    return out


def example_grid(head, fmodel, test, images, P, seed):
    rng = np.random.default_rng(seed)
    picks = [i for label in (1, 0) for i in rng.choice(np.flatnonzero(test.label == label),
                                                       min(4, (test.label == label).sum()), replace=False)]
    fmaps = compute_fmaps(fmodel, images[picks])
    pred = head.predict(fmaps, seed=seed)
    cams = head.cam(fmaps, "gradcam++")
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax, i, cam, d, p in zip(axes.ravel(), picks, cams, pred["decision"], pred["prob"]):
        ax.imshow(overlay_heatmap(images[i], cam))
        truth = "PNEUMONIA" if test.label.iloc[i] else "NORMAL"
        ax.set_title(f"truth {truth}\n→ {d}  (p = {p:.2f})", fontsize=10)
        ax.axis("off")
    fig.suptitle("TrustEdge-CXR — Grad-CAM++ on official test images", fontweight="bold")
    fig.tight_layout()
    fig.savefig(P.results / "heatmap_examples.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--student", default="student")
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--perturbations", type=int, default=8)
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    P = Paths(args.artifacts)
    P.ensure(P.results)
    rng = np.random.default_rng(args.seed)

    banner("STEP 6 — Explanation quality")
    sdir = P.root / args.student
    model = load_student(sdir / "student.keras")
    fmodel = feature_model(model)
    head = TrustEdgeHead.load(sdir / "head.npz", sdir / "calibration.json")

    df = load_splits(P.splits, args.limit, args.seed)
    test = df[df.split == "test"].reset_index(drop=True)
    images = load_images(test, P.cache_clahe)
    idx = rng.choice(len(test), min(args.n_images, len(test)), replace=False)

    report = {"psc": psc_test(fmodel, head, images[idx], rng, args.perturbations)}
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    for method in METHODS:
        r = report["psc"][method]
        print(f"  {method:10s} PSC ρ = {r['psc_spearman']:.3f} · map change when prediction is stable "
              f"{fmt(r['map_change_when_prediction_stable'])} · when it moves "
              f"{fmt(r['map_change_when_prediction_moves'])}")
    if P.rsna_manifest.exists():
        report["pointing_game"] = pg = pointing_game(fmodel, head, P)
        if pg:
            print(f"  pointing game on {pg['n_images']} RSNA images (chance {pg['chance_hit_rate']:.2f}): "
                  + ", ".join(f"{m} {pg[m]['hit_rate']:.2f}" for m in METHODS))
    else:
        print("  RSNA manifest not found — pointing game skipped (run `python -m trustedge.rsna`).")

    example_grid(head, fmodel, test, images, P, args.seed)
    (P.results / "explanations.json").write_text(json.dumps(report, indent=2))
    print(f"\n✅ explanations → {P.results / 'explanations.json'}")


if __name__ == "__main__":
    main()
