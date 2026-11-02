"""
evaluate.py — Step 5
────────────────────
Before/after comparison on the official 624-image test set:

  Review 1 baseline      models/pneumonia_mobilenetv2_final.keras, threshold 0.5
  RAD-DINO heads         linear probe and MLP teacher (from step 2)
  Student without KD     artifacts/student_nokd (ablation, if trained)
  TrustEdge student      calibrated, tuned threshold, MC dropout + referral
  Edge INT8 / federated  copied in when those steps have run

Writes results/test_metrics.json, results/comparison.md, per-image
predictions and three figures (confusion matrices, ROC, reliability).

Run:
    python -m trustedge.evaluate
    python -m trustedge.evaluate --external     # also the RSNA adult test set
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .config import BASELINE_MODEL, ROOT, Paths, add_common_args, banner, seed_everything  # noqa: E402
from .metrics import binary_metrics, referral_metrics, sigmoid  # noqa: E402
from .model import compute_fmaps, feature_model, load_student  # noqa: E402
from .splits import load_splits  # noqa: E402
from .tfdata import load_images, load_raw_images  # noqa: E402

sys.path.insert(0, str(ROOT))
from edge.trustedge_runtime import TrustEdgeHead  # noqa: E402

PRIMARY = ["accuracy", "sensitivity", "specificity", "f1", "auc", "ece"]


def eval_student(sdir: Path, images, y, seed):
    model = load_student(sdir / "student.keras")
    head = TrustEdgeHead.load(sdir / "head.npz", sdir / "calibration.json")
    fmaps = compute_fmaps(feature_model(model), images)
    pred = head.predict(fmaps, seed=seed)
    res = {
        "uncalibrated@0.5": binary_metrics(y, sigmoid(pred["logit"]), 0.5),
        "calibrated@tuned": binary_metrics(y, pred["prob"], head.threshold),
        "referral": referral_metrics(y, pred["prob"], head.threshold, pred["refer"]),
        "params": int(model.count_params()),
    }
    return res, pred


def eval_baseline(df, data_dir, y):
    import keras
    model = keras.models.load_model(BASELINE_MODEL, compile=False)
    prob = model.predict(load_raw_images(df, data_dir), batch_size=64, verbose=0)[:, 0]
    return {"@0.5": binary_metrics(y, prob, 0.5), "params": int(model.count_params())}, prob


def fmt_row(name, m, extra=""):
    cells = [f"{m[k]:.3f}" if k in ("auc", "ece") else f"{100 * m[k]:.1f}%" for k in PRIMARY]
    return f"| {name} | {m['threshold']:.3f} | " + " | ".join(cells) + f" | {extra} |"


def plots(P, y, curves, cms, reliab):
    fig, axes = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4.2))
    for ax, (name, m) in zip(np.atleast_1d(axes), cms.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["NORMAL", "PNEUMONIA"])
        ax.set_yticks([0, 1], ["NORMAL", "PNEUMONIA"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nacc {m['accuracy']:.1%} · spec {m['specificity']:.1%}")
    fig.tight_layout()
    fig.savefig(P.results / "confusion_matrices.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, prob in curves.items():
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {binary_metrics(y, prob)['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — official test set")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(P.results / "roc_curves.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    edges = np.linspace(0, 1, 11)
    for name, prob in reliab.items():
        idx = np.clip(np.digitize(prob, edges[1:-1]), 0, 9)
        xs = [prob[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(xs, ys, "o-", label=f"{name} (ECE {binary_metrics(y, prob)['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("Predicted P(pneumonia)")
    ax.set_ylabel("Observed fraction with pneumonia")
    ax.set_title("Reliability diagram — official test set")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(P.results / "reliability.png", dpi=150)
    plt.close("all")


def evaluate_external(P, args, report):
    if not P.rsna_manifest.exists():
        print("RSNA manifest not found — run `python -m trustedge.rsna` first; skipping external test.")
        return
    banner("External test — RSNA adult chest X-rays")
    ext = pd.read_csv(P.rsna_manifest)
    y = ext["label"].to_numpy()
    out = {}
    if BASELINE_MODEL.exists():
        import keras
        model = keras.models.load_model(BASELINE_MODEL, compile=False)
        raw = load_raw_images(ext.rename(columns={"raw_relpath": "relpath"}), P.rsna)
        out["baseline@0.5"] = binary_metrics(y, model.predict(raw, batch_size=64, verbose=0)[:, 0], 0.5)
    images = load_images(ext.rename(columns={"clahe_relpath": "cache_relpath"}), P.rsna)
    res, _ = eval_student(P.root / args.student, images, y, args.seed)
    out["trustedge"] = res
    for name, m in (("Review 1 baseline", out.get("baseline@0.5")),
                    ("TrustEdge student", res["calibrated@tuned"])):
        if m:
            print(f"  {name:20s} acc {m['accuracy']:.3f} sens {m['sensitivity']:.3f} "
                  f"spec {m['specificity']:.3f} AUC {m['auc']:.3f}")
    report["external_rsna"] = out


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--student", default="student", help="sub-folder of artifacts/")
    parser.add_argument("--external", action="store_true", help="also evaluate on RSNA")
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    P = Paths(args.artifacts)
    P.ensure(P.results)

    banner("STEP 5 — Test-set comparison")
    df = load_splits(P.splits, args.limit, args.seed)
    test = df[df.split == "test"].reset_index(drop=True)
    y = test["label"].to_numpy()
    report, rows, curves, cms, reliab = {}, [], {}, {}, {}
    preds = test[["relpath", "label"]].copy()

    if BASELINE_MODEL.exists():
        res, prob = eval_baseline(test, args.data_dir, y)
        report["baseline"] = res
        rows.append(fmt_row("Review 1 baseline (MobileNetV2)", res["@0.5"], f"{res['params'] / 1e6:.1f} M params"))
        curves["Review 1 baseline"] = reliab["Review 1 baseline"] = prob
        cms["Review 1 baseline"] = res["@0.5"]
        preds["baseline_prob"] = prob
    else:
        print(f"Baseline model not found at {BASELINE_MODEL} — skipping that row.")

    if P.teacher_metrics.exists():
        tm = json.loads(P.teacher_metrics.read_text())
        report["teacher"] = tm
        t = pd.read_csv(P.teacher_logits).set_index("relpath")
        for name, label in (("linear_probe", "RAD-DINO linear probe [10]"), ("mlp_head", "RAD-DINO + MLP head (teacher)")):
            rows.append(fmt_row(label, tm[name]["test@tuned"], "86.6 M params, 518 px"))
        if set(test["relpath"]) <= set(t.index):
            curves["RAD-DINO teacher"] = sigmoid(t.loc[test["relpath"], "teacher_logit"].to_numpy())

    images = load_images(test, P.cache_clahe)
    nokd = P.root / "student_nokd"
    if (nokd / "calibration.json").exists():
        res, _ = eval_student(nokd, images, y, args.seed)
        report["student_nokd"] = res
        rows.append(fmt_row("Student without KD (ablation)", res["calibrated@tuned"], "2.4 M params"))

    sdir = P.root / args.student
    res, pred = eval_student(sdir, images, y, args.seed)
    report["trustedge"] = res
    m, r = res["calibrated@tuned"], res["referral"]
    rows.append(fmt_row("Student, uncalibrated @0.5", res["uncalibrated@0.5"], "2.4 M params"))
    rows.append(fmt_row("**TrustEdge student (all cases)**", m, f"{res['params'] / 1e6:.1f} M params"))
    kept = {**m, "accuracy": r.get("kept_accuracy", np.nan), "sensitivity": r.get("kept_sensitivity", np.nan),
            "specificity": r.get("kept_specificity", np.nan)}
    rows.append(f"| TrustEdge, auto-decided cases only | {m['threshold']:.3f} | {100 * kept['accuracy']:.1f}% | "
                f"{100 * kept['sensitivity']:.1f}% | {100 * kept['specificity']:.1f}% | – | – | – | "
                f"{r['referred']} of {len(y)} referred |")
    curves["TrustEdge student"] = reliab["TrustEdge (calibrated)"] = pred["prob"]
    reliab["TrustEdge (uncalibrated)"] = sigmoid(pred["logit"])
    cms["TrustEdge student"] = m
    preds = preds.assign(student_prob=pred["prob"], uncertainty=pred["uncertainty"], decision=pred["decision"])

    edge_metrics = P.edge_bundle / "edge_metrics.json"
    if edge_metrics.exists():
        em = json.loads(edge_metrics.read_text())
        report["edge_int8"] = em
        rows.append(fmt_row("Edge INT8 TFLite + numpy head", em["int8"]["test@tuned"], f"{em['int8_mb']:.1f} MB backbone"))
    for fl_metrics in sorted(P.root.glob("federated*/fl_metrics.json")):
        fm = json.loads(fl_metrics.read_text())
        report[fl_metrics.parent.name] = fm
        dp = fm.get("dp")
        note = (f"DP σ={dp['noise_multiplier']}, ε={dp['epsilon']:.1f}" if dp and dp.get("epsilon")
                else "DP on" if dp else "no DP")
        rows.append(fmt_row(f"Federated student ({fm['config']['clients']} sites)", fm["test@tuned"], note))

    if args.external:
        evaluate_external(P, args, report)

    header = ("| Model | Threshold | Accuracy | Sensitivity | Specificity | F1 | AUC | ECE | Notes |\n"
              "|---|---|---|---|---|---|---|---|---|")
    table = header + "\n" + "\n".join(rows)
    (P.results / "comparison.md").write_text(
        f"# Official test set ({len(y)} images)\n\n{table}\n")
    (P.results / "test_metrics.json").write_text(json.dumps(report, indent=2, default=float))
    preds.to_csv(P.results / "test_predictions.csv", index=False)
    plots(P, y, curves, cms, reliab)
    print("\n" + table)
    print(f"\n✅ results → {P.results}")


if __name__ == "__main__":
    main()
