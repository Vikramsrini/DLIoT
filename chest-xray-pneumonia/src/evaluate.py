"""
evaluate.py
───────────
Full test-set evaluation of the trained pneumonia classifier.

Outputs:
  • Classification report  (precision / recall / F1 per class)
  • ROC-AUC score
  • Confusion matrix PNG   → models/confusion_matrix.png
  • Training curves PNG    → models/training_curves.png  (if CSV logs exist)

Run:
    cd src && python evaluate.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from data import get_datasets

# ── Load data & model ─────────────────────────────────────────────────────────
_, _, test_ds, class_names = get_datasets()

MODEL_PATH = "../models/pneumonia_mobilenetv2_final.keras"
print(f"Loading model from {MODEL_PATH} …")
model = tf.keras.models.load_model(MODEL_PATH)

# ── Predictions ───────────────────────────────────────────────────────────────
y_true      = np.concatenate([y.numpy() for _, y in test_ds], axis=0).flatten()
y_pred_prob = model.predict(test_ds, verbose=1).flatten()
y_pred      = (y_pred_prob > 0.5).astype(int)

# ── Scalar metrics ────────────────────────────────────────────────────────────
accuracy = (y_pred == y_true).mean()
auc      = roc_auc_score(y_true, y_pred_prob)

print("\n" + "═" * 55)
print(f"  Test Accuracy : {accuracy * 100:.2f} %")
print(f"  ROC-AUC       : {auc:.4f}")
print("═" * 55)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Chest X-Ray Pneumonia Classifier — Evaluation", fontsize=14, fontweight="bold")

# Left: Confusion matrix heat-map
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[0],
    linewidths=0.5,
)
axes[0].set_xlabel("Predicted", fontsize=12)
axes[0].set_ylabel("Actual", fontsize=12)
axes[0].set_title("Confusion Matrix")

# Right: ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}", color="#1f77b4")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlabel("False Positive Rate", fontsize=12)
axes[1].set_ylabel("True Positive Rate", fontsize=12)
axes[1].set_title("ROC Curve")
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
cm_path = "../models/confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
print(f"\n✅ Evaluation plots saved → {cm_path}")
plt.show()

# ── Training Curves (if CSV logs are present) ─────────────────────────────────
csv1 = "../models/history_phase1.csv"
csv2 = "../models/history_phase2.csv"

if os.path.exists(csv1) and os.path.exists(csv2):
    h1 = pd.read_csv(csv1)
    h2 = pd.read_csv(csv2)

    # Offset phase-2 epoch indices
    offset = len(h1)
    h2_epochs = h2["epoch"] + offset

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
    fig2.suptitle("Training Curves (Phase 1 → Phase 2)", fontsize=13, fontweight="bold")

    for ax, metric, title in zip(
        axes2,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        train_m, val_m = metric
        ax.plot(h1["epoch"], h1[train_m],     label="Train P1",   color="#1f77b4")
        ax.plot(h1["epoch"], h1[val_m],       label="Val   P1",   color="#1f77b4", linestyle="--")
        ax.plot(h2_epochs,   h2[train_m],     label="Train P2",   color="#ff7f0e")
        ax.plot(h2_epochs,   h2[val_m],       label="Val   P2",   color="#ff7f0e", linestyle="--")
        ax.axvline(x=offset - 0.5, color="gray", linestyle=":", lw=1.5, label="Fine-tune start")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} over Epochs")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    curves_path = "../models/training_curves.png"
    plt.savefig(curves_path, dpi=150)
    print(f"✅ Training curves saved  → {curves_path}")
    plt.show()
