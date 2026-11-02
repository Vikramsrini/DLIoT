"""
train.py
────────
Two-phase training for the pneumonia classifier:

  Phase 1 (Epochs 1-15)  — Frozen MobileNetV2 backbone
      Train only the classification head at lr=1e-3.
      Fast convergence; 5-epoch early stopping.

  Phase 2 (Epochs 1-10)  — Fine-tune top-30 backbone layers
      Unfreeze the last 30 MobileNetV2 layers at lr=1e-5.
      Careful fine-tuning to avoid catastrophic forgetting.

Run:
    cd src && python train.py
"""

import numpy as np
import tensorflow as tf

from data  import get_datasets, data_augmentation
from model import build_model

# ── Load data ─────────────────────────────────────────────────────────────────
train_ds, val_ds, test_ds, class_names = get_datasets()
print(f"Classes: {class_names}")

# Apply augmentation to training batches only
train_ds_aug = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)

# ── Class-weight calculation (handles ~3:1 PNEUMONIA:NORMAL imbalance) ────────
train_labels = np.concatenate(
    [y.numpy() for _, y in train_ds], axis=0
).flatten().astype(int)

neg, pos  = np.bincount(train_labels)
total     = neg + pos
class_weight = {
    0: total / (2.0 * neg),   # NORMAL
    1: total / (2.0 * pos),   # PNEUMONIA
}
print(f"Class weights → NORMAL: {class_weight[0]:.3f}  PNEUMONIA: {class_weight[1]:.3f}")

# ── Build model ───────────────────────────────────────────────────────────────
model, base_model = build_model()

METRICS = [
    "accuracy",
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Frozen backbone, train classification head only
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  PHASE 1 — Classification head only (backbone frozen)")
print("═" * 60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=METRICS,
)

callbacks_phase1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "../models/best_phase1.keras", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.CSVLogger("../models/history_phase1.csv"),
]

history1 = model.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weight,
    callbacks=callbacks_phase1,
)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Unfreeze top-30 backbone layers, fine-tune at low LR
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  PHASE 2 — Fine-tuning top MobileNetV2 layers")
print("═" * 60)

base_model.trainable = True

# Keep earlier layers frozen to preserve low-level feature detectors
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Trainable backbone layers: {trainable_count}/{len(base_model.layers)}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=METRICS,
)

callbacks_phase2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "../models/best_final.keras", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.CSVLogger("../models/history_phase2.csv"),
]

history2 = model.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight,
    callbacks=callbacks_phase2,
)

# ── Save final model ──────────────────────────────────────────────────────────
SAVE_PATH = "../models/pneumonia_mobilenetv2_final.keras"
model.save(SAVE_PATH)
print(f"\n✅ Training complete. Final model saved → {SAVE_PATH}")
