"""
model.py
────────
MobileNetV2 student for TrustEdge-CXR.

    image (224×224×3, 0–255)
      → Rescaling to [-1, 1]
      → MobileNetV2 backbone (ImageNet weights)   ← runs on the edge as INT8 TFLite
      → GlobalAveragePooling2D
      → Dropout(0.3) → Dense(128, relu) → Dropout(0.2)
      → Dense(1)  logit                           ← exported to head.npz (numpy)

The model outputs a logit (no sigmoid) so distillation, temperature
scaling and MC dropout all work on the same quantity.
"""

import keras
import numpy as np

from .config import IMG_SIZE

DROPOUT1, DROPOUT2, HIDDEN = 0.3, 0.2, 128


def build_student(weights="imagenet"):
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale")(inputs)
    backbone = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=weights)
    backbone.trainable = False
    # training=False keeps BatchNorm in inference mode even when fine-tuning
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(DROPOUT1, name="head_dropout1")(x)
    x = keras.layers.Dense(HIDDEN, activation="relu", name="head_dense")(x)
    x = keras.layers.Dropout(DROPOUT2, name="head_dropout2")(x)
    logit = keras.layers.Dense(1, name="head_logit")(x)
    return keras.Model(inputs, logit, name="trustedge_student"), backbone


def get_backbone(model) -> keras.Model:
    return next(layer for layer in model.layers if isinstance(layer, keras.Model))


def unfreeze_top(backbone, n_layers: int):
    """Train the last n_layers of the backbone; BatchNorm layers stay frozen."""
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    return sum(1 for layer in backbone.layers if layer.trainable)


def feature_model(model) -> keras.Model:
    """image → (7, 7, 1280) feature map, sharing the trained layers."""
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    fmap = get_backbone(model)(model.get_layer("scale")(inp), training=False)
    return keras.Model(inp, fmap, name="trustedge_backbone")


def head_arrays(model) -> dict:
    w1, b1 = model.get_layer("head_dense").get_weights()
    w2, b2 = model.get_layer("head_logit").get_weights()
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2,
            "dropout1": np.float32(model.get_layer("head_dropout1").rate),
            "dropout2": np.float32(model.get_layer("head_dropout2").rate)}


def save_head(model, path):
    np.savez(path, **head_arrays(model))


def compute_fmaps(fmodel, images, batch_size: int = 64) -> np.ndarray:
    """Feature maps in batches, converting uint8 → float32 one batch at a time."""
    out = [fmodel(np.asarray(images[i:i + batch_size], np.float32), training=False).numpy()
           for i in range(0, len(images), batch_size)]
    return np.concatenate(out)


def load_student(path):
    return keras.models.load_model(path, compile=False)
