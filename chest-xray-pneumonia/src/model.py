"""
model.py
────────
MobileNetV2 transfer-learning model for binary chest X-ray classification.

Architecture overview:
    Input (224×224×3)
        → MobileNetV2 preprocessing
        → MobileNetV2 backbone (frozen in Phase 1)
        → GlobalAveragePooling2D
        → Dropout(0.3)
        → Dense(128, relu)
        → Dropout(0.2)
        → Dense(1, sigmoid)          ← 0 = NORMAL, 1 = PNEUMONIA

Usage:
    model, base_model = build_model()
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model

IMG_SIZE = (224, 224)


def build_model():
    """
    Build and return:
        model      – full Keras Model (ready to compile)
        base_model – MobileNetV2 backbone (expose for fine-tuning)
    """
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Phase 1: frozen backbone

    # ── Functional API ────────────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image_input")

    # Scale [0, 255] → [-1, 1] as expected by MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Backbone (training=False keeps BatchNorm layers in inference mode even
    # when the backbone is unfrozen during fine-tuning)
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2, name="dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="pneumonia_classifier")
    return model, base_model


def model_summary():
    """Pretty-print the model architecture."""
    model, _ = build_model()
    model.summary(line_length=90)


if __name__ == "__main__":
    model_summary()
