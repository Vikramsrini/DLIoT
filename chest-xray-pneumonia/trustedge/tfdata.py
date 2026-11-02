"""
tfdata.py
─────────
tf.data pipelines over the pre-processed PNG cache.

Labels are packed as y = [label, teacher_logit] so one Keras loss can
combine the ground truth with the teacher's soft target (the teacher
column is 0 when distillation is off).
"""

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .config import IMG_SIZE, SEED


def augmenter() -> keras.Sequential:
    # No horizontal flip: chest anatomy is not left-right symmetric.
    return keras.Sequential([
        keras.layers.RandomRotation(0.05, fill_mode="constant"),
        keras.layers.RandomZoom(0.10, fill_mode="constant"),
        keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant"),
        keras.layers.RandomContrast(0.10),
    ], name="augmentation")


def _targets(df, teacher_logits=None) -> np.ndarray:
    label = df["label"].to_numpy(np.float32)
    if teacher_logits is None:
        teacher = np.zeros_like(label)
    else:
        teacher = df["relpath"].map(teacher_logits).to_numpy(np.float32)
        if np.isnan(teacher).any():
            raise ValueError("Missing teacher logits for some images — rerun trustedge.teacher.")
    return np.stack([label, teacher], axis=1)


def _decode(path):
    img = tf.io.decode_png(tf.io.read_file(path), channels=1)
    img = tf.image.grayscale_to_rgb(img)
    img = tf.cast(img, tf.float32)
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


def make_dataset(df, cache_dir, batch_size=32, training=False,
                 teacher_logits=None, seed=SEED) -> tf.data.Dataset:
    paths = [str(Path(cache_dir) / p) for p in df["cache_relpath"]]
    ds = tf.data.Dataset.from_tensor_slices((paths, _targets(df, teacher_logits)))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_decode(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if training:
        aug = augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_images(df, cache_dir) -> np.ndarray:
    """All images of df as a (N, 224, 224, 3) uint8 array, in df order."""
    from PIL import Image
    out = np.empty((len(df), IMG_SIZE, IMG_SIZE, 3), np.uint8)
    for i, rel in enumerate(df["cache_relpath"]):
        gray = np.asarray(Image.open(Path(cache_dir) / rel), np.uint8)
        out[i] = gray[..., None]
    return out


def load_raw_images(df, data_dir) -> np.ndarray:
    """Review 1 preprocessing: RGB decode + bilinear resize, no CLAHE."""
    def load(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    paths = [str(Path(data_dir) / p) for p in df["relpath"]]
    ds = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(64)
    return np.concatenate([b.numpy() for b in ds]).astype(np.float32)
