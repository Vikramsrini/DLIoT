"""
data.py
───────
TensorFlow data pipeline for the chest X-ray pneumonia dataset.

- Loads train / val2 / test splits from disk.
- Returns prefetched tf.data.Dataset objects ready for training.
- Exports a data_augmentation Sequential model used in train.py.
"""

import tensorflow as tf

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
DATA_DIR   = "../data/chest_xray"


def get_datasets():
    """
    Returns:
        train_ds   – prefetched training dataset
        val_ds     – prefetched validation dataset  (val2/, re-split)
        test_ds    – prefetched test dataset        (no shuffle, ordered)
        class_names – list e.g. ['NORMAL', 'PNEUMONIA']
    """
    _load = lambda split, shuffle=True: tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/{split}",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=42,
        shuffle=shuffle,
    )

    train_ds = _load("train")
    val_ds   = _load("val2")
    test_ds  = _load("test", shuffle=False)

    class_names = train_ds.class_names  # ['NORMAL', 'PNEUMONIA']

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# ── Augmentation (applied to training batches only) ───────────────────────────
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="data_augmentation",
)
