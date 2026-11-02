"""
splits.py
─────────
Patient-level data handling (no TensorFlow / PyTorch imports).

The Kaggle filenames encode the patient:
    person1000_bacteria_2931.jpeg   → person1000
    IM-0115-0001.jpeg               → IM-0115
    NORMAL2-IM-0381-0001.jpeg       → NORMAL2-IM-0381

Review 1 moved random *images* into val2/, so the same child's X-rays
could land in both train and validation. Here train/val are split by
*patient*, and the official 624-image test set is kept untouched so the
results stay comparable with Review 1.
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from edge.trustedge_runtime import preprocess_xray  # noqa: E402

from .config import CLASSES, IMG_SIZE, SEED  # noqa: E402

PATIENT_PATTERNS = (r"^(person\d+)_", r"^(NORMAL2-IM-\d+)-", r"^(IM-\d+)-")
IMAGE_EXTS = {".jpeg", ".jpg", ".png"}


def patient_id(filename: str) -> str:
    for pattern in PATIENT_PATTERNS:
        m = re.match(pattern, filename)
        if m:
            return m.group(1)
    return Path(filename).stem  # unknown naming → treat each image as its own patient


def build_manifest(data_dir) -> pd.DataFrame:
    """One row per image: relpath, label, patient, source folder."""
    data_dir = Path(data_dir)
    rows = []
    for source in ("train", "val", "test"):
        for label, cls in enumerate(CLASSES):
            folder = data_dir / source / cls
            if not folder.exists():
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith("."):
                    rows.append({
                        "relpath": f"{source}/{cls}/{f.name}",
                        "label": label,
                        # test patients are numbered independently of train
                        "patient": f"{source if source == 'test' else 'trainval'}:{patient_id(f.name)}",
                        "source": source,
                    })
    if not rows:
        raise FileNotFoundError(f"No images found under {data_dir}/(train|val|test)/(NORMAL|PNEUMONIA)")
    return pd.DataFrame(rows)


def assign_splits(df: pd.DataFrame, val_frac: float = 0.15, seed: int = SEED) -> pd.DataFrame:
    """train/val split grouped by patient and stratified by label; test = official test."""
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.copy()
    df["split"] = np.where(df["source"] == "test", "test", "train")
    pool = df[df["split"] == "train"]
    n_splits = max(2, round(1.0 / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(pool, pool["label"], groups=pool["patient"]))
    df.loc[pool.index[val_idx], "split"] = "val"
    df["cache_relpath"] = df["relpath"].str.replace(r"\.[^.]+$", ".png", regex=True)
    return df


def check_no_patient_overlap(df: pd.DataFrame) -> int:
    train = set(df.loc[df.split == "train", "patient"])
    val = set(df.loc[df.split == "val", "patient"])
    return len(train & val)


def image_level_leakage(df: pd.DataFrame, val_frac: float = 0.10, seed: int = SEED) -> dict:
    """
    Reproduce Review 1's random image-level split of the official train
    folder and measure how many validation images share a patient with
    the remaining training images.
    """
    rng = np.random.default_rng(seed)
    pool = df[df["source"] == "train"]
    val_mask = np.zeros(len(pool), bool)
    for label in (0, 1):
        idx = np.flatnonzero(pool["label"].values == label)
        val_mask[rng.choice(idx, int(val_frac * len(idx)), replace=False)] = True
    train_patients = set(pool.loc[~val_mask, "patient"])
    leaked = pool.loc[val_mask, "patient"].isin(train_patients)
    return {
        "val_images": int(val_mask.sum()),
        "val_images_sharing_patient_with_train": int(leaked.sum()),
        "leak_fraction": float(leaked.mean()),
    }


def limit_per_split(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Smoke tests: keep at most n images per (split, label)."""
    if not n:
        return df
    keep = []
    for _, group in df.groupby(["split", "label"]):
        keep.extend(group.sample(min(len(group), n), random_state=seed).index)
    return df.loc[sorted(keep)]


def build_cache(df: pd.DataFrame, data_dir, cache_dir, clahe: bool = True,
                size: int = IMG_SIZE, workers: int = 8) -> int:
    """Pre-process every image once (resize + optional CLAHE) to grayscale PNG."""
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)

    def work(row):
        out = cache_dir / row.cache_relpath
        if out.exists():
            return 0
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = preprocess_xray(data_dir / row.relpath, size=size, clahe=clahe)
        from PIL import Image
        Image.fromarray(arr[..., 0]).save(out)
        return 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(work, df.itertuples(index=False)))


def load_splits(path, limit: int = 0, seed: int = SEED) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m trustedge.prepare_data` first.")
    return limit_per_split(pd.read_csv(path), limit, seed)
