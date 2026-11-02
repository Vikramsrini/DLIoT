"""
config.py
─────────
Shared paths and constants. Every step accepts --data-dir and --artifacts,
and the environment variables TRUSTEDGE_DATA / TRUSTEDGE_ARTIFACTS change
the defaults (used by the Colab notebook).
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TRUSTEDGE_DATA", ROOT / "data" / "chest_xray"))
ARTIFACTS = Path(os.environ.get("TRUSTEDGE_ARTIFACTS", ROOT / "artifacts"))
BASELINE_MODEL = ROOT / "models" / "pneumonia_mobilenetv2_final.keras"

IMG_SIZE = 224
SEED = 42
CLASSES = ("NORMAL", "PNEUMONIA")


class Paths:
    """Every file the pipeline reads or writes, under one artifacts folder."""

    def __init__(self, artifacts=ARTIFACTS):
        a = Path(artifacts)
        self.root = a
        self.splits = a / "splits.csv"
        self.cache_clahe = a / "cache" / "clahe224"
        self.cache_raw = a / "cache" / "raw224"
        self.teacher_dir = a / "teacher"
        self.teacher_embeddings = a / "teacher" / "embeddings.npz"
        self.teacher_logits = a / "teacher" / "teacher_logits.csv"
        self.teacher_metrics = a / "teacher" / "teacher_metrics.json"
        self.student_dir = a / "student"
        self.student_model = a / "student" / "student.keras"
        self.head = a / "student" / "head.npz"
        self.calibration = a / "student" / "calibration.json"
        self.results = a / "results"
        self.edge_bundle = a / "edge_bundle"
        self.federated = a / "federated"
        self.rsna = a / "rsna"
        self.rsna_manifest = a / "rsna" / "manifest.csv"

    def ensure(self, *dirs):
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Kaggle chest_xray folder (train/ val/ test/)")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS,
                        help="where splits, models and results are written")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke test: use at most N images per split and class")
    return parser


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Only seed frameworks the step already imported (the PyTorch teacher
    # step must not pull TensorFlow onto the GPU).
    if "tensorflow" in sys.modules:
        sys.modules["tensorflow"].random.set_seed(seed)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


def banner(title: str):
    print("\n" + "═" * 64)
    print(f"  {title}")
    print("═" * 64)
