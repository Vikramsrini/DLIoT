"""
rsna.py — optional external test set
────────────────────────────────────
Prepares a balanced subset of the RSNA Pneumonia Detection Challenge
(adult chest X-rays; Shih et al., Radiol. Artif. Intell. 2019) as a
cross-population test for a model trained on children:

  positives  "Lung Opacity" images, with radiologist bounding boxes
  negatives  "Normal" images
  ("No Lung Opacity / Not Normal" is left out to mirror the binary task)

Reads straight from the competition zip (or an extracted folder) and writes
224-px PNGs with and without CLAHE, plus manifest.csv with boxes in
224-px coordinates.

Download (accept the competition rules on kaggle.com first):
    kaggle competitions download -c rsna-pneumonia-detection-challenge
Run:
    python -m trustedge.rsna --source rsna-pneumonia-detection-challenge.zip
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import ROOT, Paths, add_common_args, banner

sys.path.insert(0, str(ROOT))
from edge.trustedge_runtime import IMG_SIZE, preprocess_xray  # noqa: E402


class Source:
    """Uniform reader for the competition zip or an extracted folder."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path) if self.path.suffix == ".zip" else None

    def read(self, name: str) -> bytes:
        return self.zip.read(name) if self.zip else (self.path / name).read_bytes()


def dicom_to_uint8(raw: bytes) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(raw))
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = arr.min(), arr.max()
    return np.uint8(255 * (arr - lo) / max(hi - lo, 1e-6))


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--source", type=Path, required=True,
                        help="competition zip or extracted folder")
    parser.add_argument("--per-class", type=int, default=1000)
    args = parser.parse_args(argv)
    P = Paths(args.artifacts)
    banner("RSNA external test set")

    src = Source(args.source)
    labels = pd.read_csv(io.BytesIO(src.read("stage_2_train_labels.csv")))
    classes = pd.read_csv(io.BytesIO(src.read("stage_2_detailed_class_info.csv"))).drop_duplicates("patientId")
    boxes = (labels[labels.Target == 1].groupby("patientId")
             .apply(lambda g: g[["x", "y", "width", "height"]].to_numpy().tolist()))
    classes = classes.set_index("patientId")["class"]

    per_class = args.per_class if not args.limit else args.limit
    pos = classes[classes == "Lung Opacity"].sample(per_class, random_state=args.seed).index
    neg = classes[classes == "Normal"].sample(per_class, random_state=args.seed).index

    rows = []
    for label, ids in ((1, pos), (0, neg)):
        for i, pid in enumerate(ids):
            gray = dicom_to_uint8(src.read(f"stage_2_train_images/{pid}.dcm"))
            h, w = gray.shape
            raw_rel, clahe_rel = f"raw224/{pid}.png", f"clahe224/{pid}.png"
            for rel in (raw_rel, clahe_rel):
                (P.rsna / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(gray).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).save(P.rsna / raw_rel)
            Image.fromarray(preprocess_xray(gray)[..., 0]).save(P.rsna / clahe_rel)
            scaled = None
            if label == 1 and pid in boxes.index:
                sx, sy = IMG_SIZE / w, IMG_SIZE / h
                scaled = json.dumps([[x * sx, y * sy, bw * sx, bh * sy] for x, y, bw, bh in boxes[pid]])
            rows.append({"patientId": pid, "label": label, "raw_relpath": raw_rel,
                         "clahe_relpath": clahe_rel, "boxes": scaled})
            if i % 200 == 0:
                print(f"  {'pneumonia' if label else 'normal'}: {i}/{len(ids)}")

    pd.DataFrame(rows).to_csv(P.rsna_manifest, index=False)
    print(f"\n✅ {len(rows)} RSNA images → {P.rsna_manifest}")


if __name__ == "__main__":
    main()
