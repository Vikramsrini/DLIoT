"""
resplit.py
──────────
Fix the official Kaggle val/ split (only 16 images) by moving 10 % of the
training data into a new val2/ directory.

Run ONCE before any training:
    cd src && python resplit.py
"""

import os
import shutil
import random

random.seed(42)

BASE = "../data/chest_xray"
CLASSES = ["NORMAL", "PNEUMONIA"]
VAL_FRACTION = 0.10  # 10 % of train → new validation set


def main():
    for cls in CLASSES:
        src_dir = os.path.join(BASE, "train", cls)
        dst_dir = os.path.join(BASE, "val2", cls)
        os.makedirs(dst_dir, exist_ok=True)

        files = [f for f in os.listdir(src_dir) if not f.startswith(".")]
        random.shuffle(files)
        n_val = int(VAL_FRACTION * len(files))

        for f in files[:n_val]:
            shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

        print(f"[{cls}] Moved {n_val}/{len(files)} images → val2/{cls}/")

    print("\n✅ Done. New validation split created at data/chest_xray/val2/")


if __name__ == "__main__":
    main()
