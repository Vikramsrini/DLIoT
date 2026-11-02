"""
prepare_data.py — Step 1
────────────────────────
• Builds splits.csv: patient-level train/val split, official test kept.
• Measures how much Review 1's image-level split leaked patients.
• Caches every image once as 224×224 CLAHE PNG (and optionally without
  CLAHE for the ablation).

Run:
    python -m trustedge.prepare_data
"""

import argparse
import json

from .config import Paths, add_common_args, banner
from .splits import (assign_splits, build_cache, build_manifest,
                     check_no_patient_overlap, image_level_leakage)


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--also-raw", action="store_true",
                        help="also cache images without CLAHE (ablation)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    P = Paths(args.artifacts)
    P.ensure(P.root, P.results)

    banner("STEP 1 — Patient-level splits + preprocessing cache")
    df = assign_splits(build_manifest(args.data_dir), args.val_frac, args.seed)
    df.to_csv(P.splits, index=False)

    summary = (df.groupby(["split", "label"])
                 .agg(images=("relpath", "size"), patients=("patient", "nunique")))
    print(summary.to_string())
    overlap = check_no_patient_overlap(df)
    print(f"\nPatients shared by train and val: {overlap}  (must be 0)")
    assert overlap == 0

    leak = image_level_leakage(df, seed=args.seed)
    print(f"Review 1 style image-level split: {leak['val_images_sharing_patient_with_train']}"
          f"/{leak['val_images']} val images ({leak['leak_fraction']:.1%}) share a patient with train")
    (P.results / "leakage.json").write_text(json.dumps(leak, indent=2))

    print("\nCaching CLAHE images …")
    n = build_cache(df, args.data_dir, P.cache_clahe, clahe=True, workers=args.workers)
    print(f"  wrote {n} new files → {P.cache_clahe}")
    if args.also_raw:
        print("Caching images without CLAHE …")
        n = build_cache(df, args.data_dir, P.cache_raw, clahe=False, workers=args.workers)
        print(f"  wrote {n} new files → {P.cache_raw}")
    print(f"\n✅ splits → {P.splits}")


if __name__ == "__main__":
    main()
