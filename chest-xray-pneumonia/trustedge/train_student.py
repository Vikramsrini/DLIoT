"""
train_student.py — Step 3
─────────────────────────
Trains the MobileNetV2 student with focal loss + distillation from the
RAD-DINO teacher, on CLAHE images and the patient-level split.

  Phase 1  backbone frozen, head only          lr 1e-3
  Phase 2  top-40 backbone layers unfrozen     lr 5e-5  (BatchNorm frozen)

Both phases keep the weights with the best validation AUC.

Run:
    python -m trustedge.train_student            # with distillation
    python -m trustedge.train_student --no-kd    # ablation: student alone
"""

import argparse
import json

import keras
import pandas as pd

from .config import Paths, add_common_args, banner, seed_everything
from .losses import DistillationLoss, LabelAccuracy, LabelAUC
from .model import build_student, save_head, unfreeze_top
from .splits import load_splits
from .tfdata import make_dataset


def compile_model(model, lr, loss):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss,
                  metrics=[LabelAUC(name="auc"), LabelAccuracy(name="accuracy")])


def run_phase(model, train_ds, val_ds, epochs, out_dir, tag, patience):
    ckpt = out_dir / f"best_{tag}.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_auc", mode="max",
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                                          patience=2, min_lr=1e-6),
        keras.callbacks.CSVLogger(out_dir / f"history_{tag}.csv"),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)
    model.load_weights(ckpt)


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--no-kd", action="store_true", help="train without the teacher")
    parser.add_argument("--kd-alpha", type=float, default=0.5, help="weight of the focal term")
    parser.add_argument("--kd-temperature", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-finetune", type=int, default=15)
    parser.add_argument("--unfreeze", type=int, default=40)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache", choices=["clahe", "raw"], default="clahe")
    parser.add_argument("--out", default="student", help="sub-folder of artifacts/")
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    P = Paths(args.artifacts)
    out_dir = P.root / args.out
    P.ensure(out_dir)
    cache_dir = P.cache_clahe if args.cache == "clahe" else P.cache_raw

    use_kd = not args.no_kd
    banner(f"STEP 3 — Student training ({'with' if use_kd else 'without'} distillation)")
    df = load_splits(P.splits, args.limit, args.seed)
    teacher = None
    if use_kd:
        if not P.teacher_logits.exists():
            raise FileNotFoundError("Teacher logits missing — run `python -m trustedge.teacher` "
                                    "or pass --no-kd.")
        t = pd.read_csv(P.teacher_logits)
        teacher = dict(zip(t["relpath"], t["teacher_logit"]))

    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    train_ds = make_dataset(train_df, cache_dir, args.batch_size, True, teacher, args.seed)
    val_ds = make_dataset(val_df, cache_dir, args.batch_size, False, teacher)

    # Up-weight the minority NORMAL class inside the focal loss
    alpha_pos = float((train_df.label == 0).mean())
    loss = DistillationLoss(alpha=args.kd_alpha if use_kd else 1.0,
                            temperature=args.kd_temperature, gamma=args.gamma, alpha_pos=alpha_pos)
    print(f"train {len(train_df)} images · val {len(val_df)} images · focal alpha_pos {alpha_pos:.3f}")

    model, backbone = build_student()

    banner("Phase 1 — head only (backbone frozen)")
    compile_model(model, args.lr_head, loss)
    run_phase(model, train_ds, val_ds, args.epochs_head, out_dir, "phase1", patience=3)

    banner(f"Phase 2 — fine-tune top {args.unfreeze} backbone layers")
    n_trainable = unfreeze_top(backbone, args.unfreeze)
    print(f"Trainable backbone layers: {n_trainable}/{len(backbone.layers)}")
    compile_model(model, args.lr_finetune, loss)
    run_phase(model, train_ds, val_ds, args.epochs_finetune, out_dir, "phase2", patience=5)

    # Keep phase 1 if fine-tuning never beat it on validation
    best = {tag: pd.read_csv(out_dir / f"history_{tag}.csv")["val_auc"].max() for tag in ("phase1", "phase2")}
    if best["phase1"] > best["phase2"]:
        print(f"Fine-tuning did not improve val AUC ({best['phase2']:.4f} < {best['phase1']:.4f}); keeping phase 1.")
        model.load_weights(out_dir / "best_phase1.weights.h5")

    model.save(out_dir / "student.keras")
    save_head(model, out_dir / "head.npz")
    config = {k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v)
              for k, v in vars(args).items()}
    (out_dir / "train_config.json").write_text(json.dumps(
        {**config, "kd": use_kd, "alpha_pos": alpha_pos,
         "n_train": int(len(train_df)), "n_val": int(len(val_df))}, indent=2))
    print(f"\n✅ student → {out_dir / 'student.keras'}")


if __name__ == "__main__":
    main()
