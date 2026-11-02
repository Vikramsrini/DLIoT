"""
teacher.py — Step 2 (PyTorch; a GPU makes this ~2 min instead of ~30)
──────────────────────────────────────────────────────────────────────
RAD-DINO teacher (Pérez-García et al., Nat. Mach. Intell. 2025): a
ViT-B/14 trained with DINOv2 self-supervision on 882k chest X-rays.

  1. Embed every image as [CLS ‖ mean patch token] (1536-d, cached).
  2. Fit two heads on the patient-level train split:
       • linear probe  — logistic regression (the "RAD-DINO linear probe"
                         baseline from the Review 2 deck)
       • MLP head      — 1536 → 256 → 1, class-weighted BCE
     and keep whichever has the better validation AUC as the teacher.
  3. Teacher logits for distillation:
       train rows → out-of-fold predictions (5 patient-grouped folds), so
                    the student never sees soft labels the head memorised
       val / test → head fitted on all train rows

Run:
    python -m trustedge.teacher
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Paths, add_common_args, banner, seed_everything
from .metrics import binary_metrics, pick_threshold, sigmoid
from .splits import load_splits


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── 1 · Embeddings ────────────────────────────────────────────────────────────
class _Images(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return Image.open(self.paths[i]).convert("RGB")


def embed(df, data_dir, model_name, batch_size, device, workers) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    loader = torch.utils.data.DataLoader(
        _Images([str(Path(data_dir) / p) for p in df["relpath"]]),
        batch_size=batch_size, num_workers=workers, collate_fn=list)

    feats = []
    for i, images in enumerate(loader):
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            hidden = model(**inputs).last_hidden_state          # (B, 1 + patches, 768)
        emb = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)
        feats.append(emb.float().cpu().numpy())
        if i % 20 == 0:
            print(f"  embedded {min((i + 1) * batch_size, len(df))}/{len(df)}")
    return np.concatenate(feats)


def load_or_embed(df, args, P) -> np.ndarray:
    cache = P.teacher_embeddings
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        index = {p: i for i, p in enumerate(stored["relpath"])}
        if all(p in index for p in df["relpath"]):
            print(f"Using cached embeddings ({cache})")
            return stored["emb"][[index[p] for p in df["relpath"]]]
    device = pick_device(args.device)
    print(f"Embedding {len(df)} images with {args.model} on {device} …")
    emb = embed(df, args.data_dir, args.model, args.batch_size, device, args.workers)
    np.savez(cache, relpath=df["relpath"].to_numpy(), emb=emb)
    return emb


# ── 2 · Heads ─────────────────────────────────────────────────────────────────
class LinearProbe:
    def __init__(self, C):
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, class_weight="balanced", max_iter=5000))

    def fit(self, X, y, X_val=None, y_val=None):
        self.pipe.fit(X, y)
        return self

    def logits(self, X):
        return self.pipe.decision_function(X)


class MLPHead:
    def __init__(self, seed, hidden=256, epochs=300, patience=30):
        self.seed, self.hidden, self.epochs, self.patience = seed, hidden, epochs, patience

    def fit(self, X, y, X_val, y_val):
        torch.manual_seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt, yt = self._t(X), torch.tensor(y, dtype=torch.float32)
        Xv, yv = self._t(X_val), torch.tensor(y_val, dtype=torch.float32)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.hidden), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(self.hidden, 1))
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-2)
        pos_weight = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        gen = torch.Generator().manual_seed(self.seed)

        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            for idx in torch.randperm(len(Xt), generator=gen).split(256):
                opt.zero_grad()
                loss_fn(self.net(Xt[idx])[:, 0], yt[idx]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(Xv)[:, 0], yv).item()
            if val_loss < best - 1e-4:
                best, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def _t(self, X):
        return torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)

    def logits(self, X):
        self.net.eval()
        with torch.no_grad():
            return self.net(self._t(X))[:, 0].numpy()


def evaluate_head(name, logits_val, y_val, logits_test, y_test) -> dict:
    p_val, p_test = sigmoid(logits_val), sigmoid(logits_test)
    t = pick_threshold(y_val, p_val)
    out = {"val": binary_metrics(y_val, p_val, 0.5),
           "test@0.5": binary_metrics(y_test, p_test, 0.5),
           "test@tuned": binary_metrics(y_test, p_test, t)}
    print(f"  {name:13s} val AUC {out['val']['auc']:.4f} │ test AUC {out['test@0.5']['auc']:.4f} │ "
          f"test acc {out['test@tuned']['accuracy']:.3f} sens {out['test@tuned']['sensitivity']:.3f} "
          f"spec {out['test@tuned']['specificity']:.3f} (threshold {t:.3f})")
    return out


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--model", default="microsoft/rad-dino")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    P = Paths(args.artifacts)
    P.ensure(P.teacher_dir)

    banner("STEP 2 — RAD-DINO teacher")
    df = load_splits(P.splits, args.limit, args.seed).reset_index(drop=True)
    X = load_or_embed(df, args, P)
    y = df["label"].to_numpy()
    tr, va, te = (df["split"] == s for s in ("train", "val", "test"))

    # Linear probe: choose C on validation
    best_c, best_auc = None, -1.0
    for C in (0.001, 0.01, 0.1, 1.0):
        auc = binary_metrics(y[va], sigmoid(LinearProbe(C).fit(X[tr], y[tr]).logits(X[va])))["auc"]
        if auc > best_auc:
            best_c, best_auc = C, auc
    factories = {
        "linear_probe": lambda: LinearProbe(best_c),
        "mlp_head": lambda: MLPHead(args.seed),
    }

    print("\nHeads trained on the patient-level train split:")
    report, full_logits = {}, {}
    for name, make in factories.items():
        head = make().fit(X[tr], y[tr], X[va], y[va])
        full_logits[name] = head.logits(X)
        report[name] = evaluate_head(name, full_logits[name][va], y[va], full_logits[name][te], y[te])
    report["linear_probe"]["C"] = best_c
    selected = max(report, key=lambda k: report[k]["val"]["auc"])
    report["selected_teacher"] = selected
    print(f"\nTeacher = {selected}")

    # Out-of-fold teacher logits for the training rows
    logits = full_logits[selected].copy()
    train_idx = np.flatnonzero(tr)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for k, (fit_i, hold_i) in enumerate(folds.split(train_idx, y[train_idx], df["patient"].to_numpy()[train_idx])):
        head = factories[selected]().fit(X[train_idx[fit_i]], y[train_idx[fit_i]], X[va], y[va])
        logits[train_idx[hold_i]] = head.logits(X[train_idx[hold_i]])
        print(f"  out-of-fold logits: fold {k + 1}/5")
    oof_auc = binary_metrics(y[tr], sigmoid(logits[tr]))["auc"]
    report["train_oof_auc"] = oof_auc
    print(f"  train out-of-fold AUC {oof_auc:.4f}")

    out = df[["relpath", "split", "label"]].assign(teacher_logit=logits)
    out.to_csv(P.teacher_logits, index=False)
    P.teacher_metrics.write_text(json.dumps(report, indent=2))
    print(f"\n✅ teacher logits → {P.teacher_logits}")


if __name__ == "__main__":
    main()
