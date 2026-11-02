"""
metrics.py
──────────
Classification, calibration and referral metrics (numpy + scikit-learn).
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def ece_score(y, prob, n_bins: int = 15) -> float:
    """Expected calibration error of P(pneumonia), equal-width bins."""
    y, prob = np.asarray(y, float), np.asarray(prob, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - prob[m].mean())
    return float(ece)


def binary_metrics(y, prob, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, float)
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "n": int(len(y)), "threshold": float(threshold),
        "accuracy": (tp + tn) / max(len(y), 1),
        "sensitivity": sens, "specificity": spec, "precision": prec,
        "f1": 2 * prec * sens / max(prec + sens, 1e-12),
        "auc": float(roc_auc_score(y, prob)) if len(set(y)) == 2 else float("nan"),
        "ece": ece_score(y, prob),
        "brier": float(np.mean((prob - y) ** 2)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def pick_threshold(y, prob, min_sensitivity: float = 0.95) -> float:
    """Highest-specificity threshold whose sensitivity is still ≥ min_sensitivity."""
    y, prob = np.asarray(y).astype(int), np.asarray(prob, float)
    best_t, best_spec = 0.0, -1.0
    for t in np.unique(prob):
        pred = prob >= t
        sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        if sens < min_sensitivity:
            continue
        spec = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        if spec > best_spec or (spec == best_spec and t > best_t):
            best_t, best_spec = float(t), spec
    return best_t


def fit_temperature(y, logits) -> float:
    """Temperature T minimising the validation negative log-likelihood."""
    y, logits = np.asarray(y, float), np.asarray(logits, float)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    nll = []
    for T in grid:
        p = np.clip(sigmoid(logits / T), 1e-7, 1 - 1e-7)
        nll.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(grid[int(np.argmin(nll))])


def referral_metrics(y, prob, threshold, refer) -> dict:
    """How the refer-if-unsure rule changes accuracy on the cases kept."""
    y, prob, refer = np.asarray(y).astype(int), np.asarray(prob, float), np.asarray(refer, bool)
    pred = (prob >= threshold).astype(int)
    kept = ~refer
    out = {"referred": int(refer.sum()), "referral_rate": float(refer.mean()),
           "coverage": float(kept.mean())}
    if kept.any():
        k = binary_metrics(y[kept], prob[kept], threshold)
        out.update({f"kept_{m}": k[m] for m in ("accuracy", "sensitivity", "specificity")})
    if refer.any():
        out["referred_accuracy_if_auto"] = float((pred[refer] == y[refer]).mean())
    return out
