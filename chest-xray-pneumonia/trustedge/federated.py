"""
federated.py — Step 8
─────────────────────
Simulates K hospitals training the student together without sharing a
single image (gap G5). Follows Pan et al. (Sci. Rep. 2024):

  partition  training patients spread over K sites with a Dirichlet(α)
             label skew, so sites differ the way real hospitals do
  client     FedProx-style penalty  μ/2 · ‖w − w_global‖²  on the local loss
  server     FedAvgM: momentum on the averaged update
  privacy    optional DP-FedAvg: each site's update is clipped to norm C
             and Gaussian noise σ·C/K is added to the average; ε is
             reported when the `dp-accounting` package is installed

Only trainable weights travel; images never leave a site.

Run:
    python -m trustedge.federated                       # no DP
    python -m trustedge.federated --dp-clip 1 --dp-noise 0.5 --out federated_dp
"""

import argparse
import json

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Paths, add_common_args, banner, seed_everything
from .losses import DistillationLoss
from .metrics import binary_metrics, fit_temperature, pick_threshold, sigmoid
from .model import build_student, save_head, unfreeze_top
from .splits import load_splits
from .tfdata import make_dataset


def partition(train_df, k, alpha, seed):
    """Patients → sites with Dirichlet label skew (each patient has one label)."""
    rng = np.random.default_rng(seed)
    patient_label = train_df.groupby("patient")["label"].first()
    sites = [[] for _ in range(k)]
    for label in (0, 1):
        pids = patient_label[patient_label == label].index.to_numpy()
        rng.shuffle(pids)
        cuts = (np.cumsum(rng.dirichlet([alpha] * k)) * len(pids)).astype(int)[:-1]
        for s, chunk in enumerate(np.split(pids, cuts)):
            sites[s].extend(chunk)
    return [train_df[train_df.patient.isin(set(s))] for s in sites]


def make_client_step(model, loss_fn, optimizer, global_vars, mu):
    @tf.function
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = loss_fn(y, model(x, training=True))
            if mu > 0:
                loss += 0.5 * mu * tf.add_n([tf.reduce_sum(tf.square(v - g))
                                             for v, g in zip(model.trainable_variables, global_vars)])
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    return step


def set_weights(variables, values):
    for v, w in zip(variables, values):
        v.assign(w)


def predict_probs(model, ds):
    logits = np.concatenate([model(x, training=False).numpy()[:, 0] for x, _ in ds])
    labels = np.concatenate([y.numpy()[:, 0] for _, y in ds]).astype(int)
    return logits, labels


def dp_epsilon(noise, rounds, delta):
    try:
        import dp_accounting
    except ImportError:
        return None
    acc = dp_accounting.rdp.RdpAccountant()
    acc.compose(dp_accounting.GaussianDpEvent(noise), rounds)
    return float(acc.get_epsilon(delta))


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx penalty")
    parser.add_argument("--server-momentum", type=float, default=0.5)
    parser.add_argument("--server-lr", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--unfreeze", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--kd", action="store_true", help="distil from the teacher at every site")
    parser.add_argument("--dp-clip", type=float, default=0.0, help="update norm C (0 = DP off)")
    parser.add_argument("--dp-noise", type=float, default=0.0, help="noise multiplier σ")
    parser.add_argument("--dp-delta", type=float, default=1e-3)
    parser.add_argument("--out", default="federated")
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    P = Paths(args.artifacts)
    out_dir = P.root / args.out
    P.ensure(out_dir)
    use_dp = args.dp_clip > 0

    banner(f"STEP 8 — Federated learning ({args.clients} sites, {args.rounds} rounds, "
           f"{'DP on' if use_dp else 'no DP'})")
    df = load_splits(P.splits, args.limit, args.seed)
    teacher = None
    if args.kd:
        t = pd.read_csv(P.teacher_logits)
        teacher = dict(zip(t["relpath"], t["teacher_logit"]))
    train_df, val_df, test_df = (df[df.split == s] for s in ("train", "val", "test"))
    sites = [s for s in partition(train_df, args.clients, args.dirichlet_alpha, args.seed) if len(s)]
    for i, s in enumerate(sites):
        print(f"  site {i + 1}: {len(s):4d} images · {s.label.mean():.0%} pneumonia")

    alpha_pos = float((train_df.label == 0).mean())
    loss_fn = DistillationLoss(alpha=0.5 if args.kd else 1.0, alpha_pos=alpha_pos)
    model, backbone = build_student()
    unfreeze_top(backbone, args.unfreeze)
    tvars = model.trainable_variables
    global_vars = [tf.Variable(v, trainable=False) for v in tvars]

    site_ds = [make_dataset(s, P.cache_clahe, args.batch_size, True, teacher, args.seed + i)
               for i, s in enumerate(sites)]
    steps = [make_client_step(model, loss_fn, keras.optimizers.Adam(args.lr), global_vars, args.mu)
             for _ in sites]
    val_ds = make_dataset(val_df, P.cache_clahe, 64, False, teacher)
    test_ds = make_dataset(test_df, P.cache_clahe, 64, False, teacher)

    w_global = [v.numpy() for v in tvars]
    velocity = [np.zeros_like(w) for w in w_global]
    rng = np.random.default_rng(args.seed)
    history, best_auc, best_w = [], -1.0, w_global
    for r in range(1, args.rounds + 1):
        deltas, sizes, losses = [], [], []
        for k, (ds, step) in enumerate(zip(site_ds, steps)):
            set_weights(tvars, w_global)
            set_weights(global_vars, w_global)
            for _ in range(args.local_epochs):
                for x, y in ds:
                    losses.append(float(step(x, y)))
            delta = [v.numpy() - w for v, w in zip(tvars, w_global)]
            if use_dp:
                norm = np.sqrt(sum(float((d ** 2).sum()) for d in delta))
                delta = [d * min(1.0, args.dp_clip / (norm + 1e-12)) for d in delta]
            deltas.append(delta)
            sizes.append(len(sites[k]))

        if use_dp:  # equal weights bound each site's influence to C / K
            avg = [np.mean(ds_, axis=0) + rng.normal(0, args.dp_noise * args.dp_clip / len(deltas), ds_[0].shape)
                   for ds_ in zip(*deltas)]
        else:
            wts = np.array(sizes, float) / sum(sizes)
            avg = [sum(w * d for w, d in zip(wts, ds_)) for ds_ in zip(*deltas)]
        velocity = [args.server_momentum * v + a for v, a in zip(velocity, avg)]
        w_global = [w + args.server_lr * v for w, v in zip(w_global, velocity)]

        set_weights(tvars, w_global)
        logits, labels = predict_probs(model, val_ds)
        auc = binary_metrics(labels, sigmoid(logits))["auc"]
        history.append({"round": r, "train_loss": float(np.mean(losses)), "val_auc": auc})
        print(f"  round {r:2d}/{args.rounds} · mean local loss {np.mean(losses):.4f} · val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, best_w = auc, [w.copy() for w in w_global]

    set_weights(tvars, best_w)
    val_logits, val_y = predict_probs(model, val_ds)
    T = fit_temperature(val_y, val_logits)
    threshold = pick_threshold(val_y, sigmoid(val_logits / T))
    test_logits, test_y = predict_probs(model, test_ds)
    report = {
        "config": {k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v)
                   for k, v in vars(args).items()},
        "site_sizes": sizes, "best_val_auc": best_auc, "temperature": T,
        "test@0.5": binary_metrics(test_y, sigmoid(test_logits), 0.5),
        "test@tuned": binary_metrics(test_y, sigmoid(test_logits / T), threshold),
    }
    if use_dp:
        report["dp"] = {"clip": args.dp_clip, "noise_multiplier": args.dp_noise,
                        "delta": args.dp_delta,
                        "epsilon": dp_epsilon(args.dp_noise, args.rounds, args.dp_delta)}
    pd.DataFrame(history).to_csv(out_dir / "fl_history.csv", index=False)
    (out_dir / "fl_metrics.json").write_text(json.dumps(report, indent=2))
    model.save(out_dir / "fl_model.keras")
    save_head(model, out_dir / "head.npz")

    m = report["test@tuned"]
    print(f"\n  test acc {m['accuracy']:.4f} · sens {m['sensitivity']:.4f} · spec {m['specificity']:.4f} "
          f"· AUC {m['auc']:.4f}")
    if use_dp:
        eps = report["dp"]["epsilon"]
        print(f"  DP: clip {args.dp_clip}, σ {args.dp_noise} → ε = "
              f"{'n/a (pip install dp-accounting)' if eps is None else f'{eps:.1f}'} at δ = {args.dp_delta}")
    print(f"\n✅ federated model → {out_dir}")


if __name__ == "__main__":
    main()
