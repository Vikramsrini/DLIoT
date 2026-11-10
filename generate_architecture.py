#!/usr/bin/env python3
"""Regenerates Figure 3.2 — a clean, easy-to-read TrustEdge-CXR architecture diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/Users/markiv/Desktop/DLIoT/chest-xray-pneumonia/artifacts/results/system_architecture.png"

fig, ax = plt.subplots(figsize=(11.5, 7.1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")

TITLE_FS = 11
BODY_FS = 10.5
ARROW_FS = 9.5

def box(x, y, w, h, title, lines, face, edge, title_fs=TITLE_FS, body_fs=BODY_FS):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.6,rounding_size=1.4",
                                linewidth=2.0, edgecolor=edge, facecolor=face, zorder=2))
    if lines:
        ax.text(x + w / 2, y + h - 3.6, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=edge, zorder=3)
        ax.text(x + w / 2, y + (h - 6.6) / 2, "\n".join(lines),
                ha="center", va="center", fontsize=body_fs, color="#222222",
                linespacing=1.55, zorder=3)
    else:  # small single-label box: centre the text
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=edge, zorder=3,
                linespacing=1.3)

def arrow(x1, y1, x2, y2, label=None, lx=0, ly=1.8, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=24,
                                 linewidth=2.4, color="#333333",
                                 linestyle="--" if dashed else "-", zorder=4))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                ha="center", va="center", fontsize=ARROW_FS, style="italic",
                color="#333333", zorder=5)

# ── Top row: training pipeline (1 → 2 → 3 → 4) ─────────────────────────────
TY, TH, TW = 38, 20, 20.5
box(1.5,  TY, TW, TH, "1 · PATIENT-LEVEL SPLIT",
    ["5,856 chest radiographs", "MD5 patient hashing", "0% subject overlap", "224×224 CLAHE cache"],
    "#E8F0FE", "#1A5FB4")
box(26.5, TY, TW, TH, "2 · FOUNDATION TEACHER",
    ["RAD-DINO ViT-B/16 (86.6M)", "518×518 in · 768-D embeds", "MLP head (val AUC 0.9998)", "Out-of-fold soft logits"],
    "#F3E8FD", "#7B1FA2")
box(51.5, TY, TW, TH, "3 · DISTILLED STUDENT",
    ["MobileNetV2 (2.4M params)", "Focal loss + logit distillation", "Two-phase fine-tuning", "Calibrated on 764 val images"],
    "#E0F2F1", "#00796B")
box(76.5, TY, TW, TH, "4 · TRUST & TRIAGE",
    ["Temperature scaling (T=1.95)", "MC-Dropout ×20 → uncertainty", "Threshold t* = 0.674", "Referral margin τ = 0.059"],
    "#FFF3E0", "#E65100")

arrow(22.6, TY + TH / 2, 25.9, TY + TH / 2, "CXR", ly=2.4)
arrow(47.6, TY + TH / 2, 50.9, TY + TH / 2, "soft logits", ly=2.4)
arrow(72.6, TY + TH / 2, 75.9, TY + TH / 2, "logits", ly=2.4)

# ── Bottom row: federated (6), edge bundle (5) ─────────────────────────────
BY, BH = 6, 20
box(1.5, BY, TW, BH, "6 · FEDERATED LEARNING",
    ["5 hospital sites (non-IID)", "FedAvg + FedProx, 15 rounds", "Only weight updates shared", "Test AUC 0.9655"],
    "#ECEFF1", "#37474F")
box(51.5, BY, TW, BH, "5 · INT8 EDGE BUNDLE",
    ["TFLite INT8 backbone: 2.73 MB", "NumPy head: 658 KB", "≈13 ms CPU inference", "Raspberry Pi / edge IoT"],
    "#E8F5E9", "#2E7D32")

# ── Triage outcomes: three small boxes side-by-side under Module 4 ─────────
OW, OH, OY = 6.6, 8.5, 17.5
X1, X2, X3 = 76.5, 83.4, 90.3
box(X1, OY, OW, OH, "NORMAL\ncleared", [], "#C8E6C9", "#2E7D32", title_fs=9)
box(X2, OY, OW, OH, "PNEUMONIA\nflagged", [], "#FFCDD2", "#C62828", title_fs=9)
box(X3, OY, OW, OH, "REFER\nradiologist", [], "#FFF9C4", "#B8860B", title_fs=9)

# Fan of arrows from Module 4 into the three outcome boxes
JUN_X, JUN_Y = 86.75, 31.0
arrow(86.75, TY, JUN_X, JUN_Y)                       # trunk down from Module 4
arrow(JUN_X, JUN_Y, X1 + OW / 2, OY + OH + 0.4)      # → NORMAL
arrow(JUN_X, JUN_Y, X2 + OW / 2, OY + OH + 0.4)      # → PNEUMONIA
arrow(JUN_X, JUN_Y, X3 + OW / 2, OY + OH + 0.4)      # → REFER

# ── Cross links ────────────────────────────────────────────────────────────
arrow(61.75, TY, 61.75, BY + BH, "INT8 export", lx=7.0, ly=0)          # 3 → 5
arrow(11.75, TY, 11.75, BY + BH, "site partitions", lx=9.0, ly=0, dashed=True)  # 1 → 6
arrow(22.6, BY + BH / 2, 50.9, BY + BH / 2, "global student weights", ly=2.4)   # 6 → 5

plt.tight_layout(pad=0.4)
plt.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print("Saved:", OUT)
