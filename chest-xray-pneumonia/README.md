# Chest X-Ray Pneumonia Detection — TrustEdge-CXR 🫁

> **Review 2:** a small, calibrated and privacy-preserving pneumonia screener for edge IoT devices.
> A 2.4 M-parameter MobileNetV2 learns from a chest-X-ray foundation model (RAD-DINO),
> says **REFER** when it is unsure, explains itself with tested Grad-CAM++ maps,
> runs as a 2.7 MB INT8 model on a Raspberry Pi, and can be trained across hospitals with federated learning.
>
> **Review 1** (MobileNetV2 transfer learning) is kept unchanged in `src/` as the baseline.

---

## What changed since Review 1

| Gap found in the literature survey | Review 1 | Review 2 (TrustEdge-CXR) | Code |
|---|---|---|---|
| **G2** inflated evaluation | random image-level val split (62 % of val images shared a patient with train) | split **by patient**; official 624-image test kept; optional RSNA adult external test | `trustedge/splits.py`, `rsna.py` |
| **G1** accuracy vs size | ImageNet MobileNetV2 | **knowledge distillation** from RAD-DINO (86.6 M params, 882k CXRs) into the 2.4 M student, focal loss, CLAHE | `teacher.py`, `train_student.py`, `losses.py` |
| **G3** no calibrated uncertainty | fixed 0.5 threshold | temperature scaling, threshold tuned for ≥95 % sensitivity, MC-dropout **refer-if-unsure** | `calibrate.py`, `edge/trustedge_runtime.py` |
| **G4** unvalidated explanations | Grad-CAM shown | exact Grad-CAM++ + PSC-style perturbation test + RSNA pointing game | `explain_eval.py` |
| **G5** privacy & edge in silos | central training, 23.8 MB Keras file | INT8 TFLite edge bundle + federated learning (FedProx + server momentum, optional DP) | `export_tflite.py`, `federated.py`, `edge/` |

Results land in `artifacts/results/comparison.md` (before/after table) and the dashboard's **Performance** tab.

---

## Project structure

```
chest-xray-pneumonia/
├── trustedge/                # Review 2 pipeline (run as python -m trustedge.<step>)
│   ├── prepare_data.py       # 1 · patient-level split + CLAHE cache
│   ├── teacher.py            # 2 · RAD-DINO teacher (PyTorch)
│   ├── train_student.py      # 3 · MobileNetV2 student, focal + distillation
│   ├── calibrate.py          # 4 · temperature, threshold, referral cut-off
│   ├── export_tflite.py      # 5 · INT8 edge bundle
│   ├── federated.py          # 6 · federated learning simulation (+ DP)
│   ├── evaluate.py           # 7 · before/after comparison + plots
│   ├── explain_eval.py       # 8 · Grad-CAM vs Grad-CAM++ tests
│   ├── rsna.py               #   · optional RSNA external test set
│   └── config.py, splits.py, tfdata.py, model.py, losses.py, metrics.py
├── edge/                     # numpy-only runtime shared by Pi, server and evaluation
│   ├── trustedge_runtime.py  # CLAHE, head + MC dropout, decision rule, Grad-CAM++
│   ├── edge_infer.py         # Raspberry Pi script (folder watch + report to server)
│   └── requirements-edge.txt
├── backend/main.py           # FastAPI dashboard (predict, metrics, edge reports)
├── frontend/                 # Diagnose · Performance · Edge devices
├── notebooks/
│   ├── TrustEdge_CXR_Colab.ipynb        # Review 2 training on a free Colab GPU
│   └── pneumonia_classifier_colab.ipynb # Review 1
├── tools/build_colab_notebook.py        # regenerates the Colab notebook from the code
├── run_pipeline.sh           # whole Review 2 pipeline on this machine
├── src/                      # Review 1 baseline (unchanged)
├── models/                   # Review 1 model + plots
└── artifacts/                # everything Review 2 writes (created by the pipeline)
```

---

## 1 · Setup

```bash
cd chest-xray-pneumonia
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Dataset (same as Review 1): put `kaggle.json` in `~/.kaggle/`, then

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data      # → data/chest_xray/{train,val,test}
```

## 2 · Train — pick one

### Option A · Google Colab (recommended, ~45–60 min on a free T4)

1. Open `notebooks/TrustEdge_CXR_Colab.ipynb` in Colab → `Runtime → Change runtime type → T4 GPU`.
2. Run all; upload `kaggle.json` (and optionally the Review 1 model) when asked.
3. Unzip the downloaded `trustedge_results.zip` **inside this folder** — it creates `artifacts/`.

### Option B · This machine

The RAD-DINO teacher needs PyTorch (`pip install torch transformers`); on a Mac it uses the
Apple GPU. The TensorFlow steps run on CPU (~1–2 h on an M2).

```bash
./run_pipeline.sh                              # everything
SKIP_FL=1 SKIP_ABLATION=1 ./run_pipeline.sh    # core results only
```

Or step by step: `python -m trustedge.prepare_data`, `… .teacher`, `… .train_student`,
`… .calibrate`, `… .export_tflite`, `… .evaluate`, `… .explain_eval`, `… .federated`.
Every step takes `--limit 32` for a quick smoke test.

## 3 · Demo dashboard

```bash
uvicorn backend.main:app --reload --port 8000     # open http://localhost:8000
```

- **Diagnose** — upload an X-ray → NORMAL / PNEUMONIA / **REFER**, calibrated probability with the
  decision threshold, MC-dropout uncertainty with the referral cut-off τ, and a Grad-CAM++ overlay.
- **Performance** — before/after table on the official test set, calibration, ROC, heatmap examples.
- **Edge devices** — live results pushed by Raspberry Pi nodes.

Without `artifacts/student/` the server falls back to the Review 1 model.

## 4 · Edge IoT node (Raspberry Pi 4/5, 64-bit OS)

Copy `artifacts/edge_bundle/` to the Pi, then:

```bash
pip install -r requirements-edge.txt
python edge_infer.py xray.jpeg                                   # one image
python edge_infer.py --watch /srv/xray_out --server http://<laptop-ip>:8000 --send-image
```

Set `TRUSTEDGE_EDGE_TOKEN` on the server and pass `--token` on the Pi to require a shared secret.
Images stay on the Pi; only the decision (and, with `--send-image`, the heatmap) is sent.

---

## Design notes

| Decision | Why |
|---|---|
| Official test set kept untouched | fair before/after comparison with Review 1's 84.8 % |
| Out-of-fold teacher logits on train | the student never distils labels the teacher head memorised |
| Model outputs a logit | distillation, temperature scaling and MC dropout all use the same quantity |
| MC dropout on the head only | backbone runs once; 20 head passes add ~1 % compute, so it is cheap on a Pi |
| Grad-CAM++ in closed form | the head is GAP → MLP, so gradients are analytic — the Pi needs no autodiff |
| No horizontal flip | chest anatomy is not left–right symmetric |
| INT8 post-training quantization | 3.3× smaller; accuracy change is measured in `edge_metrics.json` |
| DP is optional | with only 5 sites, client-level DP noise costs a lot of accuracy — reported as a trade-off |

---

## Review 1 baseline (unchanged)

```bash
cd src
python resplit.py   # creates val2/ (image-level split — see G2 above)
python train.py     # two-phase MobileNetV2 training
python evaluate.py  # confusion matrix + ROC
```

Measured on the official test set: 84.8 % accuracy, 99.0 % sensitivity, 61.1 % specificity, AUC 0.966.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Library not loaded: libtensorflow_cc.2.dylib` | the TensorFlow install is damaged: `pip install --force-reinstall --no-deps tensorflow==2.21.0` |
| `splits.csv not found` | run `python -m trustedge.prepare_data` first |
| `Teacher logits missing` | run `python -m trustedge.teacher`, or train with `--no-kd` |
| `CLAHE needs OpenCV` | `pip install opencv-python-headless` |
| `Install a TFLite interpreter` | `pip install ai-edge-litert` |
| Colab out of memory in the teacher step | `python -m trustedge.teacher --batch-size 16` |

> ⚠️ Educational project — not a medical device, not for clinical use.
