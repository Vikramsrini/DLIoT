"""
build_colab_notebook.py
───────────────────────
Generates notebooks/TrustEdge_CXR_Colab.ipynb with the current trustedge/
and edge/ source files embedded (%%writefile cells), so the notebook always
matches the code in this repository.

Run after changing any pipeline file:
    python tools/build_colab_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "TrustEdge_CXR_Colab.ipynb"
PROJECT = "/content/trustedge"
EMBED = [
    *sorted((ROOT / "trustedge").glob("*.py")),
    ROOT / "edge" / "__init__.py",
    ROOT / "edge" / "trustedge_runtime.py",
    ROOT / "edge" / "edge_infer.py",
    ROOT / "edge" / "README.md",
    ROOT / "edge" / "requirements-edge.txt",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
# TrustEdge-CXR — Review 2 training notebook

Trains the improved pneumonia screener end to end on a free Colab GPU:

| Step | What it does | Gap closed | T4 time |
|---|---|---|---|
| 1 · prepare_data | patient-level train/val split (official test kept), CLAHE cache | G2 | ~1 min |
| 2 · teacher | RAD-DINO (86.6 M params, 882k CXRs) embeddings → linear probe + MLP teacher | G1, G2 | ~5 min |
| 3 · train_student | MobileNetV2 (2.4 M) with focal loss + knowledge distillation | G1 | ~15 min |
| 4 · calibrate | temperature scaling, tuned threshold, MC-dropout referral cut-off | G3 | <1 min |
| 5 · export_tflite | INT8 TFLite edge bundle for Raspberry Pi / Android | G1, G5 | ~2 min |
| 6 · federated | 5 simulated hospitals, FedProx + server momentum (optional DP) | G5 | ~15 min |
| 7 · evaluate + explain_eval | before/after table, calibration, Grad-CAM++ PSC test | G3, G4 | ~3 min |

**How to run**
1. `Runtime → Change runtime type → T4 GPU`
2. Run the first cells and upload `kaggle.json` when asked (kaggle.com → Settings → API → Create New Token).
3. Optionally upload the Review 1 model (`models/pneumonia_mobilenetv2_final.keras`) so the table starts from your baseline.
4. `Runtime → Run all`. At the end, `trustedge_results.zip` downloads.
5. On your laptop, unzip it **inside the `chest-xray-pneumonia/` folder** (it creates `artifacts/`), then start the demo:
   `uvicorn backend.main:app --port 8000`.

The code cells under *Project code* are generated from the repository by `tools/build_colab_notebook.py`, so edit the `.py` files there rather than here.
"""

SETUP = f"""
# ── Cell 1 · Runtime check, packages, folders ─────────────────────────────────
import os, subprocess

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "none — use Runtime → Change runtime type → T4 GPU")

%pip install -q pydicom dp-accounting ai-edge-litert opencv-python-headless
try:
    import tensorflow as tf
except ImportError:
    %pip install -q tensorflow
    import tensorflow as tf
import keras, torch, transformers
print(f"TensorFlow {{tf.__version__}} · Keras {{keras.__version__}} · PyTorch {{torch.__version__}} · "
      f"transformers {{transformers.__version__}}")

PROJECT = "{PROJECT}"
DATA = "/content/data/chest_xray"
for d in ("trustedge", "edge", "models", "artifacts"):
    os.makedirs(f"{{PROJECT}}/{{d}}", exist_ok=True)
os.environ["TRUSTEDGE_DATA"] = DATA
os.environ["TRUSTEDGE_ARTIFACTS"] = f"{{PROJECT}}/artifacts"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
"""

OPTIONS = """
# ── Cell 2 · Options ─────────────────────────────────────────────────────────
RUN_ABLATION     = True    # student without distillation, for the ablation row (+~15 min)
RUN_FEDERATED    = True    # 5 simulated hospitals, no DP (+~15 min)
RUN_FEDERATED_DP = False   # same with DP noise — shows the privacy/accuracy trade-off (+~15 min)
RUN_RSNA         = False   # external adult test set; accept the competition rules on kaggle.com
                           # (rsna-pneumonia-detection-challenge) first — 3.7 GB download
SAVE_TO_DRIVE    = False   # also copy trustedge_results.zip to Google Drive
"""

KAGGLE = """
# ── Cell 3 · Kaggle credentials & dataset ────────────────────────────────────
from google.colab import files

if not os.path.exists("/root/.kaggle/kaggle.json"):
    print("Upload your kaggle.json (kaggle.com → Settings → API → Create New Token)")
    uploaded = files.upload()
    os.makedirs("/root/.kaggle", exist_ok=True)
    with open("/root/.kaggle/kaggle.json", "wb") as f:
        f.write(uploaded["kaggle.json"])
    os.chmod("/root/.kaggle/kaggle.json", 0o600)

if not os.path.exists(DATA):
    !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p /content --quiet
    !unzip -q -o /content/chest-xray-pneumonia.zip -d /content/data
for split in ("train", "val", "test"):
    for cls in ("NORMAL", "PNEUMONIA"):
        print(f"{split}/{cls}: {len(os.listdir(f'{DATA}/{split}/{cls}'))} images")
"""

BASELINE = """
# ── Cell 4 · (optional) Review 1 baseline model ──────────────────────────────
# Upload models/pneumonia_mobilenetv2_final.keras (23.8 MB) from your laptop so the
# comparison table starts from the Review 1 model. Cancel the picker to skip.
baseline = f"{PROJECT}/models/pneumonia_mobilenetv2_final.keras"
if not os.path.exists(baseline):
    try:
        up = files.upload()
        for name, data in up.items():
            if name.endswith(".keras"):
                open(baseline, "wb").write(data)
    except Exception as e:
        print("No baseline uploaded:", e)
print("Baseline present:", os.path.exists(baseline))
"""

STEPS = [
    ("Step 1 · Patient-level splits + CLAHE cache", "!python -m trustedge.prepare_data"),
    ("Step 2 · RAD-DINO teacher (PyTorch)", "!python -m trustedge.teacher --workers 2"),
    ("Step 3 · Student with distillation (TensorFlow)", "!python -m trustedge.train_student"),
    ("Step 3b · Ablation: student without distillation",
     "if RUN_ABLATION:\n    !python -m trustedge.train_student --no-kd --out student_nokd\n"
     "    !python -m trustedge.calibrate --student student_nokd"),
    ("Step 4 · Calibration, threshold and referral rule", "!python -m trustedge.calibrate"),
    ("Step 5 · INT8 TFLite edge bundle", "!python -m trustedge.export_tflite"),
    ("Step 6 · Federated learning (5 simulated hospitals)",
     "if RUN_FEDERATED:\n    !python -m trustedge.federated\n"
     "if RUN_FEDERATED_DP:\n    !python -m trustedge.federated --dp-clip 1.0 --dp-noise 0.3 --out federated_dp"),
    ("Step 7 · (optional) RSNA external test set",
     "if RUN_RSNA:\n"
     "    if not os.path.exists('/content/rsna-pneumonia-detection-challenge.zip'):\n"
     "        !kaggle competitions download -c rsna-pneumonia-detection-challenge -p /content\n"
     "    !python -m trustedge.rsna --source /content/rsna-pneumonia-detection-challenge.zip"),
    ("Step 8 · Evaluation and explanation tests",
     "ext = '--external' if RUN_RSNA else ''\n"
     "!python -m trustedge.evaluate $ext\n"
     "!python -m trustedge.explain_eval"),
]

SHOW = """
# ── Results ──────────────────────────────────────────────────────────────────
from IPython.display import Image, Markdown, display
R = f"{PROJECT}/artifacts/results"
display(Markdown(open(f"{R}/comparison.md").read()))
for fig in ("confusion_matrices.png", "roc_curves.png", "reliability.png", "heatmap_examples.png"):
    if os.path.exists(f"{R}/{fig}"):
        display(Image(f"{R}/{fig}", width=900))
"""

PACKAGE = """
# ── Package & download (unzip inside chest-xray-pneumonia/ on your laptop) ────
import zipfile, glob
skip = ("/cache/", "embeddings.npz", "student_savedmodel", ".weights.h5", "/rsna/raw224/", "/rsna/clahe224/")
zip_path = "/content/trustedge_results.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for path in glob.glob(f"{PROJECT}/artifacts/**", recursive=True):
        if os.path.isfile(path) and not any(s in path for s in skip):
            z.write(path, os.path.relpath(path, PROJECT))
print(f"{zip_path}: {os.path.getsize(zip_path) / 1e6:.1f} MB")
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    !cp /content/trustedge_results.zip /content/drive/MyDrive/
files.download(zip_path)
"""


def main():
    cells = [md(INTRO), code(SETUP), code(OPTIONS), code(KAGGLE),
             md("## Project code\nGenerated from the repository — run these cells, no need to edit."),]
    for path in EMBED:
        rel = path.relative_to(ROOT).as_posix()
        cells.append(code(f"%%writefile {PROJECT}/{rel}\n" + path.read_text()))
    cells += [code(BASELINE), md("## Pipeline"), code(f"%cd {PROJECT}")]
    for title, cmd in STEPS:
        cells += [md(f"### {title}"), code(cmd)]
    cells += [md("## Results"), code(SHOW), code(PACKAGE)]

    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "name": OUT.name},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(cells)} cells, {len(EMBED)} embedded files)")


if __name__ == "__main__":
    main()
