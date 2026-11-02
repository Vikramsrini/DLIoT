#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — runs the whole TrustEdge-CXR pipeline on this machine.
# (On Colab, use notebooks/TrustEdge_CXR_Colab.ipynb instead.)
#
#   ./run_pipeline.sh                      # everything
#   SKIP_FL=1 SKIP_ABLATION=1 ./run_pipeline.sh
#
# PY          python with TensorFlow           (default: venv/bin/python)
# TEACHER_PY  python with torch + transformers (default: $PY)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-venv/bin/python}
TEACHER_PY=${TEACHER_PY:-$PY}
TEACHER_ARGS=${TEACHER_ARGS:-}

run() { echo; echo "▶ $*"; "$@"; }

run "$PY" -m trustedge.prepare_data
run "$TEACHER_PY" -m trustedge.teacher $TEACHER_ARGS
run "$PY" -m trustedge.train_student
run "$PY" -m trustedge.calibrate
run "$PY" -m trustedge.export_tflite
run "$PY" -m trustedge.evaluate
run "$PY" -m trustedge.explain_eval

extra=0
if [[ -z "${SKIP_ABLATION:-}" ]]; then
  run "$PY" -m trustedge.train_student --no-kd --out student_nokd
  run "$PY" -m trustedge.calibrate --student student_nokd
  extra=1
fi
if [[ -z "${SKIP_FL:-}" ]]; then
  run "$PY" -m trustedge.federated
  extra=1
fi
if [[ $extra == 1 ]]; then
  run "$PY" -m trustedge.evaluate     # refresh the table with the extra rows
fi
echo; echo "✅ Pipeline finished — see artifacts/results/comparison.md"
