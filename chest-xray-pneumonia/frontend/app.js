const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const dropzoneEmpty = document.getElementById("dropzone-empty");
const previewImg = document.getElementById("preview-img");
const analyzeBtn = document.getElementById("analyze-btn");
const resetBtn = document.getElementById("reset-btn");
const results = document.getElementById("results");

let selectedFile = null;

// ── Tabs ─────────────────────────────────────────────────────────────────
const tabs = document.querySelectorAll(".tab");
const panels = {
  diagnose: document.getElementById("panel-diagnose"),
  performance: document.getElementById("panel-performance"),
  edge: document.getElementById("panel-edge"),
};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");

    Object.entries(panels).forEach(([key, el]) => {
      el.hidden = key !== tab.dataset.tab;
    });

    if (tab.dataset.tab === "performance") loadMetrics();
    if (tab.dataset.tab === "edge") startEdgePolling();
    else stopEdgePolling();
  });
});

// ── Small DOM helper (text only — never innerHTML with server data) ─────────
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

const pct = (v) => (v === null || v === undefined || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(1)}%`);

// ── Model info ──────────────────────────────────────────────────────────
(async function loadInfo() {
  try {
    const res = await fetch("/api/info");
    if (!res.ok) return;
    const info = await res.json();
    const badge = document.getElementById("model-badge");
    badge.textContent =
      info.mode === "trustedge"
        ? `Model: ${info.model} · threshold ${info.threshold.toFixed(3)} · τ ${info.tau.toFixed(3)}`
        : `Model: ${info.model} — train TrustEdge-CXR to enable calibration and referral`;
    badge.hidden = false;
  } catch {
    /* server not ready */
  }
})();

// ── Dropzone interactions ───────────────────────────────────────────────
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) return;
  selectedFile = file;

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.hidden = false;
  dropzoneEmpty.hidden = true;

  analyzeBtn.disabled = false;
  resetBtn.hidden = false;
  results.hidden = true;
}

resetBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewImg.hidden = true;
  dropzoneEmpty.hidden = false;
  analyzeBtn.disabled = true;
  resetBtn.hidden = true;
  results.hidden = true;
});

// ── Analyze ──────────────────────────────────────────────────────────────
const btnLabel = analyzeBtn.querySelector(".btn-label");
const btnSpinner = analyzeBtn.querySelector(".spinner");

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setLoading(true);
  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch("/api/predict", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    alert(`Prediction failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  btnSpinner.hidden = !isLoading;
  btnLabel.textContent = isLoading ? "Analyzing…" : "Analyze X-ray";
}

function renderResult(data) {
  document.getElementById("result-original").src = data.original;
  document.getElementById("result-heatmap").src = data.heatmap_overlay;
  document.getElementById("heatmap-caption").textContent = `${data.explainer} attention`;

  const decision = data.decision || data.label;
  const label = document.getElementById("verdict-label");
  label.textContent = decision === "REFER" ? "REFER TO RADIOLOGIST" : decision;
  label.className = "verdict-label " + decision.toLowerCase();

  document.getElementById("verdict-sub").textContent =
    decision === "REFER"
      ? `Model leans ${data.label} but is unsure`
      : `${(data.confidence * 100).toFixed(1)}% confidence`;

  document.getElementById("confidence-fill").style.width = `${data.prob_pneumonia * 100}%`;
  document.getElementById("confidence-pct").textContent = `P(PNEUMONIA) = ${data.prob_pneumonia.toFixed(3)}`;
  document.getElementById("threshold-marker").style.left = `${data.threshold * 100}%`;

  // Std of Bernoulli probabilities is at most 0.5, so scale the meter to 0–0.5
  const uncWrap = document.getElementById("uncertainty-wrap");
  if (data.uncertainty !== null && data.uncertainty !== undefined) {
    document.getElementById("uncertainty-fill").style.width = `${Math.min(data.uncertainty / 0.5, 1) * 100}%`;
    document.getElementById("tau-marker").style.left = `${Math.min(data.tau / 0.5, 1) * 100}%`;
    document.getElementById("uncertainty-text").textContent =
      `${data.uncertainty.toFixed(3)} (refer above τ = ${data.tau.toFixed(3)})`;
    uncWrap.hidden = false;
  } else {
    uncWrap.hidden = true;
  }

  document.getElementById("result-note").textContent =
    decision === "REFER"
      ? "The 20 dropout passes disagree more than the referral cut-off allows, so this film goes to a radiologist instead of being auto-labelled."
      : "The heatmap highlights lung regions that most influenced the model's decision — brighter (red/yellow) regions carried more weight. The amber marker on the bar is the decision threshold.";

  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Performance metrics ─────────────────────────────────────────────────
let metricsLoaded = false;

async function loadMetrics() {
  if (metricsLoaded) return;
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) throw new Error("metrics unavailable");
    const data = await res.json();
    if (data.mode === "trustedge") {
      renderComparison(data);
      metricsLoaded = true;
      return;
    }

    document.getElementById("stat-accuracy").textContent = `${(data.val_accuracy * 100).toFixed(1)}%`;
    document.getElementById("stat-auc").textContent = data.val_auc.toFixed(3);
    document.getElementById("stat-precision").textContent = data.val_precision.toFixed(3);
    document.getElementById("stat-recall").textContent = data.val_recall.toFixed(3);
    document.getElementById(
      "stat-caption"
    ).textContent = `Phase 1: ${data.phase1_epochs} epochs (frozen backbone) · Phase 2: ${data.phase2_epochs} epochs (fine-tuning) · metrics on the val2/ split.`;

    metricsLoaded = true;
  } catch {
    document.getElementById("stat-caption").textContent = "Training history not available.";
  }
}

function renderComparison(data) {
  document.getElementById("legacy-section").hidden = true;
  document.getElementById("compare-section").hidden = false;
  document.getElementById("perf-sub").textContent =
    "Official 624-image test set. Thresholds are chosen on the patient-level validation split, never on test.";

  const table = document.getElementById("compare-table");
  table.replaceChildren();
  const head = el("tr");
  ["Model", "Threshold", "Accuracy", "Sensitivity", "Specificity", "AUC", "ECE", "Notes"].forEach((h) =>
    head.appendChild(el("th", null, h))
  );
  table.appendChild(el("thead")).appendChild(head);
  const body = table.appendChild(el("tbody"));
  data.rows.forEach((r) => {
    const tr = el("tr", r.name === "TrustEdge student" ? "highlight" : "");
    [
      r.name,
      r.threshold?.toFixed(3),
      pct(r.accuracy),
      pct(r.sensitivity),
      pct(r.specificity),
      r.auc?.toFixed(3),
      r.ece?.toFixed(3),
      r.note,
    ].forEach((v) => tr.appendChild(el("td", null, v ?? "–")));
    body.appendChild(tr);
  });

  const grid = document.getElementById("insight-grid");
  grid.replaceChildren();
  const card = (value, label) => {
    const c = el("div", "stat-card");
    c.append(el("span", "stat-value", value), el("span", "stat-label", label));
    grid.appendChild(c);
  };
  if (data.leakage) card(pct(data.leakage.leak_fraction), "Review 1 val images that shared a patient with train");
  if (data.referral) {
    card(`${data.referral.referred}`, "test films referred to a radiologist");
    card(pct(data.referral.kept_accuracy), "accuracy on the films decided automatically");
  }
  if (data.explanations?.psc?.["gradcam++"]) {
    card(data.explanations.psc["gradcam++"].psc_spearman.toFixed(2), "Grad-CAM++ PSC ρ (higher = more faithful)");
  }

  const charts = document.getElementById("compare-charts");
  charts.replaceChildren();
  const captions = {
    "confusion_matrices.png": "Confusion matrices — Review 1 vs TrustEdge",
    "roc_curves.png": "ROC curves",
    "reliability.png": "Calibration (reliability diagram)",
    "heatmap_examples.png": "Grad-CAM++ examples",
  };
  data.figures.forEach((f) => {
    const fig = el("figure", "chart-figure");
    fig.appendChild(el("figcaption", null, captions[f] || f));
    const img = el("img");
    img.src = `/api/assets/${encodeURIComponent(f)}`;
    img.alt = captions[f] || f;
    fig.appendChild(img);
    charts.appendChild(fig);
  });
}

// ── Edge devices ────────────────────────────────────────────────────────
let edgeTimer = null;

function startEdgePolling() {
  loadEdgeReports();
  if (!edgeTimer) edgeTimer = setInterval(loadEdgeReports, 5000);
}

function stopEdgePolling() {
  clearInterval(edgeTimer);
  edgeTimer = null;
}

async function loadEdgeReports() {
  const list = document.getElementById("edge-list");
  try {
    const res = await fetch("/api/edge/reports");
    const { reports } = await res.json();
    list.replaceChildren();
    if (!reports.length) {
      list.appendChild(el("p", "stat-caption", "No reports yet — run edge_infer.py with --server pointing here."));
      return;
    }
    reports.forEach((r) => {
      const item = el("article", "edge-item");
      if (r.overlay_png && /^[A-Za-z0-9+/=]+$/.test(r.overlay_png)) {
        const img = el("img", "edge-thumb");
        img.src = `data:image/png;base64,${r.overlay_png}`;
        img.alt = `Heatmap for ${r.file}`;
        item.appendChild(img);
      }
      const info = el("div", "edge-info");
      const top = el("div", "edge-top");
      top.append(el("span", `edge-badge ${String(r.decision).toLowerCase()}`, r.decision), el("strong", null, r.file));
      info.appendChild(top);
      const when = r.timestamp ? new Date(r.timestamp * 1000).toLocaleTimeString() : "";
      info.appendChild(
        el(
          "p",
          "edge-meta",
          `${r.device_id} · P(pneumonia) ${Number(r.prob_pneumonia).toFixed(3)}` +
            (r.uncertainty != null ? ` · uncertainty ${Number(r.uncertainty).toFixed(3)}` : "") +
            (r.latency_ms != null ? ` · ${Number(r.latency_ms).toFixed(0)} ms on device` : "") +
            (when ? ` · ${when}` : "")
        )
      );
      item.appendChild(info);
      list.appendChild(item);
    });
  } catch {
    list.replaceChildren(el("p", "stat-caption", "Could not load edge reports."));
  }
}
