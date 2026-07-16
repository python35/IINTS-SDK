const tauriCore = window.__TAURI__?.core;
const invoke = tauriCore?.invoke;

let selectedWorkflow = null;
let workflows = [];
let lastPreview = null;

const $ = (id) => document.getElementById(id);

function setText(id, value) {
  $(id).textContent = value;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

async function call(command, args = {}) {
  if (!invoke) {
    throw new Error("Tauri invoke API is not available. Run this through `npm run tauri dev`.");
  }
  return await invoke(command, args);
}

function workflowCard(workflow) {
  const card = document.createElement("article");
  card.className = "workflow-card";
  card.tabIndex = 0;
  card.dataset.key = workflow.key;
  card.innerHTML = `
    <h3>${escapeHtml(workflow.title)}</h3>
    <p><strong>${escapeHtml(workflow.audience)}</strong> · ${escapeHtml(workflow.preset_name)}</p>
    <p>${escapeHtml(workflow.description)}</p>
  `;
  card.addEventListener("click", () => selectWorkflow(workflow.key));
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectWorkflow(workflow.key);
    }
  });
  return card;
}

function renderWorkflows() {
  const list = $("workflow-list");
  list.replaceChildren();
  for (const workflow of workflows) {
    list.appendChild(workflowCard(workflow));
  }
  if (!selectedWorkflow && workflows.length) {
    selectWorkflow(workflows[0].key);
  } else {
    markSelectedWorkflow();
  }
}

function markSelectedWorkflow() {
  document.querySelectorAll(".workflow-card").forEach((card) => {
    card.classList.toggle("selected", card.dataset.key === selectedWorkflow);
  });
}

function selectWorkflow(key) {
  selectedWorkflow = key;
  markSelectedWorkflow();
}

async function loadStatus() {
  try {
    const status = await call("desktop_status");
    $("sdk-status").textContent = `SDK ${status.sdk_version} via ${status.python_executable}`;
    setText("run-status", "Python SDK bridge ready.\nSelect a workflow and run it.");
  } catch (error) {
    $("sdk-status").textContent = "Python bridge unavailable";
    setText("run-status", String(error));
  }
}

async function loadWorkflows() {
  setText("run-status", "Loading curated SDK workflows...");
  try {
    const payload = await call("list_workflows");
    workflows = payload.workflows || [];
    renderWorkflows();
    setText("run-status", `Loaded ${workflows.length} workflows.`);
  } catch (error) {
    setText("run-status", String(error));
  }
}

async function runSelectedWorkflow() {
  if (!selectedWorkflow) {
    setText("run-status", "Select a workflow first.");
    return;
  }
  const outputDir = $("output-dir").value.trim();
  const seed = Number.parseInt($("seed").value || "42", 10);
  setBusy(true);
  setText("run-status", `Running ${selectedWorkflow} through the Python SDK...\nThis may take a minute.`);
  try {
    const result = await call("run_workflow", {
      workflowKey: selectedWorkflow,
      outputDir,
      seed
    });
    setText("run-status", result.summary || pretty(result));
    if (result.results_csv) {
      $("csv-path").value = result.results_csv;
      await previewCsv();
    }
  } catch (error) {
    setText("run-status", String(error));
  } finally {
    setBusy(false);
  }
}

async function previewCsv() {
  const csv = $("csv-path").value.trim();
  if (!csv) {
    return;
  }
  setText("run-status", `Loading preview: ${csv}`);
  try {
    const preview = await call("preview_results", { csv, maxRows: 80 });
    lastPreview = preview;
    renderMetrics(preview.metrics || {});
    renderTable(preview.columns || [], preview.rows || []);
    drawGlucoseChart(preview);
    setText("run-status", `Preview loaded: ${preview.row_count} rows\n${csv}`);
  } catch (error) {
    setText("run-status", String(error));
  }
}

function renderMetrics(metrics) {
  const container = $("metrics");
  container.replaceChildren();
  for (const [key, value] of Object.entries(metrics)) {
    const item = document.createElement("div");
    item.className = "metric";
    item.innerHTML = `<small>${escapeHtml(key)}</small><strong>${escapeHtml(String(value))}</strong>`;
    container.appendChild(item);
  }
}

function renderTable(columns, rows) {
  const table = $("preview-table");
  table.replaceChildren();
  if (!columns.length) {
    return;
  }
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.textContent = column;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const cell of row) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
}

function drawGlucoseChart(preview) {
  const svg = $("glucose-chart");
  svg.replaceChildren();
  const rows = preview.rows || [];
  const columns = preview.columns || [];
  const glucoseIndex = firstIndex(columns, [
    "glucose_actual_mgdl",
    "glucose",
    "cgm_mgdl",
    "sensor_glucose_mgdl",
    "glucose_mgdl"
  ]);
  if (glucoseIndex < 0 || rows.length < 2) {
    svg.appendChild(svgText(24, 40, "No glucose preview available."));
    return;
  }
  const values = rows.map((row) => Number.parseFloat(row[glucoseIndex])).filter(Number.isFinite);
  if (values.length < 2) {
    svg.appendChild(svgText(24, 40, "Glucose values are not numeric."));
    return;
  }

  const width = 700;
  const height = 260;
  const pad = { left: 54, right: 18, top: 18, bottom: 38 };
  const minY = Math.min(50, Math.floor(Math.min(...values) / 10) * 10);
  const maxY = Math.max(260, Math.ceil(Math.max(...values) / 10) * 10);
  const xScale = (index) => pad.left + (index / (values.length - 1)) * (width - pad.left - pad.right);
  const yScale = (value) =>
    pad.top + (1 - (value - minY) / (maxY - minY)) * (height - pad.top - pad.bottom);

  svg.appendChild(svgRect(pad.left, yScale(180), width - pad.left - pad.right, yScale(70) - yScale(180), "#dcfce7", 0.75));
  for (const tick of [70, 180]) {
    svg.appendChild(svgLine(pad.left, yScale(tick), width - pad.right, yScale(tick), tick === 70 ? "#b91c1c" : "#ca8a04", 1, "6 5"));
    svg.appendChild(svgText(8, yScale(tick) + 4, String(tick)));
  }
  svg.appendChild(svgLine(pad.left, pad.top, pad.left, height - pad.bottom, "#8ca1b2", 1));
  svg.appendChild(svgLine(pad.left, height - pad.bottom, width - pad.right, height - pad.bottom, "#8ca1b2", 1));

  const points = values.map((value, index) => `${xScale(index).toFixed(1)},${yScale(value).toFixed(1)}`).join(" ");
  const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  polyline.setAttribute("points", points);
  polyline.setAttribute("fill", "none");
  polyline.setAttribute("stroke", "#0f766e");
  polyline.setAttribute("stroke-width", "3");
  polyline.setAttribute("stroke-linejoin", "round");
  polyline.setAttribute("stroke-linecap", "round");
  svg.appendChild(polyline);
  svg.appendChild(svgText(pad.left, height - 10, "Preview samples"));
  svg.appendChild(svgText(width - 128, 28, "70-180 mg/dL target"));
}

async function checkAi() {
  setText("ai-status", "Checking local Ollama...");
  try {
    const payload = await call("check_local_ai", {
      model: $("ai-model").value.trim(),
      host: $("ai-host").value.trim()
    });
    setText("ai-status", pretty(payload));
  } catch (error) {
    setText("ai-status", String(error));
  }
}

async function listAiModels() {
  setText("ai-status", "Listing local Ollama models...");
  try {
    const payload = await call("list_local_ai_models", { host: $("ai-host").value.trim() });
    setText("ai-status", pretty(payload));
  } catch (error) {
    setText("ai-status", String(error));
  }
}

function setBusy(isBusy) {
  $("run-btn").disabled = isBusy;
  $("refresh-btn").disabled = isBusy;
  $("preview-btn").disabled = isBusy;
}

function firstIndex(columns, candidates) {
  const lower = columns.map((column) => String(column).toLowerCase());
  for (const candidate of candidates) {
    const index = lower.indexOf(candidate.toLowerCase());
    if (index >= 0) return index;
  }
  return -1;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function svgText(x, y, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "text");
  node.setAttribute("x", String(x));
  node.setAttribute("y", String(y));
  node.setAttribute("fill", "#607080");
  node.setAttribute("font-size", "12");
  node.textContent = text;
  return node;
}

function svgLine(x1, y1, x2, y2, stroke, width, dash = "") {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "line");
  node.setAttribute("x1", String(x1));
  node.setAttribute("y1", String(y1));
  node.setAttribute("x2", String(x2));
  node.setAttribute("y2", String(y2));
  node.setAttribute("stroke", stroke);
  node.setAttribute("stroke-width", String(width));
  if (dash) node.setAttribute("stroke-dasharray", dash);
  return node;
}

function svgRect(x, y, width, height, fill, opacity) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  node.setAttribute("x", String(x));
  node.setAttribute("y", String(y));
  node.setAttribute("width", String(width));
  node.setAttribute("height", String(height));
  node.setAttribute("fill", fill);
  node.setAttribute("opacity", String(opacity));
  return node;
}

$("run-btn").addEventListener("click", runSelectedWorkflow);
$("refresh-btn").addEventListener("click", loadWorkflows);
$("preview-btn").addEventListener("click", previewCsv);
$("ai-check-btn").addEventListener("click", checkAi);
$("ai-models-btn").addEventListener("click", listAiModels);

await loadStatus();
await loadWorkflows();
