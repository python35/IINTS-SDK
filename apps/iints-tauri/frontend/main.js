const tauriCore = window.__TAURI__?.core;
const invoke = tauriCore?.invoke;

let selectedWorkflow = null;
let workflows = [];
let lastPreview = null;
let lastRun = null;
let lastMdmp = null;
let molecules = [];
let evidenceConnectors = [];
let updateInfo = null;
let lastGenomics = null;
let lastTissue = null;

const $ = (id) => document.getElementById(id);

function setText(id, value) {
  $(id).textContent = value;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function errorMessage(error) {
  return String(error).replace(/^Error:\s*/i, "").trim();
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
    setText("run-status", errorMessage(error));
  }
}

async function runDiagnostics() {
  const grid = $("diagnostics-grid");
  grid.replaceChildren(statusPill("loading", "Running diagnostics..."));
  try {
    const payload = await call("desktop_diagnostics");
    renderDiagnostics(payload);
  } catch (error) {
    grid.replaceChildren(statusPill("bad", errorMessage(error)));
  }
}

async function loadUpdateInfo() {
  setText("update-status", "Checking SDK/app update information...");
  try {
    updateInfo = await call("desktop_update_info");
    setText(
      "update-status",
      [
        `Installed SDK: ${updateInfo.current_version || "unknown"}`,
        `Python: ${updateInfo.python_executable || "unknown"}`,
        `Package: ${updateInfo.package_spec}`,
        "",
        "SDK update command:",
        updateInfo.pip_command,
        "",
        "Use Open app downloads for .exe/.dmg/Linux bundles."
      ].join("\n")
    );
  } catch (error) {
    setText("update-status", errorMessage(error));
  }
}

async function openAppDownloads() {
  const url = updateInfo?.app_download_url || "https://github.com/python35/IINTS-SDK/releases/tag/desktop-beta-latest";
  await openExternalUrl(url, "update-status");
}

async function openUpdateDocs() {
  const url = updateInfo?.update_docs_url || "https://python35.github.io/IINTS-SDK/APP_INSTALL/";
  await openExternalUrl(url, "update-status");
}

async function copyUpdateCommand() {
  if (!updateInfo) {
    await loadUpdateInfo();
  }
  const command = updateInfo?.pip_command;
  if (!command) {
    setText("update-status", "No SDK update command available yet.");
    return;
  }
  try {
    await navigator.clipboard.writeText(command);
    setText("update-status", `Copied SDK update command:\n${command}`);
  } catch (_error) {
    setText("update-status", `Copy failed. Select and copy manually:\n${command}`);
  }
}

async function openSdkUpdateTerminal() {
  setText("update-status", "Opening a terminal with the fixed SDK update command...");
  try {
    await call("open_sdk_update_terminal");
    setText(
      "update-status",
      [
        "SDK update terminal launched.",
        "The command is fixed by the Rust layer and updates the Python SDK package with desktop/research extras.",
        "",
        updateInfo?.pip_command || "Run Check update info to see the exact command."
      ].join("\n")
    );
  } catch (error) {
    setText("update-status", errorMessage(error));
  }
}

async function loadMolecules() {
  const list = $("molecule-list");
  list.replaceChildren(statusPill("loading", "Loading molecule assets..."));
  try {
    const payload = await call("list_molecule_assets");
    molecules = payload.molecules || [];
    renderMolecules(molecules);
  } catch (error) {
    list.replaceChildren(statusPill("bad", errorMessage(error)));
  }
}

async function loadEvidenceConnectors() {
  const list = $("evidence-list");
  list.replaceChildren(statusPill("loading", "Loading official evidence connectors..."));
  setText("evidence-status", "Loading evidence connector metadata from the Python SDK bridge...");
  try {
    const payload = await call("list_evidence_connectors");
    evidenceConnectors = payload.connectors || [];
    renderEvidenceConnectors(evidenceConnectors);
    setText(
      "evidence-status",
      `Loaded ${evidenceConnectors.length} allowlisted evidence connectors. External portals open in your browser; the SDK does not embed remote web content.`
    );
  } catch (error) {
    list.replaceChildren(statusPill("bad", errorMessage(error)));
    setText("evidence-status", errorMessage(error));
  }
}

function renderEvidenceConnectors(items) {
  const list = $("evidence-list");
  list.replaceChildren();
  if (!items.length) {
    list.appendChild(statusPill("warn", "No evidence connectors returned by the SDK bridge."));
    return;
  }
  for (const connector of items) {
    const card = document.createElement("article");
    card.className = "evidence-card";
    card.innerHTML = `
      <div class="connector-meta">
        <span>${escapeHtml(connector.category || "Evidence")}</span>
        <span class="connector-status">${escapeHtml(connector.integration_status || "Connector")}</span>
      </div>
      <h3>${escapeHtml(connector.title || connector.key || "Evidence connector")}</h3>
      <p>${escapeHtml(connector.why_it_matters || "")}</p>
      <p><strong>Workbench use:</strong> ${escapeHtml(connector.app_use || "")}</p>
      <p class="muted"><strong>Typical query:</strong> ${escapeHtml(connector.default_query || "")}</p>
    `;

    const actions = document.createElement("div");
    actions.className = "button-row";
    actions.appendChild(actionButton("Open portal", () => openExternalUrl(connector.primary_url, "evidence-status"), !connector.primary_url));
    actions.appendChild(actionButton("Open API docs", () => openExternalUrl(connector.docs_url, "evidence-status"), !connector.docs_url));
    card.appendChild(actions);
    list.appendChild(card);
  }
}

function renderMolecules(items) {
  const list = $("molecule-list");
  list.replaceChildren();
  if (!items.length) {
    list.appendChild(statusPill("warn", "No molecule assets found."));
    return;
  }
  for (const molecule of items) {
    const card = document.createElement("article");
    card.className = "molecule-card";

    if (molecule.image_data_url) {
      const image = document.createElement("img");
      image.src = molecule.image_data_url;
      image.alt = `${molecule.title} AlphaFold render`;
      image.loading = "lazy";
      card.appendChild(image);
    }

    const body = document.createElement("div");
    body.className = "molecule-body";
    body.innerHTML = `
      <h3>${escapeHtml(molecule.title)}</h3>
      <p class="muted">UniProt ${escapeHtml(molecule.uniprot_id)}</p>
      <p>${escapeHtml(molecule.explanation)}</p>
      <p><strong>${escapeHtml(molecule.sdk_link)}</strong></p>
      <p class="muted">${escapeHtml(molecule.pae_note || "")}</p>
    `;

    const actions = document.createElement("div");
    actions.className = "button-row";
    actions.appendChild(actionButton("Open PNG", () => openPath(molecule.image_path, "biology-status")));
    actions.appendChild(actionButton("Open mmCIF", () => openPath(molecule.structure_path, "biology-status")));
    if (molecule.pae_path) {
      actions.appendChild(
        actionButton(molecule.pae_exists ? "Open PAE" : "PAE not generated", () => openPath(molecule.pae_path, "biology-status"), !molecule.pae_exists)
      );
    }
    body.appendChild(actions);
    card.appendChild(body);
    list.appendChild(card);
  }
}

function actionButton(label, handler, disabled = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.disabled = disabled;
  button.addEventListener("click", handler);
  return button;
}

function renderDiagnostics(payload) {
  const grid = $("diagnostics-grid");
  grid.replaceChildren();
  grid.appendChild(statusPill("good", `SDK ${payload.sdk_version}`));
  grid.appendChild(statusPill("good", `Python ${payload.python_version}`));
  grid.appendChild(statusPill(payload.ollama_on_path ? "good" : "warn", payload.ollama_on_path ? "Ollama found" : "Ollama not on PATH"));
  const modules = payload.optional_modules || {};
  for (const [name, available] of Object.entries(modules)) {
    grid.appendChild(statusPill(available ? "good" : "warn", `${name}: ${available ? "ready" : "missing"}`));
  }
  if ((payload.recommended_checks || []).length) {
    const note = document.createElement("div");
    note.className = "diagnostic-note";
    note.innerHTML = `
      <strong>Recommended checks</strong>
      <ul>${payload.recommended_checks.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    `;
    grid.appendChild(note);
  }
}

function statusPill(kind, text) {
  const pill = document.createElement("div");
  pill.className = `status-pill ${kind}`;
  pill.textContent = text;
  return pill;
}

async function loadWorkflows() {
  setText("run-status", "Loading curated SDK workflows...");
  try {
    const payload = await call("list_workflows");
    workflows = payload.workflows || [];
    renderWorkflows();
    setText("run-status", `Loaded ${workflows.length} workflows.`);
  } catch (error) {
    setText("run-status", errorMessage(error));
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
    lastRun = result;
    setText("run-status", result.summary || pretty(result));
    if (result.results_csv) {
      $("csv-path").value = result.results_csv;
      await previewCsv();
    }
    await loadHistory();
  } catch (error) {
    setText("run-status", errorMessage(error));
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
    setText("run-status", errorMessage(error));
  }
}

async function loadHistory() {
  const outputDir = $("output-dir").value.trim();
  if (!outputDir) {
    setText("run-status", "Output folder is required before loading history.");
    return;
  }
  try {
    const payload = await call("run_history", { outputDir, limit: 25 });
    renderHistory(payload.history || []);
  } catch (error) {
    setText("run-status", String(error));
  }
}

function renderHistory(entries) {
  const container = $("history-list");
  container.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No previous runs found for this output folder.";
    container.appendChild(empty);
    return;
  }
  for (const entry of entries) {
    const item = document.createElement("article");
    item.className = "history-item";
    item.innerHTML = `
      <div>
        <strong>${escapeHtml(entry.workflow_title || entry.preset_name || "IINTS run")}</strong>
        <span>${escapeHtml(entry.timestamp_utc || "")}</span>
      </div>
      <p>${escapeHtml(entry.output_dir || "")}</p>
    `;
    if (entry.results_csv) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Preview";
      button.addEventListener("click", async () => {
        $("csv-path").value = entry.results_csv;
        lastRun = {
          output_dir: entry.output_dir,
          results_csv: entry.results_csv,
          report_pdf: entry.report_pdf || null
        };
        await previewCsv();
      });
      item.appendChild(button);
    }
    container.appendChild(item);
  }
}

async function certifyMdmp() {
  const csv = $("csv-path").value.trim();
  if (!csv) {
    setText("mdmp-status", "Load or run a results CSV first.");
    return;
  }
  setText("mdmp-status", "Creating MDMP certificate using the standard diabetes contract...");
  try {
    const payload = await call("certify_mdmp", { csv, quickRows: 5000, full: false });
    lastMdmp = payload;
    setText(
      "mdmp-status",
      [
        `Grade: ${payload.grade}`,
        `Compliance score: ${payload.compliance_score}`,
        `Rows reviewed: ${payload.row_count}`,
        `Certificate: ${payload.certificate_path}`,
        `Report: ${payload.report_path}`,
        `Public key: ${payload.public_key_path}`
      ].join("\n")
    );
  } catch (error) {
    setText("mdmp-status", errorMessage(error));
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
    setText("ai-status", errorMessage(error));
  }
}

async function startAi() {
  setText("ai-status", "Starting Ollama and preparing the selected local model...");
  try {
    const payload = await call("start_local_ai", {
      model: $("ai-model").value.trim(),
      host: $("ai-host").value.trim(),
      noPull: false
    });
    setText("ai-status", pretty(payload));
  } catch (error) {
    setText("ai-status", errorMessage(error));
  }
}

async function listAiModels() {
  setText("ai-status", "Listing local Ollama models...");
  try {
    const payload = await call("list_local_ai_models", { host: $("ai-host").value.trim() });
    setText("ai-status", pretty(payload));
  } catch (error) {
    setText("ai-status", errorMessage(error));
  }
}

async function askAi() {
  const question = $("ai-question").value.trim();
  if (!question) {
    setText("ai-answer", "Write a question first.");
    return;
  }
  const csv = $("csv-path").value.trim();
  setText("ai-answer", "Running local AI analysis. This can take a while on small machines...");
  try {
    const payload = await call("ask_local_ai", {
      question,
      model: $("ai-model").value.trim(),
      host: $("ai-host").value.trim(),
      csv: csv || null
    });
    setText(
      "ai-answer",
      [
        `Model: ${payload.model}`,
        `CSV context used: ${payload.context_used ? "yes" : "no"}`,
        `Policy guard: ${payload.policy_action || ((payload.policy_violations || []).length ? "blocked" : "clear")}`,
        ...(payload.policy_violations || []).map((violation) => `- ${violation}`),
        ...(payload.policy_warnings || []).map((warning) => `- warning: ${warning}`),
        "",
        payload.answer
      ].join("\n")
    );
  } catch (error) {
    setText("ai-answer", errorMessage(error));
  }
}

async function openPath(path, statusId = "run-status") {
  if (!path) {
    setText(statusId, "Nothing to open yet.");
    return;
  }
  try {
    await call("open_path", { path });
  } catch (error) {
    setText(statusId, errorMessage(error));
  }
}

async function openExternalUrl(url, statusId = "evidence-status") {
  if (!url) {
    setText(statusId, "No evidence URL available for this connector.");
    return;
  }
  try {
    await call("open_external_url", { url });
    setText(statusId, `Opened allowlisted evidence link:\n${url}`);
  } catch (error) {
    setText(statusId, errorMessage(error));
  }
}

async function openOutputFolder() {
  await openPath($("output-dir").value.trim(), "run-status");
}

async function openLatestRunFolder() {
  await openPath(lastRun?.output_dir, "run-status");
}

async function openLatestReport() {
  await openPath(lastRun?.report_pdf, "run-status");
}

async function openLoadedCsv() {
  await openPath($("csv-path").value.trim(), "run-status");
}

async function openLatestCertificate() {
  await openPath(lastMdmp?.certificate_path, "mdmp-status");
}

async function openStructuralFolder() {
  await openPath(`${$("output-dir").value.trim()}/structural`, "biology-status");
}

async function runGenomicsSimulation() {
  const gene = $("genomics-gene").value.trim() || "INSR";
  const variant = $("genomics-variant").value.trim();
  const outputDir = $("output-dir").value.trim();
  if (!variant) {
    setText("biology-status", "Variant is required, e.g. V938M.");
    return;
  }
  setText("biology-status", `Running genomics simulation for ${gene} ${variant}...`);
  setResearchBusy(true);
  try {
    const payload = await call("run_genomics_simulation", {
      gene,
      variant,
      outputDir,
      durationMinutes: 360
    });
    lastGenomics = payload;
    setText(
      "biology-status",
      [
        `Genomics simulation completed: ${gene} ${variant}`,
        `Plot: ${payload.html_path}`,
        `Description: ${payload.metadata?.desc || "n/a"}`,
        `Affinity scalar: ${payload.metadata?.scalar ?? "n/a"}`,
        "Research only: not a medical device."
      ].join("\n")
    );
  } catch (error) {
    setText("biology-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
  }
}

async function runTissueStressTest() {
  const outputDir = $("output-dir").value.trim();
  const musclePercent = Number.parseFloat($("tissue-muscle").value || "30");
  const liverPercent = Number.parseFloat($("tissue-liver").value || "100");
  setText("biology-status", `Running tissue stress test: muscle ${musclePercent}%, liver ${liverPercent}%...`);
  setResearchBusy(true);
  try {
    const payload = await call("run_tissue_stress", {
      musclePercent,
      liverPercent,
      outputDir
    });
    lastTissue = payload;
    setText(
      "biology-status",
      [
        "Tissue-specific resistance stress test completed.",
        `Plot: ${payload.html_path}`,
        `Muscle scalar: ${payload.metadata?.muscle}`,
        `Liver scalar: ${payload.metadata?.liver}`,
        "Research only: not a medical device."
      ].join("\n")
    );
  } catch (error) {
    setText("biology-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
  }
}

async function openGenomicsPlot() {
  await openPath(lastGenomics?.html_path, "biology-status");
}

async function openTissuePlot() {
  await openPath(lastTissue?.html_path, "biology-status");
}

function setBusy(isBusy) {
  $("run-btn").disabled = isBusy;
  $("refresh-btn").disabled = isBusy;
  $("preview-btn").disabled = isBusy;
  $("history-btn").disabled = isBusy;
  $("mdmp-btn").disabled = isBusy;
  $("open-run-folder-btn").disabled = isBusy;
  $("open-report-btn").disabled = isBusy;
}

function setResearchBusy(isBusy) {
  $("genomics-run-btn").disabled = isBusy;
  $("tissue-run-btn").disabled = isBusy;
  $("molecule-refresh-btn").disabled = isBusy;
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
$("history-btn").addEventListener("click", loadHistory);
$("diagnostics-btn").addEventListener("click", runDiagnostics);
$("open-output-btn").addEventListener("click", openOutputFolder);
$("update-refresh-btn").addEventListener("click", loadUpdateInfo);
$("update-download-btn").addEventListener("click", openAppDownloads);
$("update-docs-btn").addEventListener("click", openUpdateDocs);
$("update-copy-btn").addEventListener("click", copyUpdateCommand);
$("update-terminal-btn").addEventListener("click", openSdkUpdateTerminal);
$("open-run-folder-btn").addEventListener("click", openLatestRunFolder);
$("open-report-btn").addEventListener("click", openLatestReport);
$("preview-btn").addEventListener("click", previewCsv);
$("mdmp-btn").addEventListener("click", certifyMdmp);
$("open-csv-btn").addEventListener("click", openLoadedCsv);
$("open-certificate-btn").addEventListener("click", openLatestCertificate);
$("ai-start-btn").addEventListener("click", startAi);
$("ai-check-btn").addEventListener("click", checkAi);
$("ai-models-btn").addEventListener("click", listAiModels);
$("ai-ask-btn").addEventListener("click", askAi);
$("molecule-refresh-btn").addEventListener("click", loadMolecules);
$("open-structural-folder-btn").addEventListener("click", openStructuralFolder);
$("genomics-run-btn").addEventListener("click", runGenomicsSimulation);
$("genomics-open-btn").addEventListener("click", openGenomicsPlot);
$("tissue-run-btn").addEventListener("click", runTissueStressTest);
$("tissue-open-btn").addEventListener("click", openTissuePlot);
$("evidence-refresh-btn").addEventListener("click", loadEvidenceConnectors);

await loadStatus();
await runDiagnostics();
await loadUpdateInfo();
await loadWorkflows();
await loadHistory();
await loadMolecules();
await loadEvidenceConnectors();
