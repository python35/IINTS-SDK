import {
  clarkeErrorGrid,
  coefficientOfVariation,
  drawSyntheticBanner,
  glycemicBands,
  seededRandom,
} from "./science.js";
import { LIVER_CARD_CAPTION, interpolateSeries, lookupEquation } from "./digital-twin-data.js";

const tauriCore = window.__TAURI__?.core;
const invoke = tauriCore?.invoke;
const nativeOpenDialog = window.__TAURI__?.dialog?.open;
const isNativeDesktop = typeof invoke === "function";
const COPYABLE_CONTEXT_SELECTOR = [
  "input",
  "textarea",
  "pre",
  "code",
  "table",
  ".metric strong",
  ".status-box",
  ".ai-answer",
  "[data-copyable]"
].join(", ");

let selectedWorkflow = null;
let workflows = [];
let lastPreview = null;
let lastRun = null;
let lastMdmp = null;
let lastAcademicBundle = null;
let molecules = [];
let evidenceConnectors = [];
let updateInfo = null;
let lastGenomics = null;
let lastTissue = null;
let lastMechanistic = null;
let lastCopasi = null;
let lastCellml = null;
let lastFmi = null;
let lastBinding = null;
let appInfo = null;
let runBusy = false;
let activeWorkflowJob = null;
let researchBusy = false;
let aiBusy = false;
let engineCompatible = false;
const REQUIRED_BRIDGE_API_VERSION = 4;
const ESSENTIAL_RESULT_COLUMNS = new Set([
  "time_minutes",
  "timestamp",
  "glucose_actual_mgdl",
  "glucose_to_algo_mgdl",
  "glucose_trend_mgdl_min",
  "predicted_glucose_30min",
  "delivered_insulin_units",
  "carb_intake_grams",
  "patient_iob_units",
  "patient_cob_grams",
  "sensor_status",
  "pump_status",
  "predictor_used",
  "predictor_uncertainty_std_mgdl",
  "safety_level",
  "safety_triggered",
  "safety_reason"
]);
const moleculeViewer = {
  molecule: null,
  rotationX: -0.35,
  rotationY: 0.55,
  zoom: 1,
  dragging: false,
  pointerX: 0,
  pointerY: 0,
  autoRotate: false,
  animationFrame: null
};

const SETTINGS_STORAGE_KEY = "iints-af.workbench.settings.v1";
const DEFAULT_SETTINGS = Object.freeze({
  outputDir: "~/IINTS-Tauri-Runs",
  seed: 42,
  aiModel: "ministral-3:8b",
  aiHost: "http://127.0.0.1:11434",
  autoDiagnostics: true
});
let workbenchSettings = { ...DEFAULT_SETTINGS };

const $ = (id) => document.getElementById(id);

const VIEW_METADATA = {
  overview: {
    eyebrow: "Workspace",
    title: "System overview",
    description: "Check the local research engine and optional tooling before starting an experiment."
  },
  settings: {
    eyebrow: "Application",
    title: "Settings",
    description: "Configure local defaults, maintain the SDK and desktop app, and open documentation."
  },
  run: {
    eyebrow: "Experiment",
    title: "Protocols and runs",
    description: "Choose a curated protocol, fix the random seed, and execute it through the Python SDK."
  },
  results: {
    eyebrow: "Analysis",
    title: "Results review",
    description: "Inspect generated metrics, glucose trajectories, tabular output, and MDMP evidence."
  },
  reproducibility: {
    eyebrow: "Research record",
    title: "Reproducibility package",
    description: "Create reviewable metadata, checksums, evidence references, and a conservative audit trail."
  },
  ai: {
    eyebrow: "Local assistant",
    title: "AI-supported interpretation",
    description: "Use a local Ollama model to summarize outputs; deterministic SDK results remain authoritative."
  },
  research: {
    eyebrow: "Methods",
    title: "Cross-scale research labs",
    description: "Inspect biological, mechanistic, and physical evidence without silently changing patient parameters."
  },
  foundation: {
    eyebrow: "Representation research",
    title: "CGM representation laboratory",
    description: "Train and inspect the independent GlucoFM reproduction, then compare traceable evaluation artifacts produced under one shared benchmark contract."
  },
  eucys: {
    eyebrow: "Publication dossier",
    title: "Scientific Portfolio & Dossier",
    description: "Browse 11 publication-grade scientific figures, Clarke Error Grids, TIR distributions, Stem-Cell Islet kinetics, and Jetson hardware latency."
  },
  evidence: {
    eyebrow: "Provenance",
    title: "Evidence connectors",
    description: "Review official scientific resources, integration boundaries, and locally generated evidence artifacts."
  }
};

const loadedViews = new Set(["overview"]);

async function loadViewData(view) {
  if (loadedViews.has(view)) return;
  loadedViews.add(view);
  if (view === "run") {
    await Promise.allSettled([loadWorkflows(), loadHistory()]);
  } else if (view === "ai") {
    await listAiModels();
  } else if (view === "foundation" || view === "eucys") {
    renderFoundationChart(activeChartTab);
  } else if (view === "research") {
    await Promise.allSettled([loadMolecules(), loadMechanisticStatus(), loadCrossScaleStatus()]);
  } else if (view === "evidence") {
    await loadEvidenceConnectors();
  } else if (view === "settings") {
    await Promise.allSettled([loadUpdateInfo(), loadAppInfo()]);
  }
}

function setActiveView(view, focusHeading = true) {
  const metadata = VIEW_METADATA[view];
  if (!metadata) return;

  const effectiveView = view === "eucys" ? "foundation" : view;
  document.querySelectorAll("[data-view-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.viewPanel !== effectiveView;
  });
  // The digital twin's render loop keeps running via requestAnimationFrame
  // even while its panel is set `hidden` (not unmounted); stop it when the
  // user navigates away, the same way closeMoleculeViewer() cancels its own
  // animation frame rather than leaving it spinning in the background.
  if (effectiveView !== "results") digitalTwin?.pause();
  document.querySelectorAll("[data-view]").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("is-active", active);
    if (active) {
      button.setAttribute("aria-current", "page");
    } else {
      button.removeAttribute("aria-current");
    }
  });

  setText("view-eyebrow", metadata.eyebrow);
  setText("view-title", metadata.title);
  setText("view-description", metadata.description);
  document.title = `${metadata.title} | IINTS-AF`;
  if (focusHeading) {
    window.scrollTo({ top: 0, behavior: "auto" });
  }
  void loadViewData(view);
}

function initializeNavigation() {
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => setActiveView(button.dataset.view));
  });
  setActiveView("overview", false);
}

function initializeNativeInteractionPolicy() {
  document.documentElement.classList.toggle("native-desktop", isNativeDesktop);
  if (!isNativeDesktop) return;

  document.addEventListener("contextmenu", (event) => {
    const target = event.target;
    if (target instanceof Element && target.closest(COPYABLE_CONTEXT_SELECTOR)) return;
    event.preventDefault();
  });

  document.addEventListener("dragstart", (event) => {
    const target = event.target;
    if (target instanceof HTMLImageElement || target instanceof HTMLButtonElement) {
      event.preventDefault();
    }
  });
}

function setText(id, value) {
  $(id).textContent = value;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function errorMessage(error) {
  return String(error).replace(/^Error:\s*/i, "").trim();
}

function readStoredSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem(SETTINGS_STORAGE_KEY) || "{}");
    return {
      outputDir: typeof stored.outputDir === "string" && stored.outputDir.trim()
        ? stored.outputDir.trim()
        : DEFAULT_SETTINGS.outputDir,
      seed: Number.isInteger(stored.seed) && stored.seed >= 0 && stored.seed <= 2147483647
        ? stored.seed
        : DEFAULT_SETTINGS.seed,
      aiModel: typeof stored.aiModel === "string" && stored.aiModel.trim()
        ? stored.aiModel.trim()
        : DEFAULT_SETTINGS.aiModel,
      aiHost: isAllowedLocalAiHost(stored.aiHost) ? stored.aiHost : DEFAULT_SETTINGS.aiHost,
      autoDiagnostics: typeof stored.autoDiagnostics === "boolean"
        ? stored.autoDiagnostics
        : DEFAULT_SETTINGS.autoDiagnostics
    };
  } catch (_error) {
    return { ...DEFAULT_SETTINGS };
  }
}

function isAllowedLocalAiHost(value) {
  if (typeof value !== "string" || !value.trim()) return false;
  try {
    const url = new URL(value.trim());
    const localHosts = new Set(["127.0.0.1", "localhost", "[::1]"]);
    return ["http:", "https:"].includes(url.protocol)
      && localHosts.has(url.hostname)
      && !url.username
      && !url.password
      && (url.pathname === "" || url.pathname === "/")
      && !url.search
      && !url.hash;
  } catch (_error) {
    return false;
  }
}

function applyWorkbenchSettings(settings) {
  workbenchSettings = { ...settings };
  $("output-dir").value = settings.outputDir;
  $("seed").value = String(settings.seed);
  $("ai-model").value = settings.aiModel;
  $("ai-host").value = settings.aiHost;
  $("settings-output-dir").value = settings.outputDir;
  $("settings-seed").value = String(settings.seed);
  $("settings-ai-model").value = settings.aiModel;
  $("settings-ai-host").value = settings.aiHost;
  $("settings-auto-diagnostics").checked = settings.autoDiagnostics;
}

function collectSettingsForm() {
  const outputDir = $("settings-output-dir").value.trim();
  const seed = Number.parseInt($("settings-seed").value, 10);
  const aiModel = $("settings-ai-model").value.trim();
  const aiHost = $("settings-ai-host").value.trim();
  if (!outputDir) throw new Error("Default output folder is required.");
  if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) {
    throw new Error("Seed must be an integer between 0 and 2147483647.");
  }
  if (!aiModel) throw new Error("Default Ollama model is required.");
  if (!isAllowedLocalAiHost(aiHost)) {
    throw new Error("Ollama host must be a local http(s) URL using localhost, 127.0.0.1, or ::1.");
  }
  return {
    outputDir,
    seed,
    aiModel,
    aiHost,
    autoDiagnostics: $("settings-auto-diagnostics").checked
  };
}

function saveWorkbenchSettings() {
  try {
    const settings = collectSettingsForm();
    const outputChanged = settings.outputDir !== workbenchSettings.outputDir;
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    applyWorkbenchSettings(settings);
    if (outputChanged) lastRun = null;
    refreshActionAvailability();
    setText("settings-status", "Settings saved locally and applied to this workbench session.");
  } catch (error) {
    setText("settings-status", errorMessage(error));
  }
}

function resetWorkbenchSettings() {
  try {
    localStorage.removeItem(SETTINGS_STORAGE_KEY);
  } catch (_error) {
    // The defaults still apply when browser storage is unavailable.
  }
  applyWorkbenchSettings({ ...DEFAULT_SETTINGS });
  lastRun = null;
  refreshActionAvailability();
  setText("settings-status", "Default settings restored and applied.");
}

function initializeSettings() {
  const settings = readStoredSettings();
  applyWorkbenchSettings(settings);
  return settings;
}

async function call(command, args = {}) {
  if (!invoke) {
    throw new Error("The native desktop bridge is unavailable. Open the installed IINTS-AF app instead of this browser preview.");
  }
  return await invoke(command, args);
}

function dialogDefaultPath(value) {
  const candidate = String(value || "").trim();
  if (/^(?:\/|[A-Za-z]:[\\/]|\\\\)/.test(candidate)) return candidate;
  return undefined;
}

async function chooseLocalPath({
  inputId,
  buttonId,
  statusId,
  title,
  directory = false,
  filters = [],
  selectedLabel
}) {
  if (typeof nativeOpenDialog !== "function") {
    setText(
      statusId,
      "The native file chooser is unavailable in this preview. Open the installed desktop application."
    );
    return;
  }

  const input = $(inputId);
  const button = $(buttonId);
  button.disabled = true;
  try {
    const options = {
      title,
      directory,
      multiple: false
    };
    const defaultPath = dialogDefaultPath(input.value);
    if (defaultPath) options.defaultPath = defaultPath;
    if (!directory && filters.length) options.filters = filters;

    const selected = await nativeOpenDialog(options);
    const path = Array.isArray(selected) ? selected[0] : selected;
    if (typeof path !== "string" || !path.trim()) return;
    const allowedExtensions = filters
      .flatMap((filter) => filter.extensions || [])
      .map((extension) => String(extension).toLowerCase());
    const selectedExtension = path.includes(".") ? path.split(".").pop().toLowerCase() : "";
    if (!directory && allowedExtensions.length && !allowedExtensions.includes(selectedExtension)) {
      input.setAttribute("aria-invalid", "true");
      setText(
        statusId,
        `Unsupported file type. Choose one of: ${allowedExtensions.map((extension) => `.${extension}`).join(", ")}.`
      );
      return;
    }

    input.value = path;
    input.title = path;
    input.setAttribute("aria-invalid", "false");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    setText(statusId, `${selectedLabel}:\n${path}`);
  } catch (error) {
    setText(statusId, `Could not open the native selector.\n${errorMessage(error)}`);
  } finally {
    refreshActionAvailability();
  }
}

async function chooseSettingsOutputFolder() {
  await chooseLocalPath({
    inputId: "settings-output-dir",
    buttonId: "settings-output-browse-btn",
    statusId: "settings-status",
    title: "Choose the default IINTS-AF output folder",
    directory: true,
    selectedLabel: "Default output folder selected"
  });
}

async function chooseRunOutputFolder() {
  await chooseLocalPath({
    inputId: "output-dir",
    buttonId: "output-browse-btn",
    statusId: "run-status",
    title: "Choose an IINTS-AF output folder",
    directory: true,
    selectedLabel: "Run output folder selected"
  });
}

async function chooseResultsCsv() {
  await chooseLocalPath({
    inputId: "csv-path",
    buttonId: "csv-browse-btn",
    statusId: "results-status",
    title: "Choose an IINTS-AF results CSV",
    filters: [{ name: "CSV results", extensions: ["csv"] }],
    selectedLabel: "Results CSV selected"
  });
}

async function chooseAcademicRunFolder() {
  await chooseLocalPath({
    inputId: "academic-run-dir",
    buttonId: "academic-run-browse-btn",
    statusId: "academic-status",
    title: "Choose a completed IINTS-AF run folder",
    directory: true,
    selectedLabel: "Completed run folder selected"
  });
}

async function chooseResearchModel(inputId, buttonId, statusId, title, name, extensions) {
  await chooseLocalPath({
    inputId,
    buttonId,
    statusId,
    title,
    filters: [{ name, extensions }],
    selectedLabel: `${name} selected`
  });
}

function workflowCard(workflow) {
  const card = document.createElement("article");
  card.className = "workflow-card";
  card.tabIndex = 0;
  card.setAttribute("role", "button");
  card.setAttribute("aria-pressed", "false");
  card.dataset.key = workflow.key;
  card.innerHTML = `
    <h3>${escapeHtml(workflow.title)}</h3>
    <p class="workflow-meta">${escapeHtml(workflow.audience)} · ${escapeHtml(workflow.preset_name)}</p>
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
    const selected = card.dataset.key === selectedWorkflow;
    card.classList.toggle("selected", selected);
    card.setAttribute("aria-pressed", selected ? "true" : "false");
  });
}

function selectWorkflow(key) {
  selectedWorkflow = key;
  markSelectedWorkflow();
  refreshActionAvailability();
}

async function loadStatus() {
  try {
    const status = await call("desktop_status");
    const bridgeApiVersion = Number(status.bridge_api_version || 0);
    engineCompatible = bridgeApiVersion >= REQUIRED_BRIDGE_API_VERSION;
    $("sdk-status").textContent = engineCompatible
      ? `SDK ${status.sdk_version} via ${status.python_executable}`
      : `SDK ${status.sdk_version} needs a compatibility update`;
    $("sdk-status-dot").className = engineCompatible ? "status-dot ok" : "status-dot warn";
    $("install-engine-btn").hidden = engineCompatible;
    setText("settings-sdk-version", status.sdk_version || "Unknown");
    setText("settings-python-path", status.python_executable || "Python path unavailable");
    setText(
      "run-status",
      engineCompatible
        ? "Python SDK bridge ready.\nSelect a workflow and run it."
        : [
            "Python SDK update required before protocols can run.",
            `Installed bridge API: ${bridgeApiVersion || "legacy"}; required: ${REQUIRED_BRIDGE_API_VERSION}.`,
            "Open Settings and choose Install or update Python SDK, then refresh versions."
          ].join("\n")
    );
    refreshActionAvailability();
  } catch (error) {
    engineCompatible = false;
    $("sdk-status").textContent = "Python bridge unavailable";
    $("sdk-status-dot").className = "status-dot error";
    $("install-engine-btn").hidden = false;
    setText("settings-sdk-version", "Unavailable");
    setText("settings-python-path", errorMessage(error));
    setText("run-status", errorMessage(error));
    refreshActionAvailability();
  }
}

async function runDiagnostics() {
  const grid = $("diagnostics-grid");
  grid.replaceChildren(diagnosticRow("Diagnostics", "Running...", "info"));
  try {
    const payload = await call("desktop_diagnostics");
    renderDiagnostics(payload);
  } catch (error) {
    grid.replaceChildren(diagnosticRow("Diagnostics", errorMessage(error), "bad"));
  }
}

function humanVersionStatus(status) {
  const labels = {
    current: "up to date",
    update_available: "update available",
    ahead: "newer than published release",
    development: "development checkout",
    unknown: "not verified"
  };
  return labels[status] || String(status || "not verified").replaceAll("_", " ");
}

async function loadUpdateInfo(refresh = false) {
  setText("update-status", "Checking SDK/app update information...");
  try {
    updateInfo = await call("desktop_update_info", { refresh });
    setText("settings-sdk-version", updateInfo.current_version || "Unknown");
    setText(
      "settings-sdk-latest",
      `Latest stable: ${updateInfo.latest_version || "not verified"} · ${humanVersionStatus(updateInfo.sdk_status)}`
    );
    setText(
      "settings-app-latest",
      `Latest beta: ${updateInfo.app_latest_version || "not verified"} · ${humanVersionStatus(updateInfo.app_status)}`
    );
    $("update-terminal-btn").textContent = updateInfo.sdk_update_available === true
      ? "Update Python SDK"
      : updateInfo.sdk_update_available === false
        ? "Repair or reinstall Python SDK"
        : "Check or update Python SDK";
    $("update-download-btn").textContent = updateInfo.app_update_available === true
      ? "Download app update"
      : "Open app downloads";
    setText("settings-python-path", updateInfo.python_executable || "Python path unavailable");
    const sdkWarning = updateInfo.sdk_check_error
      ? `SDK check warning: ${updateInfo.sdk_check_error}`
      : "";
    const appWarning = updateInfo.app_check_error
      ? `App check warning: ${updateInfo.app_check_error}`
      : "";
    const metadataWarning = updateInfo.version_metadata_matches_code === false
      ? `Environment warning: package metadata reports ${updateInfo.current_version}, but active code reports ${updateInfo.active_code_version}. Repair the Python SDK environment.`
      : "";
    const warnings = [sdkWarning, appWarning, metadataWarning].filter(Boolean);
    setText(
      "update-status",
      [
        `Installed SDK: ${updateInfo.current_version || "unknown"}`,
        `Active SDK code: ${updateInfo.active_code_version || "unknown"}`,
        `Latest stable SDK: ${updateInfo.latest_version || "not verified"}`,
        `SDK status: ${humanVersionStatus(updateInfo.sdk_status)} (${updateInfo.sdk_check_source || "unknown source"})`,
        `Installed app: ${updateInfo.app_current_version || appInfo?.app_version || "unknown"}`,
        `Latest app beta: ${updateInfo.app_latest_version || "not verified"}`,
        `App status: ${humanVersionStatus(updateInfo.app_status)} (${updateInfo.app_check_source || "unknown source"})`,
        `Python: ${updateInfo.python_executable || "unknown"}`,
        `Package: ${updateInfo.package_spec}`,
        ...(warnings.length ? ["", ...warnings] : []),
        "",
        "SDK update command:",
        updateInfo.pip_command,
        "",
        "Use Open app downloads for .exe/.dmg/Linux bundles."
      ].join("\n")
    );
  } catch (error) {
    setText("settings-sdk-latest", "Latest release could not be checked");
    setText("settings-app-latest", "Latest release could not be checked");
    setText(
      "update-status",
      `${errorMessage(error)}\n\nUse 'Install or update Python SDK' to create or repair the private app engine.`
    );
  }
}

async function loadAppInfo() {
  try {
    appInfo = await call("desktop_app_info");
    setText("settings-app-version", appInfo.app_version || "Unknown");
    setText(
      "settings-app-platform",
      `${appInfo.platform || "native"} · ${appInfo.architecture || "unknown architecture"}`
    );
  } catch (error) {
    setText("settings-app-version", "Unavailable");
    setText("settings-app-platform", errorMessage(error));
  }
}

async function refreshSoftwareVersions() {
  setText("update-status", "Refreshing app and Python-engine compatibility...");
  // Python imports can touch large research environments. Run bridge checks
  // sequentially so removable disks and first-start caches cannot deadlock startup.
  await loadAppInfo();
  await loadStatus();
  await loadUpdateInfo(true);
  await runDiagnostics();
}

async function openAppDownloads() {
  const url = updateInfo?.app_download_url || appInfo?.release_url || "https://github.com/python35/IINTS-SDK/releases/tag/tauri-beta-latest";
  await openExternalUrl(url, "update-status");
}

async function openUpdateDocs() {
  const url = updateInfo?.update_docs_url || "https://python35.github.io/IINTS-SDK/APP_INSTALL/";
  await openExternalUrl(url, "update-status");
}

async function openUserGuide() {
  await openExternalUrl(
    "https://python35.github.io/IINTS-SDK/RESEARCH_WORKBENCH_GUIDE/",
    "sdk-status"
  );
}

async function openInstallGuide() {
  await openExternalUrl("https://python35.github.io/IINTS-SDK/APP_INSTALL/", "settings-status");
}

async function openDocsHome() {
  await openExternalUrl("https://python35.github.io/IINTS-SDK/", "settings-status");
}

async function openProjectWebsite() {
  await openExternalUrl("https://iints.org/", "settings-status");
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
  setText("update-status", "Opening a terminal to install or update the private Python SDK engine...");
  try {
    await call("open_sdk_update_terminal");
    setText(
      "update-status",
      [
        "Python engine maintenance terminal launched.",
        "If no engine exists, the app creates ~/.iints-af/python-engine and installs the SDK there.",
        "If an engine already exists, the fixed Rust-owned command updates it.",
        "When the terminal reports completion, choose Refresh versions. Restarting the app is not normally required.",
        "",
        updateInfo?.pip_command || "The installed SDK version will appear after maintenance completes."
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
    const payload = await call("list_molecule_assets", {
      outputDir: $("output-dir").value.trim()
    });
    molecules = payload.molecules || [];
    renderMolecules(molecules);
    setText(
      "biology-status",
      `Loaded ${molecules.length} bundled structures. Select View 3D for local inspection or generate a PAE artifact from AlphaFold evidence.`
    );
  } catch (error) {
    list.replaceChildren(statusPill("bad", errorMessage(error)));
    setText("biology-status", errorMessage(error));
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
    const integrated = evidenceConnectors.filter((item) => item.integration_level === "integrated").length;
    const partial = evidenceConnectors.filter((item) => item.integration_level === "partial").length;
    setText(
      "evidence-status",
      `Loaded ${evidenceConnectors.length} curated resources: ${integrated} integrated and ${partial} partially integrated. Portal-only and planned resources are labelled explicitly; remote pages open in your browser.`
    );
  } catch (error) {
    list.replaceChildren(statusPill("bad", errorMessage(error)));
    setText("evidence-status", errorMessage(error));
  }
}

function renderEvidenceConnectors(items) {
  const list = $("evidence-list");
  list.replaceChildren();
  const query = $("evidence-search").value.trim().toLowerCase();
  const level = $("evidence-level").value;
  const visibleItems = items.filter((connector) => {
    if (level !== "all" && connector.integration_level !== level) return false;
    if (!query) return true;
    return [
      connector.title,
      connector.key,
      connector.category,
      connector.why_it_matters,
      connector.workbench_use,
      connector.integration_status
    ].some((value) => String(value || "").toLowerCase().includes(query));
  });
  setText("evidence-status", `Showing ${visibleItems.length} of ${items.length} curated evidence resources.`);
  if (!visibleItems.length) {
    list.appendChild(statusPill(
      "warn",
      items.length ? "No evidence resources match the current filters." : "No evidence connectors returned by the SDK bridge."
    ));
    return;
  }
  for (const connector of visibleItems) {
    const card = document.createElement("article");
    card.className = "evidence-card";
    card.innerHTML = `
      <div class="connector-meta">
        <span>${escapeHtml(connector.category || "Evidence")}</span>
        <span class="connector-status ${escapeHtml(connector.integration_level || "portal")}">${escapeHtml(connector.integration_status || "Connector")}</span>
      </div>
      <h3>${escapeHtml(connector.title || connector.key || "Evidence connector")}</h3>
      <p>${escapeHtml(connector.why_it_matters || "")}</p>
      <p><strong>Workbench use:</strong> ${escapeHtml(connector.app_use || "")}</p>
      <p><strong>Access:</strong> ${escapeHtml(connector.access_mode || "Official portal")}</p>
      <p><strong>Local evidence:</strong> ${connector.writes_local_evidence ? "writes a reviewable artifact" : "no automatic local evidence artifact"}</p>
      <p class="muted">${escapeHtml(connector.provenance_note || "External evidence must be reviewed before it supports a scientific claim.")}</p>
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
    actions.appendChild(
      moleculeActionButton(
        "View 3D",
        () => openMoleculeViewer(molecule),
        !molecule.backbone?.atoms?.length
      )
    );
    actions.appendChild(
      moleculeActionButton(
        "Open PNG",
        () => openPath(molecule.image_path, "biology-status"),
        !molecule.image_exists
      )
    );
    actions.appendChild(
      moleculeActionButton(
        "Reveal mmCIF",
        () => revealPath(molecule.structure_path, "biology-status"),
        !molecule.structure_exists
      )
    );
    if (molecule.pae_path) {
      actions.appendChild(
        moleculeActionButton(
          molecule.pae_exists ? "Open PAE" : "Generate PAE",
          () => molecule.pae_exists
            ? openPath(molecule.pae_path, "biology-status")
            : generateMoleculePae(molecule),
          !molecule.pae_target
        )
      );
    }
    actions.appendChild(
      moleculeActionButton(
        "AlphaFold entry",
        () => openExternalUrl(molecule.alphafold_url, "biology-status"),
        !molecule.alphafold_url
      )
    );
    if (molecule.structure_error) {
      const warning = document.createElement("p");
      warning.className = "inline-warning";
      warning.textContent = `3D viewer unavailable: ${molecule.structure_error}`;
      body.appendChild(warning);
    }
    body.appendChild(actions);
    card.appendChild(body);
    list.appendChild(card);
  }
}

function confidenceColor(confidence) {
  if (confidence === null || confidence === undefined || confidence === "") {
    return "#687780";
  }
  const value = Number(confidence);
  if (!Number.isFinite(value)) return "#687780";
  if (value >= 90) return "#1f3b8f";
  if (value >= 70) return "#41b6c4";
  if (value >= 50) return "#f0c94d";
  return "#d96b31";
}

function resetMoleculeViewer() {
  moleculeViewer.rotationX = -0.35;
  moleculeViewer.rotationY = 0.55;
  moleculeViewer.zoom = 1;
  drawMoleculeViewer();
}

function openMoleculeViewer(molecule) {
  if (!molecule.backbone?.atoms?.length) {
    setText(
      "biology-status",
      molecule.structure_error || `No readable C-alpha backbone is available for ${molecule.title}.`
    );
    return;
  }
  moleculeViewer.molecule = molecule;
  const panel = $("molecule-viewer-panel");
  panel.hidden = false;
  setText("molecule-viewer-title", molecule.title);
  setText(
    "molecule-viewer-meta",
    `UniProt ${molecule.uniprot_id} · ${molecule.backbone.atoms.length} C-alpha atoms · ${molecule.backbone.chain_count} chain(s)`
  );
  resetMoleculeViewer();
  $("molecule-viewer-canvas").focus({ preventScroll: true });
  panel.scrollIntoView({ block: "start", behavior: "auto" });
  setText(
    "biology-status",
    `Inspecting ${molecule.title} locally. AlphaFold confidence is structural evidence only and is not converted into a physiological effect.`
  );
  refreshActionAvailability();
}

function closeMoleculeViewer() {
  setMoleculeAutoRotate(false);
  moleculeViewer.molecule = null;
  $("molecule-viewer-panel").hidden = true;
  refreshActionAvailability();
}

function setMoleculeAutoRotate(enabled) {
  moleculeViewer.autoRotate = Boolean(enabled);
  const button = $("molecule-viewer-rotate-btn");
  button.setAttribute("aria-pressed", String(moleculeViewer.autoRotate));
  button.textContent = `Auto-rotate: ${moleculeViewer.autoRotate ? "on" : "off"}`;
  if (moleculeViewer.autoRotate && moleculeViewer.animationFrame === null) {
    moleculeViewer.animationFrame = requestAnimationFrame(animateMoleculeViewer);
  } else if (!moleculeViewer.autoRotate && moleculeViewer.animationFrame !== null) {
    cancelAnimationFrame(moleculeViewer.animationFrame);
    moleculeViewer.animationFrame = null;
  }
}

function animateMoleculeViewer() {
  moleculeViewer.animationFrame = null;
  if (!moleculeViewer.autoRotate || !moleculeViewer.molecule) return;
  moleculeViewer.rotationY += 0.008;
  drawMoleculeViewer();
  moleculeViewer.animationFrame = requestAnimationFrame(animateMoleculeViewer);
}

function projectedBackbone(canvas) {
  const backbone = moleculeViewer.molecule?.backbone;
  if (!backbone?.atoms?.length) return [];
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, rect.width);
  const height = Math.max(260, rect.height);
  const center = backbone.center || [0, 0, 0];
  const radius = Math.max(1, Number(backbone.radius) || 1);
  const cosY = Math.cos(moleculeViewer.rotationY);
  const sinY = Math.sin(moleculeViewer.rotationY);
  const cosX = Math.cos(moleculeViewer.rotationX);
  const sinX = Math.sin(moleculeViewer.rotationX);
  const scale = Math.min(width, height) * 0.4 * moleculeViewer.zoom / radius;

  return backbone.atoms.map((atom) => {
    const x = Number(atom.x) - Number(center[0]);
    const y = Number(atom.y) - Number(center[1]);
    const z = Number(atom.z) - Number(center[2]);
    const rotatedX = x * cosY - z * sinY;
    const firstZ = x * sinY + z * cosY;
    const rotatedY = y * cosX - firstZ * sinX;
    const rotatedZ = y * sinX + firstZ * cosX;
    const perspective = 1 / Math.max(0.55, 1 + rotatedZ / (radius * 6));
    return {
      ...atom,
      screenX: width / 2 + rotatedX * scale * perspective,
      screenY: height / 2 - rotatedY * scale * perspective,
      depth: rotatedZ,
      pointRadius: Math.max(1.8, 2.7 * perspective)
    };
  });
}

function drawMoleculeViewer() {
  const canvas = $("molecule-viewer-canvas");
  if (!canvas || $("molecule-viewer-panel").hidden) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(320, Math.round(rect.width));
  const height = Math.max(260, Math.round(rect.height));
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#f7f9f9";
  context.fillRect(0, 0, width, height);

  const points = projectedBackbone(canvas);
  if (!points.length) {
    context.fillStyle = "#53636d";
    context.font = '14px "Avenir Next", "Segoe UI", sans-serif';
    context.fillText("No readable backbone data.", 18, 28);
    return;
  }

  context.lineWidth = 2;
  context.lineCap = "round";
  context.lineJoin = "round";
  for (let index = 1; index < points.length; index += 1) {
    const previous = points[index - 1];
    const current = points[index];
    if (
      previous.chain_id !== current.chain_id
      || Math.abs(Number(current.residue_index) - Number(previous.residue_index)) > 2
    ) {
      continue;
    }
    context.beginPath();
    context.moveTo(previous.screenX, previous.screenY);
    context.lineTo(current.screenX, current.screenY);
    const averageConfidence = previous.confidence !== null
      && previous.confidence !== undefined
      && current.confidence !== null
      && current.confidence !== undefined
      && Number.isFinite(Number(previous.confidence))
      && Number.isFinite(Number(current.confidence))
      ? (Number(previous.confidence) + Number(current.confidence)) / 2
      : null;
    context.strokeStyle = confidenceColor(averageConfidence);
    context.stroke();
  }

  const sortedPoints = [...points].sort((left, right) => left.depth - right.depth);
  for (const point of sortedPoints) {
    context.beginPath();
    context.arc(point.screenX, point.screenY, point.pointRadius, 0, Math.PI * 2);
    context.fillStyle = confidenceColor(point.confidence);
    context.fill();
  }

  context.fillStyle = "#53636d";
  context.font = '12px "SFMono-Regular", "Cascadia Code", monospace';
  context.fillText("C-alpha backbone · colour = pLDDT", 14, height - 14);
}

async function generateMoleculePae(molecule) {
  const outputDir = $("output-dir").value.trim();
  if (!outputDir) {
    setText("biology-status", "Choose an output folder before generating a PAE heatmap.");
    return;
  }
  setResearchBusy(true);
  setText("biology-status", `Downloading AlphaFold PAE evidence for ${molecule.title}...`);
  try {
    const payload = await call("generate_molecule_pae", {
      target: molecule.pae_target,
      outputDir
    });
    molecule.pae_path = payload.html_path;
    molecule.pae_exists = true;
    renderMolecules(molecules);
    setText(
      "biology-status",
      [
        `PAE heatmap generated for ${molecule.title}.`,
        `Artifact: ${payload.html_path}`,
        `Residues: ${payload.residue_count}; maximum reported PAE: ${payload.max_predicted_aligned_error}`,
        "This is structural prediction evidence only; it is not a dosing or physiological-severity metric."
      ].join("\n")
    );
    await openPath(payload.html_path, "biology-status");
  } catch (error) {
    setText("biology-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
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

function moleculeActionButton(label, handler, unavailable = false) {
  const button = actionButton(label, handler, unavailable || researchBusy);
  button.dataset.unavailable = String(Boolean(unavailable));
  return button;
}

function renderDiagnostics(payload) {
  const grid = $("diagnostics-grid");
  grid.replaceChildren();
  grid.appendChild(diagnosticRow("IINTS-AF SDK", payload.sdk_version, "good"));
  const bridgeApiVersion = Number(payload.bridge_api_version || 0);
  grid.appendChild(diagnosticRow(
    "Desktop bridge",
    bridgeApiVersion >= REQUIRED_BRIDGE_API_VERSION
      ? `Compatible (API ${bridgeApiVersion})`
      : `Update required (API ${bridgeApiVersion || "legacy"})`,
    bridgeApiVersion >= REQUIRED_BRIDGE_API_VERSION ? "good" : "bad"
  ));
  grid.appendChild(diagnosticRow("Python", payload.python_version, "good"));
  grid.appendChild(diagnosticRow("Ollama", payload.ollama_on_path ? "Available on PATH" : "Not found on PATH", payload.ollama_on_path ? "good" : "warn"));
  const modules = payload.optional_modules || {};
  for (const [name, available] of Object.entries(modules)) {
    grid.appendChild(diagnosticRow(name, available ? "Ready" : "Optional dependency missing", available ? "good" : "warn"));
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

function diagnosticRow(label, state, kind = "info") {
  const row = document.createElement("div");
  row.className = "diagnostic-row";
  const name = document.createElement("strong");
  name.textContent = label;
  const value = document.createElement("span");
  value.className = `diagnostic-state ${kind}`;
  value.textContent = state;
  row.append(name, value);
  return row;
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

const wait = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function renderWorkflowProgress(job) {
  const panel = $("run-progress-panel");
  const rawPercent = Number(job?.progress_percent ?? 0);
  const percent = Number.isFinite(rawPercent) ? Math.min(100, Math.max(0, rawPercent)) : 0;
  panel.hidden = false;
  $("run-progress").value = percent;
  $("run-progress").textContent = `${Math.round(percent)}%`;
  setText("run-progress-value", `${Math.round(percent)}%`);
  setText("run-progress-label", job?.message || job?.phase || "Workflow running");
  $("run-cancel-btn").disabled = !["queued", "running"].includes(job?.status);
}

async function waitForWorkflowJob(jobId) {
  while (activeWorkflowJob === jobId) {
    const job = await call("workflow_job_status", { jobId });
    renderWorkflowProgress(job);
    if (job.status === "completed") return job.result;
    if (job.status === "cancelled") {
      throw new Error(job.message || "Workflow cancelled by the user.");
    }
    if (job.status === "failed") {
      throw new Error(job.error || job.message || "Workflow failed.");
    }
    await wait(350);
  }
  throw new Error("Workflow monitoring stopped before completion.");
}

async function cancelSelectedWorkflow() {
  if (!activeWorkflowJob) return;
  $("run-cancel-btn").disabled = true;
  setText("run-progress-label", "Requesting cancellation at a safe simulation boundary...");
  try {
    const job = await call("cancel_workflow_job", { jobId: activeWorkflowJob });
    renderWorkflowProgress(job);
  } catch (error) {
    setText("run-status", `Cancellation request failed.\n${errorMessage(error)}`);
  }
}

async function runSelectedWorkflow() {
  if (!engineCompatible) {
    setText(
      "run-status",
      "Update the Python SDK in Settings before running a protocol. This prevents the app and scientific engine from using incompatible workflow parameters."
    );
    return;
  }
  if (!selectedWorkflow) {
    setText("run-status", "Select a workflow first.");
    return;
  }
  const outputDir = $("output-dir").value.trim();
  const seed = Number.parseInt($("seed").value || "42", 10);
  if (!outputDir) {
    $("output-dir").setAttribute("aria-invalid", "true");
    setText("run-status", "Choose an output folder before starting the protocol.");
    return;
  }
  if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) {
    $("seed").setAttribute("aria-invalid", "true");
    setText("run-status", "Seed must be an integer between 0 and 2147483647.");
    return;
  }
  setBusy(true);
  $("run-progress-panel").hidden = false;
  $("run-complete-actions").hidden = true;
  $("run-progress").value = 0;
  setText("run-progress-value", "0%");
  setText("run-progress-label", "Queueing deterministic workflow...");
  setText("run-status", `Running ${selectedWorkflow} through the Python SDK...\nThis may take a minute.`);
  try {
    const started = await call("start_workflow_job", {
      workflowKey: selectedWorkflow,
      outputDir,
      seed
    });
    activeWorkflowJob = started.job_id;
    const result = await waitForWorkflowJob(activeWorkflowJob);
    lastRun = result;
    lastAcademicBundle = null;
    if (result.output_dir) {
      $("academic-run-dir").value = result.output_dir;
    }
    setText("run-status", result.summary || pretty(result));
    if (result.results_csv) {
      $("csv-path").value = result.results_csv;
      await previewCsv();
      $("run-complete-actions").hidden = false;
    }
    await loadHistory();
  } catch (error) {
    const message = errorMessage(error);
    setText(
      "run-status",
      message.includes("unexpected keyword argument")
        ? `The installed Python engine is incompatible with this protocol. Update it in Settings, refresh versions, and retry.\n\nTechnical summary: ${message.split("\n", 1)[0]}`
        : message
    );
  } finally {
    activeWorkflowJob = null;
    $("run-cancel-btn").disabled = true;
    setBusy(false);
  }
}

async function previewCsv() {
  const csv = $("csv-path").value.trim();
  if (!csv) {
    setText("results-status", "Choose a results CSV before loading a preview.");
    return;
  }
  setText("results-status", `Loading preview:\n${csv}`);
  try {
    const preview = await call("preview_results", { csv, maxRows: 80 });
    lastPreview = preview;
    lastMdmp = null;
    lastAcademicBundle = null;
    $("academic-run-dir").value = parentPath(preview.csv_path || csv);
    renderMetrics(preview.metrics || {});
    renderTable(preview.columns || [], preview.rows || []);
    drawGlucoseChart(preview);
    await loadCompartmentTimeline(preview.csv_path || csv);
    setText("results-status", `Loaded ${preview.row_count} rows.\n${csv}`);
    setText("run-status", `Preview loaded: ${preview.row_count} rows\n${csv}`);
    setText("ai-context", `Attached result CSV: ${csv}`);
    setActiveView("results", false);
    refreshActionAvailability();
  } catch (error) {
    lastPreview = null;
    setText("results-status", errorMessage(error));
    refreshActionAvailability();
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
        lastAcademicBundle = null;
        $("academic-run-dir").value = entry.output_dir || parentPath(entry.results_csv);
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
    refreshActionAvailability();
  } catch (error) {
    setText("mdmp-status", errorMessage(error));
  }
}

async function exportAcademicBundle() {
  const runDir = $("academic-run-dir").value.trim()
    || lastRun?.output_dir
    || parentPath(lastPreview?.csv_path || $("csv-path").value.trim());
  if (!runDir) {
    setText("academic-status", "Run a workflow or choose a completed run folder first.");
    return;
  }
  $("academic-run-dir").value = runDir;
  const sourceIds = $("academic-source-ids").value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  setText("academic-status", "Hashing run artifacts and building the reproducibility package...");
  $("academic-export-btn").disabled = true;
  try {
    const payload = await call("export_academic_bundle", {
      runDir,
      title: $("academic-title").value.trim() || null,
      description: null,
      creator: $("academic-creator").value.trim() || null,
      orcid: $("academic-orcid").value.trim() || null,
      licenseId: $("academic-license").value.trim() || "NOASSERTION",
      sourceIds
    });
    lastAcademicBundle = payload;
    setText(
      "academic-status",
      [
        `Readiness: ${payload.readiness_status}`,
        `Audit score: ${payload.readiness_score_pct}%`,
        `Artifacts inventoried: ${payload.artifact_count}`,
        `Evidence sources associated: ${payload.source_count}`,
        `RO-Crate: ${payload.ro_crate_metadata}`,
        `Audit: ${payload.audit_json}`,
        `Review guide: ${payload.readme_md}`,
        "",
        "This package supports review and reuse; it is not peer review, privacy approval, or clinical validation."
      ].join("\n")
    );
    refreshActionAvailability();
  } catch (error) {
    setText("academic-status", errorMessage(error));
  } finally {
    refreshActionAvailability();
  }
}

async function openAcademicMetadata() {
  await openPath(lastAcademicBundle?.ro_crate_metadata, "academic-status");
}

async function openAcademicAudit() {
  await openPath(lastAcademicBundle?.audit_json, "academic-status");
}

async function openAcademicGuide() {
  await openPath(lastAcademicBundle?.readme_md, "academic-status");
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
    setText("table-preview-summary", "No tabular data loaded");
    return;
  }
  const showAll = $("show-all-columns").checked;
  let columnIndices = columns
    .map((column, index) => ({ column: String(column), index }))
    .filter(({ column }) => showAll || ESSENTIAL_RESULT_COLUMNS.has(column.toLowerCase()));
  if (!columnIndices.length) {
    columnIndices = columns.slice(0, 12).map((column, index) => ({ column: String(column), index }));
  }
  setText(
    "table-preview-summary",
    `Showing ${columnIndices.length} of ${columns.length} columns and ${rows.length} bounded rows`
  );
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const { column } of columnIndices) {
    const th = document.createElement("th");
    th.textContent = column;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const { index } of columnIndices) {
      const cell = row[index] ?? "";
      const td = document.createElement("td");
      const fullText = String(cell);
      if (fullText.length > 180) {
        td.textContent = `${fullText.slice(0, 176)}...`;
        td.title = fullText.length <= 2000 ? fullText : "Long structured value. Open the CSV for the complete content.";
        td.classList.add("truncated-cell");
      } else {
        td.textContent = fullText;
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
}

function drawGlucoseChart(preview) {
  const svg = $("glucose-chart");
  svg.replaceChildren();
  svg.dataset.renderMode = "immediate";
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
    setText(
      "ai-status",
      [
        `Connection: ${payload.available ? "ready" : "not ready"}`,
        `Model: ${payload.resolved_model || $("ai-model").value.trim() || "not resolved"}`,
        payload.message || "No additional status returned."
      ].join("\n")
    );
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
    setText(
      "ai-status",
      [
        `Connection: ${payload.available ? "ready" : "not ready"}`,
        `Model: ${payload.resolved_model || $("ai-model").value.trim() || "not resolved"}`,
        `Ollama started by app: ${payload.started_process ? "yes" : "no"}`,
        `Model downloaded: ${payload.pulled_model ? "yes" : "no"}`,
        payload.message || ""
      ].filter(Boolean).join("\n")
    );
  } catch (error) {
    setText("ai-status", errorMessage(error));
  }
}

async function listAiModels() {
  setText("ai-status", "Listing local Ollama models...");
  try {
    const payload = await call("list_local_ai_models", { host: $("ai-host").value.trim() });
    const models = Array.isArray(payload.models) ? payload.models.filter(Boolean) : [];
    const options = $("ai-model-options");
    options.replaceChildren();
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model;
      options.appendChild(option);
    }
    setText(
      "ai-status",
      models.length
        ? `Selectable models (${models.length}). Installed models appear first; a suggested model may need to be downloaded:\n${models.map((model) => `- ${model}`).join("\n")}`
        : "No local or recommended models were returned."
    );
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
  aiBusy = true;
  refreshActionAvailability();
  setText("ai-answer", "Running local AI analysis. The model is local; generation can take several minutes on small machines.");
  try {
    const payload = await call("ask_local_ai", {
      question,
      model: $("ai-model").value.trim(),
      host: $("ai-host").value.trim(),
      csv: csv || null
    });
    renderAiAnswer(payload);
  } catch (error) {
    setText("ai-answer", errorMessage(error));
  } finally {
    aiBusy = false;
    refreshActionAvailability();
  }
}

function renderAiAnswer(payload) {
  const container = $("ai-answer");
  container.replaceChildren();

  const metadata = document.createElement("dl");
  metadata.className = "ai-metadata";
  const guard = payload.policy_action || ((payload.policy_violations || []).length ? "blocked" : "clear");
  const suppressedLineCount = Number(payload.suppressed_line_count || 0);
  for (const [label, value] of [
    ["Model", payload.model || "unknown"],
    ["CSV context", payload.context_used ? "used" : "not used"],
    ["Policy guard", guard],
    [
      "Numeric claim audit",
      suppressedLineCount
        ? `${suppressedLineCount} unsupported line${suppressedLineCount === 1 ? "" : "s"} hidden`
        : "no unsupported values detected"
    ],
    ["AI scope", payload.interpretation_restricted ? "limitations + next checks" : "general research question"]
  ]) {
    const item = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = value;
    item.append(term, description);
    metadata.appendChild(item);
  }
  container.appendChild(metadata);

  const deterministicMetrics = payload.deterministic_metrics || {};
  if (Object.keys(deterministicMetrics).length) {
    const factsHeading = document.createElement("h3");
    factsHeading.textContent = "Deterministic SDK facts";
    container.appendChild(factsHeading);
    const facts = document.createElement("dl");
    facts.className = "ai-facts";
    for (const [label, value] of Object.entries(deterministicMetrics)) {
      const item = document.createElement("div");
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = String(value);
      item.append(term, description);
      facts.appendChild(item);
    }
    container.appendChild(facts);
  }

  const alerts = [
    ...(payload.policy_violations || []).map((text) => `Policy violation: ${text}`),
    ...(payload.policy_warnings || []).map((text) => `Policy warning: ${text}`),
    ...(payload.numeric_claim_warnings || []).map((text) => `Factuality warning: ${text}`)
  ];
  if (alerts.length) {
    const list = document.createElement("ul");
    list.className = "ai-alert-list";
    for (const alert of alerts) {
      const item = document.createElement("li");
      item.textContent = alert;
      list.appendChild(item);
    }
    container.appendChild(list);
  }

  appendReadableText(container, payload.answer || "No answer returned.");
}

function appendReadableText(container, text) {
  let activeList = null;
  for (const rawLine of String(text).split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      activeList = null;
      continue;
    }
    const normalized = line.replace(/^#{1,6}\s*/, "").replaceAll("**", "").trim();
    if ([
      "Deterministic Facts",
      "Interpretation",
      "Limitations",
      "Next Checks",
      "Clinical Overview",
      "Biomathematical Observations",
      "Algorithmic Behavior",
      "Conclusions"
    ].includes(normalized)) {
      activeList = null;
      const heading = document.createElement("h3");
      heading.textContent = normalized;
      container.appendChild(heading);
      continue;
    }
    if (/^(?:[-*]|•)\s+/.test(line)) {
      if (!activeList) {
        activeList = document.createElement("ul");
        container.appendChild(activeList);
      }
      const item = document.createElement("li");
      item.textContent = normalized.replace(/^(?:[-*]|•)\s+/, "");
      activeList.appendChild(item);
      continue;
    }
    activeList = null;
    const paragraph = document.createElement("p");
    paragraph.textContent = normalized;
    container.appendChild(paragraph);
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

async function revealPath(path, statusId = "run-status") {
  if (!path) {
    setText(statusId, "Nothing to reveal yet.");
    return;
  }
  try {
    await call("reveal_path", { path });
    setText(statusId, `Revealed local artifact:\n${path}`);
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
  await openPath($("csv-path").value.trim(), "results-status");
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
        `Scenario functional scalar: ${payload.metadata?.scalar ?? "n/a"}`,
        "The scalar is an explicit research assumption; AlphaFold pLDDT is not pathogenicity or metabolic severity.",
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

async function loadMechanisticStatus() {
  try {
    const payload = await call("mechanistic_engine_status");
    const lines = [
      `SBML inspection: ${payload.inspection_available ? "available" : "unavailable"}`,
      `Execution engine: ${payload.engine} ${payload.version || "not installed"}`,
      payload.message,
      "Execution is independent reference-model evidence, not biological validation."
    ];
    setText("mechanistic-status", lines.join("\n"));
  } catch (error) {
    setText("mechanistic-status", errorMessage(error));
  }
}

async function inspectMechanisticModel() {
  const model = $("mechanistic-model").value.trim();
  if (!model) {
    setText("mechanistic-status", "Choose a local .xml or .sbml model first.");
    return;
  }
  setResearchBusy(true);
  setText("mechanistic-status", "Inspecting SBML structure without executing equations...");
  try {
    const payload = await call("inspect_mechanistic_model", { model });
    const summary = payload.summary || {};
    const counts = summary.counts || {};
    const warnings = Array.isArray(summary.warnings) && summary.warnings.length
      ? summary.warnings.map((warning) => `- ${warning}`).join("\n")
      : "- No structural inspection warnings.";
    setText(
      "mechanistic-status",
      [
        `Model: ${summary.model_name || summary.model_id || model}`,
        `SBML: Level ${summary.sbml_level ?? "?"}, version ${summary.sbml_version ?? "?"}`,
        `Readiness: ${summary.readiness_status}`,
        `Species: ${counts.species ?? 0}; reactions: ${counts.reactions ?? 0}; parameters: ${counts.parameters ?? 0}`,
        `Model units: ${pretty(summary.model_units || {})}`,
        `SHA-256: ${summary.sha256 || "n/a"}`,
        `Warnings:\n${warnings}`,
        "This is safe structural inspection, not full SBML schema or biological validation."
      ].join("\n")
    );
  } catch (error) {
    setText("mechanistic-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
  }
}

async function runMechanisticModel() {
  const model = $("mechanistic-model").value.trim();
  if (!model) {
    setText("mechanistic-status", "Choose a local .xml or .sbml model first.");
    return;
  }
  const outputDir = joinPath($("output-dir").value.trim(), "mechanistic_reference");
  const variables = $("mechanistic-variables").value
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  const sourceUrl = $("mechanistic-source-url").value.trim();
  const modelLicense = $("mechanistic-license").value.trim() || "NOASSERTION";
  const start = Number.parseFloat($("mechanistic-start").value || "0");
  const end = Number.parseFloat($("mechanistic-end").value || "1440");
  const points = Number.parseInt($("mechanistic-points").value || "289", 10);
  setResearchBusy(true);
  setText("mechanistic-status", "Executing independent SBML reference model through libRoadRunner...");
  try {
    const payload = await call("run_mechanistic_model", {
      request: {
        model,
        outputDir,
        start,
        end,
        points,
        variables,
        sourceUrl: sourceUrl || null,
        modelLicense
      }
    });
    lastMechanistic = payload;
    setText(
      "mechanistic-status",
      [
        `Reference run completed with ${payload.engine} ${payload.engine_version}.`,
        `Rows: ${payload.row_count}`,
        `Selections: ${(payload.selections || []).join(", ")}`,
        `Run folder: ${payload.run_dir}`,
        `Results: ${payload.results_csv}`,
        `Manifest: ${payload.manifest_json}`,
        "No unit conversion or automatic IINTS calibration was performed."
      ].join("\n")
    );
  } catch (error) {
    setText("mechanistic-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
  }
}

async function openMechanisticFolder() {
  await openPath(lastMechanistic?.run_dir, "mechanistic-status");
}

async function openMechanisticReport() {
  await openPath(lastMechanistic?.report_md, "mechanistic-status");
}

async function openMechanisticResults() {
  await openPath(lastMechanistic?.results_csv, "mechanistic-status");
}

function warningLines(warnings) {
  return Array.isArray(warnings) && warnings.length
    ? warnings.map((warning) => `- ${warning}`).join("\n")
    : "- None reported by static inspection.";
}

async function loadCrossScaleStatus() {
  setText("cross-scale-status", "Checking optional academic engines...");
  try {
    const payload = await call("cross_scale_engine_status");
    const copasi = payload.copasi || {};
    const opencor = payload.opencor || {};
    const fmpy = payload.fmpy || {};
    setText(
      "cross-scale-status",
      [
        "Static inspection: COPASI, CellML, and FMU available without optional engines.",
        `CopasiSE: ${copasi.available ? "available" : "not found"}${copasi.path ? ` · ${copasi.path}` : ""}`,
        `OpenCOR: ${opencor.available ? "available" : "not found"}${opencor.version ? ` · ${opencor.version}` : ""}`,
        `FMPy: ${fmpy.available ? `available · ${fmpy.version || "unknown version"}` : "not installed"}`,
        "BindingDB: read-only HTTPS evidence connector; network required.",
        "No external result is coupled to patient parameters automatically."
      ].join("\n")
    );
  } catch (error) {
    setText("cross-scale-status", errorMessage(error));
  }
}

async function inspectCopasiModel() {
  const model = $("copasi-model").value.trim();
  if (!model) {
    setText("copasi-status", "Choose a local .cps model first.");
    return;
  }
  setResearchBusy(true);
  setText("copasi-status", "Inspecting COPASI tasks without executing them...");
  try {
    const payload = await call("inspect_copasi_model", { model });
    const summary = payload.summary || {};
    const tasks = Array.isArray(summary.tasks) ? summary.tasks : [];
    const taskLines = tasks.length
      ? tasks.map((task) => `- ${task.name || task.raw_type}: ${task.kind}; scheduled=${task.scheduled}; method=${task.method_name || "n/a"}`).join("\n")
      : "- No tasks found.";
    setText(
      "copasi-status",
      [
        `Model: ${summary.model_name || model}`,
        `Readiness: ${summary.readiness_status}`,
        `Sensitivity tasks: ${summary.sensitivity_task_count ?? 0}`,
        `Parameter-estimation tasks: ${summary.parameter_estimation_task_count ?? 0}`,
        `Scheduled tasks: ${summary.scheduled_task_count ?? 0}`,
        `SHA-256: ${summary.sha256 || "n/a"}`,
        `Tasks:\n${taskLines}`,
        `Warnings:\n${warningLines(summary.warnings)}`,
        "Task presence does not prove identifiability or convergence."
      ].join("\n")
    );
  } catch (error) {
    setText("copasi-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
  }
}

async function runCopasiAnalysis() {
  const model = $("copasi-model").value.trim();
  if (!model) {
    setText("copasi-status", "Choose a local .cps model first.");
    return;
  }
  if (!$("copasi-consent").checked) {
    setText("copasi-status", "Review the configured tasks/data paths and tick the execution confirmation first.");
    return;
  }
  const outputDir = joinPath($("output-dir").value.trim(), "copasi");
  const task = $("copasi-task").value.trim();
  const timeoutSeconds = Number.parseInt($("copasi-timeout").value || "900", 10);
  setResearchBusy(true);
  setText("copasi-status", "Running the reviewed COPASI task in an evidence directory...");
  try {
    const payload = await call("run_copasi_analysis", {
      request: {
        model,
        outputDir,
        task: task || null,
        timeoutSeconds,
        allowExternalExecution: true
      }
    });
    lastCopasi = payload;
    setText(
      "copasi-status",
      [
        "COPASI analysis completed.",
        `Task: ${payload.selected_task || "task scheduled in model"}`,
        `Run folder: ${payload.run_dir}`,
        `Report: ${payload.report_txt}`,
        `Manifest: ${payload.manifest_json}`,
        "Review residuals, units, bounds, convergence, and profile-likelihood evidence before interpretation."
      ].join("\n")
    );
  } catch (error) {
    setText("copasi-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
  }
}

async function inspectCellmlModel() {
  const model = $("cellml-model").value.trim();
  if (!model) {
    setText("cellml-status", "Choose a local .cellml or .xml model first.");
    return;
  }
  setResearchBusy(true);
  setText("cellml-status", "Inspecting CellML metadata without resolving imports...");
  try {
    const payload = await call("inspect_cellml_reference", { model });
    const summary = payload.summary || {};
    setText(
      "cellml-status",
      [
        `Model: ${summary.model_name || model}`,
        `CellML: ${summary.cellml_version || "unknown"}`,
        `Readiness: ${summary.readiness_status}`,
        `Components: ${summary.component_count ?? 0}; variables: ${summary.variable_count ?? 0}; MathML blocks: ${summary.math_block_count ?? 0}`,
        `Imports: ${(summary.imports || []).join(", ") || "none"}`,
        `SHA-256: ${summary.sha256 || "n/a"}`,
        `Warnings:\n${warningLines(summary.warnings)}`,
        "Static inspection does not execute equations or trust imported files."
      ].join("\n")
    );
  } catch (error) {
    setText("cellml-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
  }
}

async function validateCellmlModel() {
  const model = $("cellml-model").value.trim();
  if (!model) {
    setText("cellml-status", "Choose a local .cellml or .xml model first.");
    return;
  }
  const outputDir = joinPath($("output-dir").value.trim(), "cellml");
  const timeoutSeconds = Number.parseInt($("cellml-timeout").value || "120", 10);
  setResearchBusy(true);
  setText("cellml-status", "Validating CellML through OpenCOR CellMLTools...");
  try {
    const payload = await call("validate_cellml_reference", {
      request: { model, outputDir, timeoutSeconds }
    });
    lastCellml = payload;
    setText(
      "cellml-status",
      [
        `OpenCOR validation result: ${payload.valid ? "valid" : "invalid or errors reported"}`,
        `Return code: ${payload.return_code}`,
        `Run folder: ${payload.run_dir}`,
        `Validation log: ${payload.validation_log}`,
        `Manifest: ${payload.manifest_json}`,
        "CellML validation does not establish biological or clinical validity."
      ].join("\n")
    );
  } catch (error) {
    setText("cellml-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
  }
}

async function inspectFmiModel() {
  const model = $("fmi-model").value.trim();
  if (!model) {
    setText("fmi-status", "Choose a local .fmu first.");
    return;
  }
  setResearchBusy(true);
  setText("fmi-status", "Reading FMU archive metadata only; native code is not being loaded...");
  try {
    const payload = await call("inspect_fmu_model", { model });
    const summary = payload.summary || {};
    setText(
      "fmi-status",
      [
        `Model: ${summary.model_name || model}`,
        `FMI: ${summary.fmi_version || "unknown"}`,
        `Interfaces: ${(summary.interfaces || []).map((item) => item.type).join(", ") || "none"}`,
        `Variables: ${summary.variable_count ?? 0}`,
        `Platforms: ${(summary.platforms || []).join(", ") || "none"}`,
        `Native binaries: ${Boolean(summary.has_native_binaries)}`,
        `SHA-256: ${summary.sha256 || "n/a"}`,
        `Warnings:\n${warningLines(summary.warnings)}`,
        "Static inspection completed without loading FMU binaries."
      ].join("\n")
    );
  } catch (error) {
    setText("fmi-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
  }
}

async function runFmiModel() {
  const model = $("fmi-model").value.trim();
  if (!model) {
    setText("fmi-status", "Choose a local .fmu first.");
    return;
  }
  if (!$("fmi-consent").checked) {
    setText("fmi-status", "Inspect the FMU and explicitly accept the native-code boundary before execution.");
    return;
  }
  const outputDir = joinPath($("output-dir").value.trim(), "fmi");
  const start = Number.parseFloat($("fmi-start").value || "0");
  const end = Number.parseFloat($("fmi-end").value || "60");
  const outputInterval = Number.parseFloat($("fmi-interval").value || "0.1");
  const timeoutSeconds = Number.parseInt($("fmi-timeout").value || "300", 10);
  const variables = $("fmi-variables").value.split(",").map((value) => value.trim()).filter(Boolean);
  setResearchBusy(true);
  setText("fmi-status", "Executing the explicitly trusted FMU through FMPy...");
  try {
    const payload = await call("run_fmi_model", {
      request: {
        model,
        outputDir,
        start,
        end,
        outputInterval,
        variables,
        timeoutSeconds,
        trustNativeCode: true
      }
    });
    lastFmi = payload;
    setText(
      "fmi-status",
      [
        `Trusted FMU run completed with ${payload.engine} ${payload.engine_version}.`,
        `Rows: ${payload.row_count}; columns: ${(payload.columns || []).join(", ")}`,
        `Run folder: ${payload.run_dir}`,
        `Results: ${payload.results_csv}`,
        `Manifest: ${payload.manifest_json}`,
        "Execution success is not bench validation and no result controls a real device."
      ].join("\n")
    );
  } catch (error) {
    setText("fmi-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
  }
}

async function queryBindingEvidence() {
  const uniprot = $("binding-uniprot").value.trim().toUpperCase();
  const outputDir = joinPath($("output-dir").value.trim(), "bindingdb");
  const cutoffNm = Number.parseInt($("binding-cutoff").value || "10000", 10);
  const maxRecords = Number.parseInt($("binding-max-records").value || "5000", 10);
  if (!uniprot) {
    setText("binding-status", "Enter one UniProt accession first.");
    return;
  }
  setResearchBusy(true);
  setText("binding-status", `Fetching measured BindingDB records for ${uniprot} over verified HTTPS...`);
  try {
    const payload = await call("query_bindingdb_evidence", {
      request: { uniprot, outputDir, cutoffNm, maxRecords, timeoutSeconds: 30 }
    });
    lastBinding = payload;
    setText(
      "binding-status",
      [
        `BindingDB query completed for ${payload.uniprot_accession}.`,
        `Cutoff: ${payload.cutoff_nm} nM; exported records: ${payload.record_count}`,
        `Truncated by local limit: ${payload.truncated}`,
        `CSV: ${payload.records_csv}`,
        `Evidence JSON: ${payload.evidence_json}`,
        "Ki, Kd, IC50, AlphaFold confidence, and in-vivo effects remain separate evidence types."
      ].join("\n")
    );
  } catch (error) {
    setText("binding-status", errorMessage(error));
  } finally {
    setResearchBusy(false);
  }
}

async function openCopasiBundle() {
  await openPath(lastCopasi?.run_dir, "copasi-status");
}

async function openCellmlBundle() {
  await openPath(lastCellml?.run_dir, "cellml-status");
}

async function openFmiBundle() {
  await openPath(lastFmi?.run_dir, "fmi-status");
}

async function openFmiResults() {
  await openPath(lastFmi?.results_csv, "fmi-status");
}

async function openBindingBundle() {
  await openPath(lastBinding?.output_dir, "binding-status");
}

async function openBindingCsv() {
  await openPath(lastBinding?.records_csv, "binding-status");
}

function setDisabled(id, disabled) {
  const element = $(id);
  if (element) element.disabled = Boolean(disabled);
}

function pathHasExtension(path, extensions) {
  const value = String(path || "").trim().toLowerCase();
  return extensions.some((extension) => value.endsWith(`.${extension.toLowerCase()}`));
}

function refreshActionAvailability() {
  const outputDir = $("output-dir").value.trim();
  const seed = Number.parseInt($("seed").value, 10);
  const seedValid = Number.isInteger(seed) && seed >= 0 && seed <= 2147483647;
  const csv = $("csv-path").value.trim();
  const csvValid = pathHasExtension(csv, ["csv"]);
  const aiQuestionValid = Boolean($("ai-question").value.trim());
  const academicRun = $("academic-run-dir").value.trim()
    || lastRun?.output_dir
    || parentPath(lastPreview?.csv_path || csv);
  const mechanisticModel = $("mechanistic-model").value.trim();
  const copasiModel = $("copasi-model").value.trim();
  const cellmlModel = $("cellml-model").value.trim();
  const fmiModel = $("fmi-model").value.trim();
  const mechanisticValid = pathHasExtension(mechanisticModel, ["xml", "sbml"]);
  const copasiValid = pathHasExtension(copasiModel, ["cps"]);
  const cellmlValid = pathHasExtension(cellmlModel, ["cellml", "xml"]);
  const fmiValid = pathHasExtension(fmiModel, ["fmu"]);

  setDisabled("run-btn", runBusy || !engineCompatible || !selectedWorkflow || !outputDir || !seedValid);
  setDisabled("run-cancel-btn", !runBusy || !activeWorkflowJob);
  setDisabled("refresh-btn", runBusy);
  setDisabled("history-btn", runBusy || !outputDir);
  setDisabled("output-browse-btn", runBusy);
  setDisabled("preview-btn", runBusy || !csvValid);
  setDisabled("csv-browse-btn", runBusy);
  setDisabled("open-csv-btn", runBusy || !csvValid);
  setDisabled("mdmp-btn", runBusy || !csvValid);
  setDisabled("open-run-folder-btn", runBusy || !lastRun?.output_dir);
  setDisabled("open-report-btn", runBusy || !lastRun?.report_pdf);
  setDisabled("open-certificate-btn", runBusy || !lastMdmp?.certificate_path);
  setDisabled("academic-export-btn", runBusy || !academicRun);
  setDisabled("academic-run-browse-btn", runBusy);
  setDisabled("academic-open-metadata-btn", !lastAcademicBundle?.ro_crate_metadata);
  setDisabled("academic-open-audit-btn", !lastAcademicBundle?.audit_json);
  setDisabled("academic-open-guide-btn", !lastAcademicBundle?.readme_md);
  setDisabled("ai-ask-btn", aiBusy || !aiQuestionValid);
  $("ai-ask-btn").textContent = aiBusy
    ? "Analyzing..."
    : csvValid
      ? "Analyze loaded result"
      : "Ask local AI without result";

  for (const id of [
    "mechanistic-model-browse-btn",
    "copasi-model-browse-btn",
    "cellml-model-browse-btn",
    "fmi-model-browse-btn"
  ]) {
    setDisabled(id, researchBusy);
  }
  setDisabled("molecule-refresh-btn", researchBusy);
  setDisabled("open-structural-folder-btn", researchBusy || !outputDir);
  setDisabled("molecule-viewer-reset-btn", !moleculeViewer.molecule);
  setDisabled("molecule-viewer-rotate-btn", !moleculeViewer.molecule);
  document.querySelectorAll("#molecule-list button[data-unavailable]").forEach((button) => {
    button.disabled = researchBusy || button.dataset.unavailable === "true";
  });
  setDisabled("genomics-run-btn", researchBusy);
  setDisabled("genomics-open-btn", researchBusy || !lastGenomics?.html_path);
  setDisabled("tissue-run-btn", researchBusy);
  setDisabled("tissue-open-btn", researchBusy || !lastTissue?.html_path);
  setDisabled("mechanistic-status-btn", researchBusy);
  setDisabled("mechanistic-inspect-btn", researchBusy || !mechanisticValid);
  setDisabled("mechanistic-run-btn", researchBusy || !mechanisticValid || !outputDir);
  setDisabled("mechanistic-open-folder-btn", researchBusy || !lastMechanistic?.run_dir);
  setDisabled("mechanistic-open-report-btn", researchBusy || !lastMechanistic?.report_md);
  setDisabled("mechanistic-open-results-btn", researchBusy || !lastMechanistic?.results_csv);
  setDisabled("cross-scale-status-btn", researchBusy);
  setDisabled("copasi-inspect-btn", researchBusy || !copasiValid);
  setDisabled("copasi-run-btn", researchBusy || !copasiValid || !outputDir || !$("copasi-consent").checked);
  setDisabled("copasi-open-btn", researchBusy || !lastCopasi?.run_dir);
  setDisabled("cellml-inspect-btn", researchBusy || !cellmlValid);
  setDisabled("cellml-validate-btn", researchBusy || !cellmlValid || !outputDir);
  setDisabled("cellml-open-btn", researchBusy || !lastCellml?.run_dir);
  setDisabled("fmi-inspect-btn", researchBusy || !fmiValid);
  setDisabled("fmi-run-btn", researchBusy || !fmiValid || !outputDir || !$("fmi-consent").checked);
  setDisabled("fmi-open-btn", researchBusy || !lastFmi?.run_dir);
  setDisabled("fmi-results-btn", researchBusy || !lastFmi?.results_csv);
  setDisabled("binding-query-btn", researchBusy || !$("binding-uniprot").value.trim() || !outputDir);
  setDisabled("binding-open-btn", researchBusy || !lastBinding?.output_dir);
  setDisabled("binding-csv-btn", researchBusy || !lastBinding?.records_csv);
}

function setBusy(isBusy) {
  runBusy = Boolean(isBusy);
  document.documentElement.classList.toggle("run-busy", runBusy);
  $("run-btn").textContent = runBusy ? "Running protocol..." : "Run selected protocol";
  $("workspace-content").setAttribute("aria-busy", String(runBusy || researchBusy));
  refreshActionAvailability();
}

function setResearchBusy(isBusy) {
  researchBusy = Boolean(isBusy);
  document.documentElement.classList.toggle("research-busy", researchBusy);
  $("workspace-content").setAttribute("aria-busy", String(runBusy || researchBusy));
  refreshActionAvailability();
}

function initializeFormState() {
  $("output-dir").addEventListener("input", () => {
    $("output-dir").setAttribute("aria-invalid", String(!$("output-dir").value.trim()));
    lastRun = null;
    refreshActionAvailability();
  });
  $("seed").addEventListener("input", () => {
    const seed = Number.parseInt($("seed").value, 10);
    $("seed").setAttribute(
      "aria-invalid",
      String(!Number.isInteger(seed) || seed < 0 || seed > 2147483647)
    );
    refreshActionAvailability();
  });
  $("csv-path").addEventListener("input", () => {
    const value = $("csv-path").value.trim();
    $("csv-path").setAttribute("aria-invalid", String(Boolean(value) && !pathHasExtension(value, ["csv"])));
    lastPreview = null;
    lastMdmp = null;
    refreshActionAvailability();
  });
  $("academic-run-dir").addEventListener("input", () => {
    lastAcademicBundle = null;
    refreshActionAvailability();
  });
  $("mechanistic-model").addEventListener("input", () => {
    const value = $("mechanistic-model").value.trim();
    $("mechanistic-model").setAttribute(
      "aria-invalid",
      String(Boolean(value) && !pathHasExtension(value, ["xml", "sbml"]))
    );
    lastMechanistic = null;
    refreshActionAvailability();
  });
  $("copasi-model").addEventListener("input", () => {
    const value = $("copasi-model").value.trim();
    $("copasi-model").setAttribute(
      "aria-invalid",
      String(Boolean(value) && !pathHasExtension(value, ["cps"]))
    );
    lastCopasi = null;
    refreshActionAvailability();
  });
  $("cellml-model").addEventListener("input", () => {
    const value = $("cellml-model").value.trim();
    $("cellml-model").setAttribute(
      "aria-invalid",
      String(Boolean(value) && !pathHasExtension(value, ["cellml", "xml"]))
    );
    lastCellml = null;
    refreshActionAvailability();
  });
  $("fmi-model").addEventListener("input", () => {
    const value = $("fmi-model").value.trim();
    $("fmi-model").setAttribute(
      "aria-invalid",
      String(Boolean(value) && !pathHasExtension(value, ["fmu"]))
    );
    lastFmi = null;
    refreshActionAvailability();
  });
  for (const id of ["copasi-consent", "fmi-consent", "binding-uniprot"]) {
    $(id).addEventListener("input", refreshActionAvailability);
    $(id).addEventListener("change", refreshActionAvailability);
  }
}

function initializeMoleculeViewer() {
  const canvas = $("molecule-viewer-canvas");
  canvas.addEventListener("pointerdown", (event) => {
    if (!moleculeViewer.molecule) return;
    moleculeViewer.dragging = true;
    moleculeViewer.pointerX = event.clientX;
    moleculeViewer.pointerY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("is-dragging");
    setMoleculeAutoRotate(false);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!moleculeViewer.dragging) return;
    const deltaX = event.clientX - moleculeViewer.pointerX;
    const deltaY = event.clientY - moleculeViewer.pointerY;
    moleculeViewer.pointerX = event.clientX;
    moleculeViewer.pointerY = event.clientY;
    moleculeViewer.rotationY += deltaX * 0.01;
    moleculeViewer.rotationX = Math.max(
      -Math.PI / 2,
      Math.min(Math.PI / 2, moleculeViewer.rotationX + deltaY * 0.01)
    );
    drawMoleculeViewer();
  });
  const stopDragging = (event) => {
    moleculeViewer.dragging = false;
    canvas.classList.remove("is-dragging");
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  };
  canvas.addEventListener("pointerup", stopDragging);
  canvas.addEventListener("pointercancel", stopDragging);
  canvas.addEventListener("wheel", (event) => {
    if (!moleculeViewer.molecule) return;
    event.preventDefault();
    const factor = event.deltaY > 0 ? 0.9 : 1.1;
    moleculeViewer.zoom = Math.max(0.45, Math.min(3.5, moleculeViewer.zoom * factor));
    drawMoleculeViewer();
  }, { passive: false });
  canvas.addEventListener("keydown", (event) => {
    if (!moleculeViewer.molecule) return;
    const rotationStep = 0.08;
    if (event.key === "ArrowLeft") moleculeViewer.rotationY -= rotationStep;
    else if (event.key === "ArrowRight") moleculeViewer.rotationY += rotationStep;
    else if (event.key === "ArrowUp") moleculeViewer.rotationX -= rotationStep;
    else if (event.key === "ArrowDown") moleculeViewer.rotationX += rotationStep;
    else if (event.key === "+" || event.key === "=") {
      moleculeViewer.zoom = Math.min(3.5, moleculeViewer.zoom * 1.1);
    } else if (event.key === "-" || event.key === "_") {
      moleculeViewer.zoom = Math.max(0.45, moleculeViewer.zoom * 0.9);
    } else if (event.key === "0") {
      resetMoleculeViewer();
      event.preventDefault();
      return;
    } else if (event.key === "Escape") {
      closeMoleculeViewer();
      event.preventDefault();
      return;
    } else {
      return;
    }
    event.preventDefault();
    drawMoleculeViewer();
  });

  if (typeof ResizeObserver === "function") {
    const observer = new ResizeObserver(() => drawMoleculeViewer());
    observer.observe(canvas);
  } else {
    window.addEventListener("resize", drawMoleculeViewer);
  }
}

function initializeKeyboardShortcuts() {
  document.addEventListener("keydown", (event) => {
    if (!(event.metaKey || event.ctrlKey) || event.altKey) return;
    if (event.key === ",") {
      event.preventDefault();
      setActiveView("settings");
      $("settings-output-dir").focus();
    } else if (event.key.toLowerCase() === "o") {
      event.preventDefault();
      setActiveView("results");
      void chooseResultsCsv();
    }
  });
}

function firstIndex(columns, candidates) {
  const lower = columns.map((column) => String(column).toLowerCase());
  for (const candidate of candidates) {
    const index = lower.indexOf(candidate.toLowerCase());
    if (index >= 0) return index;
  }
  return -1;
}

function parentPath(path) {
  const value = String(path || "").trim().replace(/[\\/]+$/, "");
  if (!value) return "";
  const separatorIndex = Math.max(value.lastIndexOf("/"), value.lastIndexOf("\\"));
  if (separatorIndex === 0) return value[0];
  if (separatorIndex === 2 && value[1] === ":") return value.slice(0, 3);
  return separatorIndex > 0 ? value.slice(0, separatorIndex) : "";
}

function joinPath(base, child) {
  const root = String(base || "").trim().replace(/[\\/]+$/, "");
  if (!root) return child;
  const separator = root.includes("\\") && !root.includes("/") ? "\\" : "/";
  return `${root}${separator}${child}`;
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

// --- Compartment model viewer -------------------------------------------
// The diagram is built from the schema the run itself exported. Nothing about
// the physiology is re-implemented here: every number drawn is a value the
// simulator computed and wrote to results.csv.

const SVG_NS = "http://www.w3.org/2000/svg";

// Anatomical left-to-right ordering. Sites absent from a model's schema are
// simply skipped, so the same layout serves any backend that declares sites.
const COMPARTMENT_SITE_ORDER = ["gut", "subcutaneous", "plasma", "periphery", "signal"];
const COMPARTMENT_SITE_LABELS = {
  gut: "Gut",
  subcutaneous: "Subcutaneous",
  plasma: "Plasma",
  periphery: "Periphery",
  signal: "Signals",
};

let compartmentTimeline = null;

function compartmentRange(series) {
  let low = Infinity;
  let high = -Infinity;
  for (const value of series) {
    if (!Number.isFinite(value)) continue;
    if (value < low) low = value;
    if (value > high) high = value;
  }
  if (!Number.isFinite(low)) return null;
  return { low, high };
}

function formatCompartmentValue(value) {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 1000 || magnitude < 0.001) return value.toExponential(2);
  if (magnitude >= 10) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(4);
}

function svgNode(name, attributes) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

function compartmentIsVisible(compartment, showSignals) {
  if (showSignals) return true;
  // Effect and legacy states are not amounts of a substance; hiding them by
  // default keeps the initial view to actual contents.
  return compartment.kind === "pool" || compartment.kind === "concentration";
}

// Moves `centre` out to the border of its 104x40 node box, along the line
// towards `towards`, plus a small gap so the arrowhead is drawn in open space
// instead of underneath the box. Returns the centre unchanged for a degenerate
// direction, which keeps the caller's zero-length guard meaningful.
function trimToBox(centre, towards, halfWidth = 52, halfHeight = 20, gap = 7) {
  const dx = towards.x - centre.x;
  const dy = towards.y - centre.y;
  const length = Math.hypot(dx, dy);
  if (!Number.isFinite(length) || length < 1) return centre;
  const scale = Math.min(
    Math.abs(dx) > 0.001 ? (halfWidth + gap) / Math.abs(dx) : Infinity,
    Math.abs(dy) > 0.001 ? (halfHeight + gap) / Math.abs(dy) : Infinity
  );
  // Never overshoot the far endpoint: for boxes closer than their own border
  // the arrow would otherwise flip direction.
  const t = Math.min(scale, 0.45);
  return { x: centre.x + dx * t, y: centre.y + dy * t };
}

function layoutCompartments(compartments) {
  const bySite = new Map();
  for (const compartment of compartments) {
    const site = COMPARTMENT_SITE_ORDER.includes(compartment.site) ? compartment.site : "signal";
    if (!bySite.has(site)) bySite.set(site, []);
    bySite.get(site).push(compartment);
  }
  const sites = COMPARTMENT_SITE_ORDER.filter((site) => bySite.has(site));
  const positions = new Map();
  const columnWidth = 760 / Math.max(sites.length, 1);
  sites.forEach((site, columnIndex) => {
    const column = bySite.get(site);
    const centreX = columnWidth * (columnIndex + 0.5);
    const spacing = 320 / Math.max(column.length, 1);
    column.forEach((compartment, rowIndex) => {
      positions.set(compartment.key, {
        x: centreX,
        y: 70 + spacing * (rowIndex + 0.5),
        site,
        compartment,
      });
    });
  });
  return { positions, sites, columnWidth };
}

function renderCompartmentDiagram() {
  const diagram = $("compartment-diagram");
  const timeline = compartmentTimeline;
  if (!diagram || !timeline || !timeline.available) return;

  const showSignals = $("compartment-show-signals").checked;
  const index = Number($("compartment-time").value) || 0;
  const compartments = (timeline.schema.compartments || []).filter((compartment) =>
    compartmentIsVisible(compartment, showSignals)
  );
  const visibleKeys = new Set(compartments.map((compartment) => compartment.key));
  const { positions, sites, columnWidth } = layoutCompartments(compartments);

  diagram.replaceChildren();
  const defs = svgNode("defs", {});
  for (const [id, colour] of [
    ["compartment-arrow", "#3f6f9f"],
    ["compartment-arrow-weak", "#b8c4cf"],
  ]) {
    const marker = svgNode("marker", {
      id,
      viewBox: "0 0 10 10",
      refX: 9,
      refY: 5,
      // Fixed size in user units. The SVG default scales the head with stroke
      // width, which lets the thickest flux draw a head wide enough to cover
      // the box it points at; thickness alone carries the magnitude.
      markerUnits: "userSpaceOnUse",
      markerWidth: 9,
      markerHeight: 9,
      orient: "auto-start-reverse",
    });
    marker.appendChild(svgNode("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: colour }));
    defs.appendChild(marker);
  }
  diagram.appendChild(defs);

  sites.forEach((site, columnIndex) => {
    diagram.appendChild(
      svgText(columnWidth * columnIndex + 12, 28, COMPARTMENT_SITE_LABELS[site] || site)
    );
  });

  // Several boundary fluxes can share one compartment -- plasma glucose has
  // production, uptake, renal clearance, exercise and dawn terms. Give each a
  // slot along the box edge, otherwise they are drawn on top of one another
  // and read as a single flux.
  const boundarySlots = new Map();
  const boundaryCounts = new Map();
  for (const flux of timeline.schema.fluxes || []) {
    if (!timeline.fluxes[flux.key]) continue;
    if (flux.source && flux.target) continue;
    const key = flux.source || flux.target;
    if (!positions.has(key)) continue;
    boundarySlots.set(flux.key, boundaryCounts.get(key) || 0);
    boundaryCounts.set(key, (boundaryCounts.get(key) || 0) + 1);
  }

  // Fluxes first, so node boxes are drawn over the arrow ends.
  let hiddenEndpointFluxes = 0;
  for (const flux of timeline.schema.fluxes || []) {
    const series = timeline.fluxes[flux.key];
    if (!series) continue; // declared but not numerically recorded
    const rate = series[index];
    if (!Number.isFinite(rate)) continue;
    const extreme = timeline.flux_extremes[flux.key] || [0, 0];
    const scale = Math.max(Math.abs(extreme[0]), Math.abs(extreme[1]));
    const strength = scale > 0 ? Math.abs(rate) / scale : 0;
    const source = positions.get(flux.source);
    const target = positions.get(flux.target);
    // One or both endpoints belong to a state the user has hidden.
    if (flux.source && flux.target && (!source || !target)) {
      hiddenEndpointFluxes += 1;
      continue;
    }
    if (!source && !target) {
      hiddenEndpointFluxes += 1;
      continue;
    }

    // A flux with only one endpoint inside the patient crosses the boundary:
    // an infusion, an elimination, or a production term. It gets a stub that
    // starts or ends outside the box, never a line between the same point.
    const boundary = !flux.source || !flux.target;
    const anchor = source || target;
    const slotCount = boundary ? boundaryCounts.get(flux.source || flux.target) || 1 : 1;
    const slot = boundary ? boundarySlots.get(flux.key) || 0 : 0;
    // Slots spread across the box width, so each boundary flux leaves the box
    // at its own point on the edge.
    const slotX = anchor.x + (slotCount > 1 ? (slot - (slotCount - 1) / 2) * (88 / (slotCount - 1)) : 0);
    // Both stub endpoints are placed directly: just clear of the box edge on
    // the inside, and 33px further out. The 27 keeps the arrowhead of an
    // inbound flux clear of the box that is drawn over it.
    const direction = flux.source ? 1 : -1;
    const inside = { x: slotX, y: anchor.y + direction * 27 };
    const outside = { x: slotX, y: anchor.y + direction * 60 };
    let start = boundary ? (flux.source ? inside : outside) : { x: source.x, y: source.y };
    let end = boundary ? (flux.source ? outside : inside) : { x: target.x, y: target.y };
    // Only a box centre needs trimming; stub endpoints are already positioned.
    let startOnBox = !boundary;
    let endOnBox = !boundary;
    // A negative rate means the term runs the other way; the arrow is
    // reversed rather than clamped, so the direction stays truthful.
    if (rate < 0) {
      [start, end] = [end, start];
      [startOnBox, endOnBox] = [endOnBox, startOnBox];
    }
    // Node boxes are drawn over the arrows, so a line between two box centres
    // has both ends -- including its head -- hidden under a box, and the
    // direction becomes invisible. Pull each end back to the box border.
    if (startOnBox) start = trimToBox(start, end);
    if (endOnBox) end = trimToBox(end, start);

    const line = svgNode("line", {
      x1: start.x,
      y1: start.y,
      x2: end.x,
      y2: end.y,
      stroke: strength > 0.01 ? "#3f6f9f" : "#b8c4cf",
      "stroke-width": (0.8 + strength * 4.2).toFixed(2),
      "stroke-dasharray": boundary ? "4 3" : "",
      "marker-end": strength > 0.01 ? "url(#compartment-arrow)" : "url(#compartment-arrow-weak)",
      opacity: strength > 0.01 ? 0.9 : 0.35,
    });
    const title = svgNode("title", {});
    title.textContent =
      `${flux.label}: ${formatCompartmentValue(rate)} ${flux.unit}\n` +
      `${flux.rate_expression}\n${flux.provenance}` +
      (flux.description ? `\n${flux.description}` : "");
    line.appendChild(title);
    diagram.appendChild(line);
  }

  for (const [key, position] of positions) {
    const series = timeline.compartments[key] || [];
    const value = series[index];
    const range = compartmentRange(series);
    // Fill is normalised per compartment over the whole run: the states have
    // incompatible units, so a shared scale would be meaningless.
    const fraction =
      range && range.high > range.low && Number.isFinite(value)
        ? (value - range.low) / (range.high - range.low)
        : 0;
    const group = svgNode("g", {});
    group.appendChild(
      svgNode("rect", {
        x: position.x - 52,
        y: position.y - 20,
        width: 104,
        height: 40,
        rx: 8,
        fill: position.compartment.provenance === "extension" ? "#f2e9dc" : "#e7eef5",
        stroke: position.compartment.kind === "pool" ? "#3f6f9f" : "#7f8f9f",
        "stroke-width": 1.2,
        "stroke-dasharray": position.compartment.provenance === "extension" ? "3 2" : "",
      })
    );
    group.appendChild(
      svgNode("rect", {
        x: position.x - 52,
        y: position.y + 12,
        width: 104 * Math.max(0, Math.min(1, fraction)),
        height: 8,
        rx: 3,
        fill: "#3f6f9f",
        opacity: 0.75,
      })
    );
    const symbol = svgText(position.x - 44, position.y - 4, position.compartment.symbol);
    symbol.setAttribute("fill", "#22303c");
    symbol.setAttribute("font-size", "13");
    group.appendChild(symbol);
    const reading = svgText(
      position.x - 44,
      position.y + 9,
      Number.isFinite(value) ? formatCompartmentValue(value) : "n/a"
    );
    reading.setAttribute("font-size", "11");
    group.appendChild(reading);
    const title = svgNode("title", {});
    title.textContent =
      `${position.compartment.label}\n` +
      `${Number.isFinite(value) ? formatCompartmentValue(value) : "n/a"} ${position.compartment.unit}\n` +
      `${position.compartment.kind}, ${position.compartment.provenance}` +
      (range ? `\nrun range ${formatCompartmentValue(range.low)} to ${formatCompartmentValue(range.high)}` : "") +
      (position.compartment.description ? `\n${position.compartment.description}` : "");
    group.appendChild(title);
    diagram.appendChild(group);
  }

  const hidden = (timeline.schema.compartments || []).length - visibleKeys.size;
  const unrecorded = (timeline.schema.fluxes || []).filter(
    (flux) => !timeline.fluxes[flux.key]
  );
  const minutes = timeline.times[index];
  $("compartment-time-label").textContent =
    Number.isFinite(minutes) ? `${Math.round(minutes)} min` : `step ${index}`;
  $("compartment-caption").textContent =
    "Bar fill inside each box and arrow thickness are normalised per variable " +
    "over this run; the states carry incompatible units, so thicknesses are not " +
    "comparable between arrows. Arrow values are instantaneous rates at the end " +
    "of a step, not amounts transferred during it. Dashed boxes are project " +
    "extensions to the published model; dashed arrows cross the patient boundary." +
    (hidden > 0
      ? ` ${hidden} signal or effect states hidden` +
        (hiddenEndpointFluxes > 0
          ? `, along with ${hiddenEndpointFluxes} fluxes that touch them.`
          : ".")
      : "") +
    (unrecorded.length > 0
      ? ` Not drawn as a rate: ${unrecorded.map((flux) => flux.label).join(", ")}.`
      : "");
  renderCompartmentTable(index, compartments);
}

function renderCompartmentTable(index, compartments) {
  const table = $("compartment-table");
  if (!table) return;
  const rows = compartments
    .map((compartment) => {
      const value = (compartmentTimeline.compartments[compartment.key] || [])[index];
      return `<tr><td>${escapeHtml(compartment.symbol)}</td><td>${escapeHtml(
        compartment.label
      )}</td><td>${escapeHtml(
        Number.isFinite(value) ? formatCompartmentValue(value) : "n/a"
      )}</td><td>${escapeHtml(compartment.unit)}</td><td>${escapeHtml(
        compartment.provenance
      )}</td></tr>`;
    })
    .join("");
  table.innerHTML =
    "<thead><tr><th>State</th><th>Compartment</th><th>Value</th><th>Unit</th><th>Source</th></tr></thead>" +
    `<tbody>${rows}</tbody>`;
}

async function loadCompartmentTimeline(csv) {
  const viewer = $("compartment-viewer");
  try {
    const timeline = await call("compartment_timeline", { csv, maxPoints: 400 });
    compartmentTimeline = timeline;
    if (!timeline.available) {
      viewer.hidden = true;
      setText("compartment-status", timeline.reason);
      setText("compartment-summary", "Not available for this run");
      return;
    }
    const slider = $("compartment-time");
    slider.max = String(Math.max(0, timeline.times.length - 1));
    slider.value = "0";
    viewer.hidden = false;
    setText(
      "compartment-status",
      `${timeline.schema.model_label}\n` +
        `${(timeline.schema.compartments || []).length} states, ` +
        `${Object.keys(timeline.fluxes).length} recorded fluxes, ` +
        `${timeline.times.length} of ${timeline.step_count} steps shown` +
        (timeline.stride > 1 ? ` (every ${timeline.stride}nd step)` : "")
    );
    setText("compartment-summary", timeline.schema.model_label);
    renderCompartmentDiagram();
    digitalTwin?.setTimeline(timeline);
  } catch (error) {
    compartmentTimeline = null;
    viewer.hidden = true;
    setText("compartment-status", errorMessage(error));
  }
}

$("compartment-time").addEventListener("input", renderCompartmentDiagram);
$("compartment-show-signals").addEventListener("change", renderCompartmentDiagram);

// --- 3D Digital Twin viewer ----------------------------------------------
// Alternative, WebGL-based view of the same compartmentTimeline data the SVG
// diagram above already draws -- no separate fetch, no re-implemented
// physiology (see digital-twin-data.js's header comment for the same house
// rule the SVG diagram follows). three.js is loaded lazily (dynamic import)
// so opening the app and using only the 2D diagram never fetches or parses
// the vendored ~1MB library.
let digitalTwin = null;
let digitalTwinLoading = null;

function currentDigitalTwinMinutes() {
  return Number($("digital-twin-time").value) || 0;
}

function showDigitalTwinCard(html) {
  const card = $("digital-twin-card");
  card.innerHTML = html;
  card.hidden = false; // instant, per the app's global no-transition rule
}

function renderCompartmentCard(compartmentKey) {
  const schemaEntry = (compartmentTimeline?.schema?.compartments || []).find((c) => c.key === compartmentKey);
  const series = compartmentTimeline?.compartments?.[compartmentKey];
  const value = interpolateSeries(compartmentTimeline?.times || [], series || [], currentDigitalTwinMinutes());
  showDigitalTwinCard(`
    <h4>${escapeHtml(schemaEntry?.label || compartmentKey)}</h4>
    <div>${Number.isFinite(value) ? value.toFixed(2) : "--"} ${escapeHtml(schemaEntry?.unit || "")}</div>
    ${schemaEntry?.description ? `<p>${escapeHtml(schemaEntry.description)}</p>` : ""}
  `);
}

function renderFluxCard(fluxKey, { isLiver = false } = {}) {
  const schemaEntry = (compartmentTimeline?.schema?.fluxes || []).find((f) => f.key === fluxKey);
  const series = compartmentTimeline?.fluxes?.[fluxKey];
  const value = interpolateSeries(compartmentTimeline?.times || [], series || [], currentDigitalTwinMinutes());
  const equation = lookupEquation(compartmentTimeline?.schema || {}, fluxKey);
  showDigitalTwinCard(`
    <h4>${escapeHtml(schemaEntry?.label || fluxKey)}</h4>
    <div>${Number.isFinite(value) ? value.toFixed(3) : "--"} ${escapeHtml(schemaEntry?.unit || "")}</div>
    ${equation ? `<code>${escapeHtml(equation)}</code>` : ""}
    ${isLiver ? `<p>${escapeHtml(LIVER_CARD_CAPTION)}</p>` : ""}
  `);
}

async function ensureDigitalTwin() {
  if (digitalTwin) return digitalTwin;
  if (!digitalTwinLoading) {
    digitalTwinLoading = import("./digital-twin-scene.js").then(({ createDigitalTwinScene }) => {
      digitalTwin = createDigitalTwinScene($("digital-twin-canvas"));
      // Register every callback before the first setTimeline() call below:
      // setTimeline() triggers an immediate render (buildOrgans() +
      // requestRender()), and that first frame is the one that computes and
      // reports flux-chip positions -- registering onFluxChipsUpdate after
      // it would silently miss that frame and show no chips until the next
      // interaction or play().
      digitalTwin.onTimeUpdate((minutes) => {
        const rounded = Math.round(minutes);
        $("digital-twin-time").value = String(rounded);
        setText("digital-twin-time-label", `${rounded} min`);
      });
      digitalTwin.onPlaybackEnded(() => {
        const button = $("digital-twin-play-btn");
        button.dataset.playing = "false";
        button.textContent = "Play";
      });
      digitalTwin.onPick((userData) => {
        if (userData.kind === "compartment") {
          renderCompartmentCard(userData.compartmentKey);
        } else if (userData.kind === "flux-proxy") {
          renderFluxCard(userData.fluxKey, { isLiver: userData.organId === "liver" });
        }
      });
      digitalTwin.onFluxChipsUpdate(updateFluxChips);
      if (compartmentTimeline) digitalTwin.setTimeline(compartmentTimeline);
      return digitalTwin;
    });
  }
  return digitalTwinLoading;
}

// Flux "chips" are small always-present buttons projected onto screen space
// at each stream's curve midpoint (fluxes aren't raycast -- see
// digital-twin-scene.js's pickOrganAt comment). Reused/repositioned every
// frame rather than recreated, to avoid needless DOM churn during playback.
const fluxChipElements = new Map();

function updateFluxChips(chips) {
  const container = $("digital-twin-canvas").parentElement;
  const seen = new Set();
  for (const chip of chips) {
    seen.add(chip.fluxKey);
    let el = fluxChipElements.get(chip.fluxKey);
    if (!el) {
      el = document.createElement("button");
      el.type = "button";
      el.className = "digital-twin-chip";
      el.addEventListener("click", () => renderFluxCard(chip.fluxKey, { isLiver: chip.fluxKey === "endogenous_production" }));
      container.appendChild(el);
      fluxChipElements.set(chip.fluxKey, el);
    }
    el.hidden = !chip.visible;
    el.textContent = chip.label;
    el.style.left = `${chip.x}px`;
    el.style.top = `${chip.y}px`;
  }
  for (const [fluxKey, el] of fluxChipElements) {
    if (!seen.has(fluxKey)) {
      el.remove();
      fluxChipElements.delete(fluxKey);
    }
  }
}

async function activateCompartmentMode(mode) {
  document.querySelectorAll("[data-compartment-mode]").forEach((b) => {
    b.classList.toggle("is-active", b.dataset.compartmentMode === mode);
  });
  document.querySelectorAll("[data-compartment-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.compartmentPanel !== mode;
  });
  if (mode === "twin") {
    await ensureDigitalTwin();
  } else {
    digitalTwin?.pause();
  }
}

document.querySelectorAll("[data-compartment-mode]").forEach((btn) => {
  btn.addEventListener("click", () => activateCompartmentMode(btn.dataset.compartmentMode));
});

$("digital-twin-time").addEventListener("input", (event) => {
  const minutes = Number(event.target.value) || 0;
  setText("digital-twin-time-label", `${minutes} min`);
  digitalTwin?.setSimMinutes(minutes);
});

$("digital-twin-play-btn").addEventListener("click", async () => {
  const twin = await ensureDigitalTwin();
  const button = $("digital-twin-play-btn");
  const nowPlaying = button.dataset.playing === "true";
  if (nowPlaying) {
    twin.pause();
    button.dataset.playing = "false";
    button.textContent = "Play";
  } else {
    twin.play();
    button.dataset.playing = "true";
    button.textContent = "Pause";
  }
});

$("digital-twin-speed").addEventListener("change", (event) => {
  digitalTwin?.setSpeed(Number(event.target.value) || 1);
});

$("run-btn").addEventListener("click", runSelectedWorkflow);
$("view-results-btn").addEventListener("click", async () => {
  setActiveView("results");
  await activateCompartmentMode("twin");
});
$("run-cancel-btn").addEventListener("click", cancelSelectedWorkflow);
$("guide-btn").addEventListener("click", openUserGuide);
$("settings-save-btn").addEventListener("click", saveWorkbenchSettings);
$("settings-reset-btn").addEventListener("click", resetWorkbenchSettings);
$("settings-output-browse-btn").addEventListener("click", chooseSettingsOutputFolder);
$("settings-guide-btn").addEventListener("click", openUserGuide);
$("settings-install-guide-btn").addEventListener("click", openInstallGuide);
$("settings-docs-btn").addEventListener("click", openDocsHome);
$("settings-website-btn").addEventListener("click", openProjectWebsite);
$("refresh-btn").addEventListener("click", loadWorkflows);
$("history-btn").addEventListener("click", loadHistory);
$("diagnostics-btn").addEventListener("click", runDiagnostics);
$("open-output-btn").addEventListener("click", openOutputFolder);
$("install-engine-btn").addEventListener("click", openSdkUpdateTerminal);
$("update-refresh-btn").addEventListener("click", refreshSoftwareVersions);
$("update-download-btn").addEventListener("click", openAppDownloads);
$("update-docs-btn").addEventListener("click", openUpdateDocs);
$("update-copy-btn").addEventListener("click", copyUpdateCommand);
$("update-terminal-btn").addEventListener("click", openSdkUpdateTerminal);
$("open-run-folder-btn").addEventListener("click", openLatestRunFolder);
$("open-report-btn").addEventListener("click", openLatestReport);
$("output-browse-btn").addEventListener("click", chooseRunOutputFolder);
$("preview-btn").addEventListener("click", previewCsv);
$("csv-browse-btn").addEventListener("click", chooseResultsCsv);
$("show-all-columns").addEventListener("change", () => {
  if (lastPreview) renderTable(lastPreview.columns || [], lastPreview.rows || []);
});
$("mdmp-btn").addEventListener("click", certifyMdmp);
$("open-csv-btn").addEventListener("click", openLoadedCsv);
$("open-certificate-btn").addEventListener("click", openLatestCertificate);
$("academic-export-btn").addEventListener("click", exportAcademicBundle);
$("academic-run-browse-btn").addEventListener("click", chooseAcademicRunFolder);
$("academic-open-metadata-btn").addEventListener("click", openAcademicMetadata);
$("academic-open-audit-btn").addEventListener("click", openAcademicAudit);
$("academic-open-guide-btn").addEventListener("click", openAcademicGuide);
$("ai-start-btn").addEventListener("click", startAi);
$("ai-check-btn").addEventListener("click", checkAi);
$("ai-models-btn").addEventListener("click", listAiModels);
$("ai-question").addEventListener("input", refreshActionAvailability);
$("ai-ask-btn").addEventListener("click", askAi);
$("molecule-refresh-btn").addEventListener("click", loadMolecules);
$("open-structural-folder-btn").addEventListener("click", openStructuralFolder);
$("molecule-viewer-reset-btn").addEventListener("click", resetMoleculeViewer);
$("molecule-viewer-rotate-btn").addEventListener("click", () => {
  setMoleculeAutoRotate(!moleculeViewer.autoRotate);
});
$("molecule-viewer-close-btn").addEventListener("click", closeMoleculeViewer);
$("genomics-run-btn").addEventListener("click", runGenomicsSimulation);
$("genomics-open-btn").addEventListener("click", openGenomicsPlot);
$("tissue-run-btn").addEventListener("click", runTissueStressTest);
$("tissue-open-btn").addEventListener("click", openTissuePlot);
$("mechanistic-status-btn").addEventListener("click", loadMechanisticStatus);
$("mechanistic-model-browse-btn").addEventListener("click", () => chooseResearchModel(
  "mechanistic-model",
  "mechanistic-model-browse-btn",
  "mechanistic-status",
  "Choose an SBML reference model",
  "SBML model",
  ["xml", "sbml"]
));
$("mechanistic-inspect-btn").addEventListener("click", inspectMechanisticModel);
$("mechanistic-run-btn").addEventListener("click", runMechanisticModel);
$("mechanistic-open-folder-btn").addEventListener("click", openMechanisticFolder);
$("mechanistic-open-report-btn").addEventListener("click", openMechanisticReport);
$("mechanistic-open-results-btn").addEventListener("click", openMechanisticResults);
$("cross-scale-status-btn").addEventListener("click", loadCrossScaleStatus);
$("copasi-model-browse-btn").addEventListener("click", () => chooseResearchModel(
  "copasi-model",
  "copasi-model-browse-btn",
  "copasi-status",
  "Choose a COPASI model",
  "COPASI model",
  ["cps"]
));
$("copasi-inspect-btn").addEventListener("click", inspectCopasiModel);
$("copasi-run-btn").addEventListener("click", runCopasiAnalysis);
$("copasi-open-btn").addEventListener("click", openCopasiBundle);
$("cellml-model-browse-btn").addEventListener("click", () => chooseResearchModel(
  "cellml-model",
  "cellml-model-browse-btn",
  "cellml-status",
  "Choose a CellML model",
  "CellML model",
  ["cellml", "xml"]
));
$("cellml-inspect-btn").addEventListener("click", inspectCellmlModel);
$("cellml-validate-btn").addEventListener("click", validateCellmlModel);
$("cellml-open-btn").addEventListener("click", openCellmlBundle);
$("fmi-model-browse-btn").addEventListener("click", () => chooseResearchModel(
  "fmi-model",
  "fmi-model-browse-btn",
  "fmi-status",
  "Choose a Functional Mock-up Unit",
  "FMI model",
  ["fmu"]
));
$("fmi-inspect-btn").addEventListener("click", inspectFmiModel);
$("fmi-run-btn").addEventListener("click", runFmiModel);
$("fmi-open-btn").addEventListener("click", openFmiBundle);
$("fmi-results-btn").addEventListener("click", openFmiResults);
$("binding-query-btn").addEventListener("click", queryBindingEvidence);
$("binding-open-btn").addEventListener("click", openBindingBundle);
$("binding-csv-btn").addEventListener("click", openBindingCsv);
$("evidence-refresh-btn").addEventListener("click", loadEvidenceConnectors);
$("evidence-search").addEventListener("input", () => renderEvidenceConnectors(evidenceConnectors));
$("evidence-level").addEventListener("change", () => renderEvidenceConnectors(evidenceConnectors));

// Foundation AI & Visualizer Logic
let activeChartTab = "glucofm";

const CHART_DESCRIPTIONS = {
  glucofm: {
    title: "IINTS GlucoFM v2 independent method reproduction",
    desc: "A mask-preserving 288-position daily grid is decomposed causally into state and event streams. Both streams use 24 one-hour patches; the fused token and pooled embedding dimension is 128. Official Google weights are not bundled."
  },
  embedding: {
    title: "Last checkpoint-backed embedding",
    desc: "The chart is populated only after a trained local checkpoint has processed a selected CGM CSV. The checkpoint SHA-256 and observed-grid coverage remain visible in the result log."
  },
  clarke: {
    title: "Clarke Error Grid Analysis (demonstration)",
    desc: "Clarke zone classification per Clarke et al. (Diabetes Care, 1987). The zone percentages shown on the chart are counted from the plotted pairs by the shared classifier in science.js. No paired evaluation data is loaded in the workbench yet, so the plotted pairs are a fixed-seed synthetic demonstration and the chart is labelled as such. ISO 15197 is a bench standard for glucose meters and does not apply to model predictions."
  }
};

// Real held-out evaluation output, produced by
// `research/export_desktop_evidence.py`. When the file is present the Clarke
// chart shows measured pairs; when it is absent the chart falls back to a
// clearly-labelled synthetic demonstration. It must never silently show
// generated data as if it were a result.
let forecastEvidence = null;
let forecastEvidenceState = "loading";

async function loadForecastEvidence() {
  try {
    const res = await fetch("./evidence/forecast_evidence.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    if (!payload?.pooled?.model?.zone_percentages || !payload?.scatter?.reference) {
      throw new Error("evidence file missing required fields");
    }
    forecastEvidence = payload;
    forecastEvidenceState = "ready";
  } catch (err) {
    forecastEvidence = null;
    forecastEvidenceState = "absent";
    console.info("No forecast evidence loaded; Clarke chart falls back to demo.", err);
  }
  if (activeChartTab === "clarke") renderFoundationChart("clarke");
}

loadForecastEvidence();

/*
 * Cross-fold evidence. The scatter above is one checkpoint on the two subjects
 * it held out; that is a demonstration of the method, not a measure of it. The
 * cross-fold file pools several checkpoints over their own held-out subjects
 * and puts a subject-level interval on every claim. When it is present the
 * panel text is qualified by it, so the app can never state a single-fold
 * advantage that the subject-level interval does not support.
 */
let crossfoldEvidence = null;
let crossfoldEvidenceState = "loading";

async function loadCrossfoldEvidence() {
  try {
    const res = await fetch("./evidence/crossfold_evidence.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    if (!payload?.paired_model_minus_persistence?.clarke_zone_a ||
        !payload?.primary_outcome_hypoglycemia_detection?.paired_model_minus_persistence) {
      throw new Error("cross-fold evidence missing required fields");
    }
    crossfoldEvidence = payload;
    crossfoldEvidenceState = "ready";
  } catch (err) {
    crossfoldEvidence = null;
    crossfoldEvidenceState = "absent";
    console.info("No cross-fold evidence loaded; panel text stays single-fold.", err);
  }
  if (activeChartTab === "clarke") renderFoundationChart("clarke");
}

loadCrossfoldEvidence();

/** Subject-level qualification of the single-fold numbers. Empty when absent. */
function crossfoldSentence() {
  if (crossfoldEvidenceState !== "ready") return "";
  const c = crossfoldEvidence;
  const za = c.paired_model_minus_persistence.clarke_zone_a;
  const hy = c.primary_outcome_hypoglycemia_detection;
  const hp = hy.paired_model_minus_persistence;
  const dyn = c.summary.model.directional.trend_dynamics;
  const zaVerdict = za.model_better_at_95pct
    ? "which excludes zero"
    : "which includes zero, so this advantage is not established";
  const hypoVerdict = hp.estimate < 0
    ? `the model detects ${Math.abs(hp.estimate).toFixed(1)} percentage points FEWER ` +
      `hypoglycemic windows than carrying the last reading forward ` +
      `(95% CI ${hp.ci_low.toFixed(1)} to ${hp.ci_high.toFixed(1)})`
    : `the model detects ${hp.estimate.toFixed(1)} percentage points more hypoglycemic ` +
      `windows than the baseline (95% CI ${hp.ci_low.toFixed(1)} to ${hp.ci_high.toFixed(1)})`;
  const flat = dyn.flat_forecast
    ? ` The forecast reproduces only ${(dyn.rate_attenuation * 100).toFixed(0)}% of the ` +
      `observed rate-of-change spread, so it carries level information but little trend.`
    : "";
  return (
    ` Across ${c.folds.length} folds covering ${c.n_subjects} held-out subjects, the ` +
    `paired Zone A advantage over the baseline is ${za.estimate >= 0 ? "+" : ""}` +
    `${za.estimate.toFixed(1)} points (95% CI ${za.ci_low.toFixed(1)} to ` +
    `${za.ci_high.toFixed(1)}), ${zaVerdict}. On the primary safety outcome, ` +
    `${hypoVerdict}.${flat}`
  );
}

/** Description text for the Clarke tab, derived from the evidence provenance. */
function clarkeDescription() {
  if (forecastEvidenceState !== "ready") return CHART_DESCRIPTIONS.clarke.desc;
  const p = forecastEvidence.provenance;
  const m = forecastEvidence.pooled.model;
  const b = forecastEvidence.pooled.persistence;
  const sl = forecastEvidence.subject_level_zone_a;
  const meal = p.meal_announcement_minutes
    ? ` Meals are announced ${p.meal_announcement_minutes} min ahead, which is information the model receives about the near future.`
    : "";
  return (
    `Clarke zone classification (Clarke et al., 1987) of ${m.n_pairs.toLocaleString()} forecast ` +
    `endpoints at a ${p.horizon_minutes}-minute horizon, on subjects ${(p.test_subjects || []).join(" and ")} — ` +
    `held out of both training and model selection. ` +
    `Model: ${m.zone_percentages.A.toFixed(1)}% Zone A, MAE ${m.mae.toFixed(1)} mg/dL. ` +
    `Carrying the last reading forward: ${b.zone_percentages.A.toFixed(1)}% Zone A, MAE ${b.mae.toFixed(1)} mg/dL. ` +
    `Zone A ranges from ${sl.min.toFixed(1)}% to ${sl.max.toFixed(1)}% across only ${sl.n_subjects} subjects, ` +
    `so the pooled figure is far more precise than the evidence warrants.${meal}` +
    crossfoldSentence()
  );
}

function renderFoundationChart(tab) {
  activeChartTab = tab;
  const canvas = $("foundation-chart-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  const info = CHART_DESCRIPTIONS[tab] || CHART_DESCRIPTIONS.glucofm;
  let title = info.title;
  let desc = info.desc;
  if (tab === "clarke" && forecastEvidenceState === "ready") {
    title = "Clarke Error Grid Analysis (held-out evaluation)";
    desc = clarkeDescription();
  }
  $("chart-explanation").innerHTML = `<h3>${title}</h3><p>${desc}</p>`;

  if (tab === "clarke") {
    drawClarkeErrorGridChart(ctx, w, h);
  } else if (tab === "glucofm") {
    drawGlucoFMDecompositionChart(ctx, w, h);
  } else if (tab === "embedding") {
    drawEmbeddingChart(ctx, w, h);
  }
}

let lastGlucoFMEmbedding = null;

function drawEmbeddingChart(ctx, w, h) {
  const values = Array.isArray(lastGlucoFMEmbedding) ? lastGlucoFMEmbedding : [];
  ctx.fillStyle = "#1e293b";
  ctx.font = "bold 14px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Last checkpoint-backed 128D embedding", 38, 34);
  if (!values.length) {
    ctx.fillStyle = "#64748b";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("No embedding has been generated in this session.", 38, 72);
    return;
  }
  const left = 48;
  const right = w - 32;
  const top = 64;
  const bottom = h - 48;
  const maxAbs = Math.max(...values.map((value) => Math.abs(Number(value) || 0)), 1e-6);
  ctx.strokeStyle = "#94a3b8";
  ctx.beginPath();
  ctx.moveTo(left, (top + bottom) / 2);
  ctx.lineTo(right, (top + bottom) / 2);
  ctx.stroke();
  const barWidth = (right - left) / values.length;
  values.forEach((raw, index) => {
    const value = Number(raw) || 0;
    const height = (Math.abs(value) / maxAbs) * ((bottom - top) / 2 - 8);
    const x = left + index * barWidth;
    const zero = (top + bottom) / 2;
    ctx.fillStyle = value >= 0 ? "#1d4ed8" : "#b45309";
    ctx.fillRect(x, value >= 0 ? zero - height : zero, Math.max(1, barWidth - 1), height);
  });
  ctx.fillStyle = "#475569";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(`Dimensions: ${values.length}; scale: +/-${maxAbs.toFixed(3)}`, left, h - 18);
}

function drawGlucoFMDecompositionChart(ctx, w, h) {
  const boxes = [
    { x: 45, y: 135, width: 150, title: "24-hour CGM", detail: "288 x 5-minute grid" },
    { x: 245, y: 72, width: 160, title: "State stream", detail: "causal Gaussian filter" },
    { x: 245, y: 205, width: 160, title: "Event stream", detail: "observed - state" },
    { x: 455, y: 72, width: 165, title: "24 state patches", detail: "12 positions each" },
    { x: 455, y: 205, width: 165, title: "24 event patches", detail: "12 positions each" },
    { x: 680, y: 135, width: 175, title: "Fused representation", detail: "24 tokens x 128D" }
  ];
  ctx.fillStyle = "#0f172a";
  ctx.font = "bold 15px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("IINTS GlucoFM v2 independent method reproduction", 38, 34);
  ctx.fillStyle = "#475569";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("Missingness is preserved with a physical observation mask; no official Google weights are bundled.", 38, 56);

  const arrows = [
    [195, 170, 245, 110], [195, 170, 245, 242],
    [405, 110, 455, 110], [405, 242, 455, 242],
    [620, 110, 680, 170], [620, 242, 680, 170]
  ];
  ctx.strokeStyle = "#64748b";
  ctx.lineWidth = 2;
  arrows.forEach(([x1, y1, x2, y2]) => {
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  });
  boxes.forEach((box, index) => {
    ctx.fillStyle = index === boxes.length - 1 ? "#e0f2fe" : "#f8fafc";
    ctx.strokeStyle = index === boxes.length - 1 ? "#0369a1" : "#94a3b8";
    ctx.lineWidth = 1.5;
    ctx.fillRect(box.x, box.y, box.width, 70);
    ctx.strokeRect(box.x, box.y, box.width, 70);
    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(box.title, box.x + box.width / 2, box.y + 28);
    ctx.fillStyle = "#475569";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(box.detail, box.x + box.width / 2, box.y + 49);
  });
  ctx.fillStyle = "#475569";
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Pretraining: masked contextual reconstruction + temporal-dynamics objective with an EMA target encoder.", 38, h - 32);
}

function drawClarkeErrorGridChart(ctx, w, h) {
  const pad = 50;
  const size = Math.min(w - 2 * pad, h - 2 * pad);
  const startX = (w - size) / 2;
  const startY = (h - size) / 2 + 10;

  // Background
  ctx.fillStyle = "#f8fafd";
  ctx.fillRect(startX, startY, size, size);

  // Zone A Polygon
  ctx.fillStyle = "rgba(30, 142, 62, 0.12)";
  ctx.beginPath();
  ctx.moveTo(startX, startY + size);
  ctx.lineTo(startX + (70/400)*size, startY + size - (56/400)*size);
  ctx.lineTo(startX + size, startY + size - (320/400)*size);
  ctx.lineTo(startX + size, startY);
  ctx.lineTo(startX + (58.33/400)*size, startY + size - (70/400)*size);
  ctx.lineTo(startX, startY + size - (70/400)*size);
  ctx.closePath();
  ctx.fill();

  // Grid box
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(startX, startY, size, size);

  // Diagonal line
  ctx.strokeStyle = "#64748b";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(startX, startY + size);
  ctx.lineTo(startX + size, startY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Scatter points.
  //
  // No paired evaluation data is loaded in the workbench yet, so these points
  // are generated. They are drawn with a fixed seed (reproducible across
  // repaints), the zone percentages below are counted from these exact points
  // by the shared classifier, and the chart is labelled as synthetic. Wire
  // real (reference, predicted) arrays into this function and both the
  // scatter and the percentages follow automatically.
  const AXIS_MAX = 400;
  const plot = (ref, pred, radius) => {
    const px = startX + (Math.min(ref, AXIS_MAX) / AXIS_MAX) * size;
    const py = startY + size - (Math.min(pred, AXIS_MAX) / AXIS_MAX) * size;
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();
  };

  let zoneA, zoneB, hazardous, nPairs, isReal = false;

  if (forecastEvidenceState === "ready") {
    // Measured pairs from a held-out evaluation. The percentages come from the
    // exporter, which computed them over ALL pairs; the scatter may be a
    // subsample for legibility, so the two are deliberately not recomputed here.
    const ev = forecastEvidence;
    const refs = ev.scatter.reference;
    const preds = ev.scatter.predicted;
    ctx.fillStyle = "rgba(26, 115, 232, 0.35)";
    for (let i = 0; i < refs.length; i++) plot(refs[i], preds[i], 1.6);
    const m = ev.pooled.model;
    zoneA = m.zone_percentages.A;
    zoneB = m.zone_percentages.B;
    hazardous = m.hazardous_pct;
    nPairs = m.n_pairs;
    isReal = true;
  } else {
    // No evaluation output present. Generate a demonstration with a fixed seed
    // (reproducible across repaints), count the zones from exactly these
    // points, and say plainly that they are synthetic.
    const rand = seededRandom(42);
    const refVals = [];
    const predVals = [];
    ctx.fillStyle = "rgba(26, 115, 232, 0.6)";
    for (let i = 0; i < 200; i++) {
      const ref = 60 + rand() * 300;
      const noise = (rand() - 0.5) * (ref * 0.08 + 4);
      const pred = Math.max(40, Math.min(380, ref + noise));
      refVals.push(ref);
      predVals.push(pred);
      plot(ref, pred, 2.5);
    }
    const ega = clarkeErrorGrid(refVals, predVals);
    zoneA = ega.percentages.A;
    zoneB = ega.percentages.B;
    hazardous = ega.hazardousPct;
    nPairs = ega.nPairs;
  }

  ctx.fillStyle = "#137333";
  ctx.font = "bold 13px system-ui, sans-serif";
  ctx.fillText(`Zone A: ${zoneA.toFixed(1)}% clinically accurate (n = ${nPairs.toLocaleString()})`,
               startX + 20, startY + 30);
  ctx.fillStyle = "#5f6368";
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText(`Zone B: ${zoneB.toFixed(1)}% | Zone C/D/E: ${hazardous.toFixed(1)}%`,
               startX + 20, startY + 48);
  ctx.fillText("Reference Glucose (mg/dL) →", startX + size/2 - 60, startY + size + 25);

  if (isReal) {
    const p = forecastEvidence.provenance;
    const b = forecastEvidence.pooled.persistence;
    ctx.fillStyle = "#5f6368";
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText(
      `${p.horizon_minutes} min horizon | held-out subjects ${(p.test_subjects || []).join(", ")} | ` +
      `persistence baseline: ${b.zone_percentages.A.toFixed(1)}% Zone A`,
      startX + 20, startY + size - 12);
  } else {
    drawSyntheticBanner(ctx, startX + 20, startY + size - 26);
  }
}

// Chart tab switching
document.querySelectorAll("[data-chart-tab]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-chart-tab]").forEach(b => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    renderFoundationChart(btn.dataset.chartTab);
  });
});

// Scientific portfolio / evidence dossier button
$("eucys-playbook-btn")?.addEventListener("click", async () => {
  setText("foundation-status", "Generating the evidence dossier. Figures without required measured inputs are skipped rather than fabricated...");
  try {
    const result = await invoke("generate_eucys_playbook", { outputDir: "results/scientific_portfolio" });
    setText("foundation-status", `Evidence portfolio generated successfully!\nTotal Figures: ${result.data?.total_figures}\nInteractive Dossier: ${result.data?.index_html_path}\nManifest: ${result.data?.manifest_json_path}`);
    renderFoundationChart("clarke");
  } catch (err) {
    setText("foundation-status", `Evidence portfolio generation failed: ${err}`);
  }
});

// Foundation model inputs and evidence-only actions
let foundationArenaResultFiles = [];

$("foundation-glucofm-train-source-browse-btn")?.addEventListener("click", () => chooseLocalPath({
  inputId: "foundation-glucofm-train-source",
  buttonId: "foundation-glucofm-train-source-browse-btn",
  statusId: "foundation-status",
  title: "Choose a multi-subject CGM training dataset",
  filters: [{ name: "CGM table", extensions: ["csv", "tsv", "txt", "parquet", "pq"] }],
  selectedLabel: "Pretraining dataset selected"
}));

$("foundation-glucofm-train-output-browse-btn")?.addEventListener("click", () => chooseLocalPath({
  inputId: "foundation-glucofm-train-output",
  buttonId: "foundation-glucofm-train-output-browse-btn",
  statusId: "foundation-status",
  title: "Choose the GlucoFM training output folder",
  directory: true,
  selectedLabel: "Training output folder selected"
}));

$("foundation-glucofm-csv-browse-btn")?.addEventListener("click", () => chooseLocalPath({
  inputId: "foundation-glucofm-csv",
  buttonId: "foundation-glucofm-csv-browse-btn",
  statusId: "foundation-status",
  title: "Choose a 24-hour CGM CSV",
  filters: [{ name: "CSV data", extensions: ["csv"] }],
  selectedLabel: "CGM CSV selected"
}));

$("foundation-glucofm-checkpoint-browse-btn")?.addEventListener("click", () => chooseLocalPath({
  inputId: "foundation-glucofm-checkpoint",
  buttonId: "foundation-glucofm-checkpoint-browse-btn",
  statusId: "foundation-status",
  title: "Choose a trained GlucoFM checkpoint",
  filters: [{ name: "PyTorch checkpoint", extensions: ["pt"] }],
  selectedLabel: "Checkpoint selected"
}));

$("foundation-glucofm-checkpoint")?.addEventListener("input", () => {
  const path = $("foundation-glucofm-checkpoint").value.trim();
  setText("metric-glucofm-checkpoint", path ? "Selected" : "Not selected");
});

$("foundation-arena-results-browse-btn")?.addEventListener("click", async () => {
  if (typeof nativeOpenDialog !== "function") {
    setText("foundation-status", "The native file chooser is available only in the installed app.");
    return;
  }
  try {
    const selected = await nativeOpenDialog({
      title: "Choose comparable foundation evaluation artifacts",
      directory: false,
      multiple: true,
      filters: [{ name: "IINTS evaluation JSON", extensions: ["json"] }]
    });
    foundationArenaResultFiles = (Array.isArray(selected) ? selected : [selected])
      .filter((path) => typeof path === "string" && path.trim());
    $("foundation-arena-results").value = foundationArenaResultFiles.join("; ");
    setText(
      "foundation-status",
      foundationArenaResultFiles.length
        ? `${foundationArenaResultFiles.length} evaluation artifact(s) selected. Comparability is validated before ranking.`
        : "No evaluation artifacts selected."
    );
  } catch (error) {
    setText("foundation-status", `Could not select evaluation artifacts: ${errorMessage(error)}`);
  }
});

$("foundation-arena-btn")?.addEventListener("click", async () => {
  if (!foundationArenaResultFiles.length) {
    setText("foundation-status", "Choose measured evaluation JSON files first. The app does not generate placeholder scores.");
    return;
  }
  setText("foundation-status", "Validating benchmark contracts and comparing supplied evidence...");
  try {
    const result = await invoke("run_foundation_arena", {
      outputDir: "results/foundation_arena",
      resultFiles: foundationArenaResultFiles
    });
    const modelCount = result.data?.total_models_evaluated ?? result.data?.models?.length ?? 0;
    setText("metric-foundation-evaluation", `${modelCount} model${modelCount === 1 ? "" : "s"}`);
    setText(
      "foundation-status",
      `Evidence comparison complete.\nBenchmark: ${result.data?.benchmark_id}\nModels: ${modelCount}\nReport: ${result.data?.report_md_path}`
    );
  } catch (error) {
    setText("foundation-status", `Evidence comparison failed: ${errorMessage(error)}`);
  }
});

$("foundation-glucofm-pretrain-btn")?.addEventListener("click", async () => {
  const source = $("foundation-glucofm-train-source").value.trim();
  const outputDir = $("foundation-glucofm-train-output").value.trim();
  if (!source || !outputDir) {
    setText("foundation-status", "Choose a multi-subject CGM dataset and an output folder first.");
    return;
  }
  setText("foundation-status", "Pretraining the independent GlucoFM reproduction. This can take a long time; the UI remains responsive...");
  try {
    const result = await invoke("pretrain_glucofm", {
      source,
      outputDir,
      glucoseColumn: $("foundation-glucose-column").value.trim() || null,
      timestampColumn: $("foundation-timestamp-column").value.trim() || null,
      subjectColumn: "subject_id",
      epochs: Number($("foundation-glucofm-epochs").value || 120),
      batchSize: Number($("foundation-glucofm-batch-size").value || 128),
      device: "auto",
      seed: 42
    });
    const checkpoint = result.data?.checkpoint_path;
    if (checkpoint) {
      $("foundation-glucofm-checkpoint").value = checkpoint;
      setText("metric-glucofm-checkpoint", "Trained");
    }
    setText(
      "foundation-status",
      `Pretraining complete.\\nTrain/validation subjects: ${result.data?.train_subjects}/${result.data?.validation_subjects}\\nBest validation loss: ${Number(result.data?.best_validation_loss).toFixed(6)}\\nCheckpoint: ${checkpoint}\\nReport: ${result.data?.report_path}`
    );
  } catch (error) {
    setText("foundation-status", `GlucoFM pretraining failed: ${errorMessage(error)}`);
  }
});

$("foundation-glucofm-btn")?.addEventListener("click", async () => {
  const csv = $("foundation-glucofm-csv").value.trim();
  const checkpoint = $("foundation-glucofm-checkpoint").value.trim();
  if (!csv || !checkpoint) {
    setText("foundation-status", "Choose both a 24-hour CGM CSV and a trained GlucoFM checkpoint.");
    return;
  }
  setText("foundation-status", "Extracting a checkpoint-backed 128D representation...");
  try {
    const result = await invoke("extract_glucofm_embedding", {
      csv,
      checkpoint,
      glucoseColumn: $("foundation-glucose-column").value.trim() || null,
      timestampColumn: $("foundation-timestamp-column").value.trim() || null
    });
    lastGlucoFMEmbedding = result.data?.embedding || null;
    setText("metric-glucofm-checkpoint", "Verified");
    setText(
      "foundation-status",
      `Embedding generated.\nModel: ${result.data?.model}\nOfficial Google checkpoint: no\nLatent dimension: ${result.data?.latent_dim}\nObserved-grid coverage: ${((result.data?.input_coverage || 0) * 100).toFixed(1)}%\nCheckpoint SHA-256: ${result.data?.checkpoint_sha256}`
    );
    document.querySelectorAll("[data-chart-tab]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.chartTab === "embedding");
    });
    renderFoundationChart("embedding");
  } catch (error) {
    setText("foundation-status", `GlucoFM embedding failed: ${errorMessage(error)}`);
  }
});

initializeNativeInteractionPolicy();
initializeFormState();
initializeMoleculeViewer();
initializeKeyboardShortcuts();
const initialSettings = initializeSettings();
initializeNavigation();
refreshActionAvailability();
await loadStatus();
if (initialSettings.autoDiagnostics) {
  await runDiagnostics();
} else {
  $("diagnostics-grid").replaceChildren(
    diagnosticRow("Diagnostics", "Automatic startup check disabled in Settings.", "info")
  );
}
