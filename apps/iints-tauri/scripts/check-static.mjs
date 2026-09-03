import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = dirname(dirname(fileURLToPath(import.meta.url)));

const required = [
  "frontend/index.html",
  "frontend/app-mark.png",
  "frontend/iints-logo.png",
  "frontend/styles.css",
  "frontend/main.js",
  "frontend/digital-twin-data.js",
  "scripts/build-brand-icons.py",
  "src-tauri/tauri.conf.json",
  "src-tauri/capabilities/main.json",
  "src-tauri/icons/icon.icns",
  "src-tauri/icons/icon.ico",
  "src-tauri/icons/icon.png",
  "src-tauri/icons/icon-source.png",
  "src-tauri/Cargo.toml",
  "src-tauri/src/main.rs"
];

for (const file of required) {
  if (!existsSync(join(appRoot, file))) {
    console.error(`Missing ${file}`);
    process.exit(1);
  }
}

const html = readFileSync(join(appRoot, "frontend/index.html"), "utf8");
// Concatenated so the id/command-reference scans below (referencedIds,
// invokedCommands) also cover digital-twin-data.js, not just main.js -- it
// never calls Tauri commands or $()/setText() directly today, but this keeps
// the check meaningful as that module grows.
const script =
  readFileSync(join(appRoot, "frontend/main.js"), "utf8") +
  "\n" +
  readFileSync(join(appRoot, "frontend/digital-twin-data.js"), "utf8");
const styles = readFileSync(join(appRoot, "frontend/styles.css"), "utf8");
const tauriConfig = JSON.parse(readFileSync(join(appRoot, "src-tauri/tauri.conf.json"), "utf8"));
const capabilities = JSON.parse(readFileSync(join(appRoot, "src-tauri/capabilities/main.json"), "utf8"));
if (!html.includes("Research only") || !html.includes("Not a medical device")) {
  console.error("frontend/index.html must contain the research-only disclaimer.");
  process.exit(1);
}

if (!html.includes("Reproducibility package") || !html.includes("academic-export-btn")) {
  console.error("frontend/index.html must expose the academic reproducibility package.");
  process.exit(1);
}

if (!html.includes("skip-link") || !html.includes("guide-btn") || !html.includes("results-status")) {
  console.error("frontend/index.html must expose accessible navigation, the user guide, and visible result status.");
  process.exit(1);
}

if (!html.includes("ai-model-options") || !script.includes("renderAiAnswer")) {
  console.error("The local AI workspace must expose model choices and readable structured output.");
  process.exit(1);
}

if (styles.includes("linear-gradient") || styles.includes("box-shadow: var(--shadow)")) {
  console.error("The academic workbench must avoid decorative gradients and card shadows.");
  process.exit(1);
}

if (
  html.includes("section-number")
  || html.includes("workbench-mark.svg")
  || !html.includes('<span class="brand-wordmark">IINTS-AF</span>')
  || !html.includes("code-panel-header")
) {
  console.error("The workbench must use text-only IINTS-AF branding and structured code panels without decorative numbering.");
  process.exit(1);
}

if (
  !html.includes('data-view="settings"')
  || !html.includes('data-view-panel="settings"')
  || !html.includes("settings-save-btn")
  || !html.includes("settings-guide-btn")
  || !html.includes("install-engine-btn")
  || !html.includes("Install or update Python SDK")
  || !script.includes("SETTINGS_STORAGE_KEY")
  || !script.includes("isAllowedLocalAiHost")
  || !script.includes("desktop_app_info")
  || !script.includes("tauri-beta-latest")
) {
  console.error("The workbench must expose persistent local settings, documentation, and safe app/SDK maintenance actions.");
  process.exit(1);
}

if (!styles.includes("user-select: none") || !styles.includes("user-select: text")) {
  console.error("The native shell must disable accidental selection while keeping research output copyable.");
  process.exit(1);
}

if (!script.includes("initializeNativeInteractionPolicy") || !script.includes('"contextmenu"')) {
  console.error("The native shell must suppress the browser context menu outside copyable research output.");
  process.exit(1);
}

if (
  !html.includes("molecule-viewer-canvas")
  || !html.includes("molecule-viewer-rotate-btn")
  || !script.includes("openMoleculeViewer")
  || !script.includes("generate_molecule_pae")
  || !script.includes("reveal_path")
  || !styles.includes(".molecule-viewer-panel")
) {
  console.error("Bundled AlphaFold assets must expose local 3D inspection, PAE generation, and safe artifact reveal actions.");
  process.exit(1);
}

const pickerIds = [
  "settings-output-browse-btn",
  "output-browse-btn",
  "csv-browse-btn",
  "academic-run-browse-btn",
  "mechanistic-model-browse-btn",
  "copasi-model-browse-btn",
  "cellml-model-browse-btn",
  "fmi-model-browse-btn"
];
if (
  pickerIds.some((id) => !html.includes(`id="${id}"`))
  || !script.includes("nativeOpenDialog")
  || !script.includes("chooseLocalPath")
  || !styles.includes(".path-picker")
) {
  console.error("Every path-based workflow must expose the shared native file/folder selector.");
  process.exit(1);
}

const permissions = capabilities.permissions || [];
if (!permissions.includes("dialog:allow-open")) {
  console.error("The native selectors require only the user-mediated dialog:allow-open permission.");
  process.exit(1);
}
// The app may check for and install its own signed updates, and restart to
// apply them -- both explicit, narrowly-scoped permissions rather than the
// plugins' full default sets. Everything else stays off: no filesystem,
// shell, or general HTTP access.
const allowedUpdaterAndProcessPermissions = new Set([
  "updater:allow-check",
  "updater:allow-download-and-install",
  "process:allow-restart"
]);
if (
  permissions.some(
    (permission) =>
      /^(?:fs|shell|http):/.test(permission)
      || (/^(?:updater|process):/.test(permission) && !allowedUpdaterAndProcessPermissions.has(permission))
  )
) {
  console.error("Native selectors must not introduce broad filesystem, shell, network, or updater/process permissions.");
  process.exit(1);
}

if (!script.includes('renderMode = "immediate"') || script.includes('behavior: "smooth"')) {
  console.error("Scientific plots and navigation must render immediately without decorative motion.");
  process.exit(1);
}

if (tauriConfig.app?.windows?.some((window) => window.devtools !== false)) {
  console.error("Production desktop windows must not expose embedded browser developer tools.");
  process.exit(1);
}

if (!html.includes("not peer review") || !html.includes("does not upload data")) {
  console.error("The academic export must state its review and upload boundaries.");
  process.exit(1);
}

if (!html.includes("Mechanistic Reference Model Lab") || !html.includes("mechanistic-inspect-btn")) {
  console.error("frontend/index.html must expose the independent SBML reference-model workflow.");
  process.exit(1);
}

if (!html.includes("never calibrates patient parameters automatically")) {
  console.error("The mechanistic lab must state its no-automatic-calibration boundary.");
  process.exit(1);
}

if (!html.includes("Cross-scale Reference Labs") || !html.includes("binding-query-btn")) {
  console.error("frontend/index.html must expose the cross-scale COPASI, CellML, FMI, and BindingDB labs.");
  process.exit(1);
}

if (!html.includes("FMI does not sandbox native code or OS access")) {
  console.error("The FMI workflow must state its native-code execution boundary.");
  process.exit(1);
}

for (const view of ["overview", "run", "results", "reproducibility", "ai", "research", "evidence", "settings"]) {
  if (!html.includes(`data-view="${view}"`) || !html.includes(`data-view-panel="${view}"`)) {
    console.error(`Missing navigation or panel mapping for ${view}.`);
    process.exit(1);
  }
}

const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index);
if (duplicates.length) {
  console.error(`Duplicate HTML IDs: ${[...new Set(duplicates)].join(", ")}`);
  process.exit(1);
}

const referencedIds = new Set([
  ...[...script.matchAll(/\$\("([^"]+)"\)/g)].map((match) => match[1]),
  ...[...script.matchAll(/setText\("([^"]+)"/g)].map((match) => match[1])
]);
const missingIds = [...referencedIds].filter((id) => !ids.includes(id));
if (missingIds.length) {
  console.error(`JavaScript references missing HTML IDs: ${missingIds.join(", ")}`);
  process.exit(1);
}

// Every command the frontend invokes must be registered in the Rust
// invoke_handler. Without this check a renamed or newly added command fails
// only at runtime, in the packaged app, with an opaque error.
const rustMain = readFileSync(join(appRoot, "src-tauri/src/main.rs"), "utf8");
const handlerBlock = rustMain.match(/generate_handler!\[([^\]]*)\]/s);
if (!handlerBlock) {
  console.error("Could not locate generate_handler! in src-tauri/src/main.rs.");
  process.exit(1);
}
const registeredCommands = new Set(
  handlerBlock[1]
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean)
);
const invokedCommands = new Set(
  [...script.matchAll(/\bcall\(\s*"([a-z0-9_]+)"/g)].map((match) => match[1])
);
const unregistered = [...invokedCommands].filter((name) => !registeredCommands.has(name));
if (unregistered.length) {
  console.error(`Frontend invokes unregistered Tauri commands: ${unregistered.join(", ")}`);
  process.exit(1);
}

const config = JSON.parse(readFileSync(join(appRoot, "src-tauri/tauri.conf.json"), "utf8"));
for (const icon of config.bundle?.icon || []) {
  if (!existsSync(join(appRoot, "src-tauri", icon))) {
    console.error(`Configured bundle icon does not exist: ${icon}`);
    process.exit(1);
  }
}

console.log("IINTS Tauri static scaffold OK");
