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
  "scripts/build-brand-icons.py",
  "src-tauri/tauri.conf.json",
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
const script = readFileSync(join(appRoot, "frontend/main.js"), "utf8");
const styles = readFileSync(join(appRoot, "frontend/styles.css"), "utf8");
const tauriConfig = JSON.parse(readFileSync(join(appRoot, "src-tauri/tauri.conf.json"), "utf8"));
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

const config = JSON.parse(readFileSync(join(appRoot, "src-tauri/tauri.conf.json"), "utf8"));
for (const icon of config.bundle?.icon || []) {
  if (!existsSync(join(appRoot, "src-tauri", icon))) {
    console.error(`Configured bundle icon does not exist: ${icon}`);
    process.exit(1);
  }
}

console.log("IINTS Tauri static scaffold OK");
