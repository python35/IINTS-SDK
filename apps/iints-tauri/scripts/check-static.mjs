import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = dirname(dirname(fileURLToPath(import.meta.url)));

const required = [
  "frontend/index.html",
  "frontend/styles.css",
  "frontend/main.js",
  "src-tauri/tauri.conf.json",
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
if (!html.includes("Research only") || !html.includes("Not a medical device")) {
  console.error("frontend/index.html must contain the research-only disclaimer.");
  process.exit(1);
}

if (!html.includes("Reproducibility package") || !html.includes("academic-export-btn")) {
  console.error("frontend/index.html must expose the academic reproducibility package.");
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

console.log("IINTS Tauri static scaffold OK");
