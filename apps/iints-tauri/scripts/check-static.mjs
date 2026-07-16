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

console.log("IINTS Tauri static scaffold OK");
