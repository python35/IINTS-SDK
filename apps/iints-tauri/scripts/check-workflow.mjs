// Exercise real event handlers against an in-memory native bridge. No model,
// desktop dialogs or external services are invoked by this regression check.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL, fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(root, "frontend/index.html"), "utf8");
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
function element(tag = "div") {
  return {
    tagName: tag, children: [], attributes: {}, dataset: {}, style: {}, listeners: {},
    value: "", textContent: "", checked: false, hidden: false, disabled: false,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    appendChild(child) { this.children.push(child); return child; },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    setAttribute(key, value) { this.attributes[key] = String(value); },
    getAttribute(key) { return this.attributes[key]; },
    removeAttribute(key) { delete this.attributes[key]; },
    addEventListener(key, fn) { (this.listeners[key] ||= []).push(fn); },
    removeEventListener() {}, focus() {}, scrollIntoView() {},
    querySelector: () => null, querySelectorAll: () => [],
    getBoundingClientRect: () => ({ width: 700, height: 260 }),
  };
}
const elements = new Map([...ids].map((id) => [id, element()]));
const get = (id) => { assert.ok(elements.has(id), `Missing element ${id}`); return elements.get(id); };
globalThis.document = {
  getElementById: get, createElement: element, createElementNS: (_, tag) => element(tag),
  createTextNode: (text) => ({ textContent: text }),
  querySelector: () => null, querySelectorAll: () => [], addEventListener() {},
  documentElement: element(), body: element(),
};
const calls = [];
const bundle = {
  status: "ready", workflow_title: "Meal experiment", preset_name: "quickstart_meal", seed: 42,
  output_dir: "/fixture/run", results_csv: "/fixture/run/results.csv", row_count: 3,
  report_pdf: "/fixture/run/report.pdf", steps: [{ key: "mdmp", title: "Data check", status: "ready" }],
  artifacts: [{ label: "PDF report", path: "/fixture/run/report.pdf" }],
  mdmp: { certificate_path: "/fixture/run/certificate.json", grade: "research", row_count: 3 },
  academic_bundle: { artifact_count: 5, readiness_status: "review", ro_crate_metadata: "/fixture/run/ro-crate-metadata.json" },
};
const preview = (csv) => ({ csv_path: csv, row_count: 3,
  rows: [["0", "100"], ["5", "125"], ["10", "110"]],
  columns: ["time_minutes", "glucose_actual_mgdl"], metrics: { "Mean glucose": "112 mg/dL" },
  bundle: csv === bundle.results_csv ? bundle : null,
});
let pendingPreview = null;
async function invoke(command, args) {
  calls.push({ command, args });
  if (command === "desktop_status") return { sdk_version: "fixture", bridge_api_version: 4, features: ["automatic_run_bundle"] };
  if (command === "run_history") return { history: [] };
  if (command === "list_workflows") return { workflows: [{ key: "booth-demo", title: "Meal", description: "A test", expected_output: "Report" }] };
  if (command === "start_workflow_job") return { job_id: "test-job" };
  if (command === "workflow_job_status") return { status: "completed", phase: "complete", result: { ...bundle, bundle } };
  if (command === "preview_results") {
    if (args.csv === "/fixture/slow.csv") return await new Promise((resolve) => { pendingPreview = resolve; });
    return preview(args.csv);
  }
  if (command === "compartment_timeline") return { available: false, reason: "No fixture compartments" };
  return {};
}
globalThis.window = { __TAURI__: { core: { invoke } }, addEventListener() {},
  scrollTo() {}, setTimeout, matchMedia: () => ({ matches: false, addEventListener() {} }), location: {} };
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
globalThis.fetch = async () => ({ ok: false, status: 404 });
globalThis.requestAnimationFrame = () => 1;
Object.defineProperty(globalThis, "navigator", { value: { userAgent: "node", clipboard: { writeText: async () => {} } }, configurable: true });
await import(pathToFileURL(join(root, "frontend/main.js")));
async function fire(id, event = "click") {
  for (const fn of get(id).listeners[event] || []) await fn({ target: get(id) });
}
await fire("refresh-btn");
await fire("run-btn");
const start = calls.find((item) => item.command === "start_workflow_job");
assert.equal(start.args.autoPrepare, true);
assert.equal(start.args.aiReview, false);
assert.equal(get("result-handover").hidden, false);
assert.equal(get("result-run-title").textContent, "Meal experiment");
assert.equal(get("open-certificate-btn").disabled, false);
assert.equal(get("academic-open-metadata-btn").disabled, false);
assert.ok(get("result-artifacts").children.some((item) => item.textContent === "PDF report"));
assert.ok(!calls.some((item) => ["certify_mdmp", "export_academic_bundle"].includes(item.command)), "Preparation was run twice");

// Editing the file clears the old experiment and invalidates pending responses.
get("csv-path").value = "/fixture/slow.csv";
await fire("csv-path", "input");
assert.equal(get("open-report-btn").disabled, true);
const slow = fire("preview-btn");
get("csv-path").value = "/fixture/new.csv";
await fire("csv-path", "input");
await fire("preview-btn");
pendingPreview(preview("/fixture/slow.csv"));
await slow;
assert.ok(get("result-run-meta").textContent.includes("/fixture/new.csv"));
assert.equal(get("open-certificate-btn").disabled, true);

// Reopening the earlier run restores its certificate/package without rerunning it.
get("csv-path").value = bundle.results_csv;
await fire("csv-path", "input");
await fire("preview-btn");
assert.equal(get("open-certificate-btn").disabled, false);
assert.equal(get("academic-open-metadata-btn").disabled, false);
console.log("IINTS run preparation, result handover and stale-response checks passed");
