// Headless check of the compartment view in frontend/main.js.
//
// The desktop frontend cannot be unit tested in a browser here, so this script
// loads the real module against a minimal DOM and a stubbed bridge, drives the
// real code path (preview button -> preview_results -> compartment_timeline),
// and asserts the invariants that matter for an honest diagram:
//
//   * every recorded flux is either drawn or named as not drawn -- an arrow
//     may never disappear silently;
//   * no drawn arrow has zero length, which is how a boundary flux looks when
//     both its endpoints collapse onto the same box;
//   * the caption keeps the caveats that make the drawing readable: the per
//     variable normalisation, and that arrows are instantaneous rates.
//
// The schema-to-equations coupling is checked on the Python side in
// tests/test_compartment_export.py; the loader in tests/test_compartment_timeline.py.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(appRoot, "frontend/index.html"), "utf8");
const declaredIds = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));

const failures = [];
function check(condition, message) {
  if (!condition) failures.push(message);
}

// A payload shaped like the bridge output: two contents, one internal flux,
// one boundary flux, and one effect state that the default view hides.
const timeline = {
  available: true,
  reason: "",
  schema: {
    model_key: "fixture",
    model_label: "Fixture model (3 states)",
    compartments: [
      { key: "Q_gut", symbol: "Q_gut", label: "Gut carbohydrate", unit: "mg", state_index: 0, kind: "pool", site: "gut", provenance: "canonical", description: "" },
      { key: "G", symbol: "G", label: "Plasma glucose", unit: "mg/dL", state_index: 1, kind: "concentration", site: "plasma", provenance: "canonical", description: "" },
      { key: "X", symbol: "X", label: "Insulin action", unit: "1/min", state_index: 2, kind: "effect", site: "signal", provenance: "canonical", description: "" },
    ],
    fluxes: [
      { key: "appearance", source: "Q_gut", target: "G", label: "Glucose appearance", unit: "mg/dL/min", rate_expression: "k * Q_gut", parameters: ["k"], provenance: "canonical", description: "", recorded: true },
      { key: "uptake", source: "G", target: null, label: "Glucose uptake", unit: "mg/dL/min", rate_expression: "(p1 + X) * G", parameters: ["p1"], provenance: "canonical", description: "", recorded: true },
      { key: "action", source: null, target: "X", label: "Insulin action", unit: "1/min^2", rate_expression: "p2 * (I - Ib)", parameters: ["p2"], provenance: "canonical", description: "", recorded: true },
      { key: "ingestion", source: null, target: "Q_gut", label: "Carbohydrate ingestion", unit: "mg", rate_expression: "discrete", parameters: [], provenance: "canonical", description: "", recorded: false },
    ],
  },
  times: [0, 5, 10],
  compartments: { Q_gut: [1000, 800, 600], G: [120, 140, 130], X: [0.001, 0.002, 0.0015] },
  // uptake goes negative at the last sample: the arrow must reverse, not vanish.
  fluxes: { appearance: [2.0, 1.6, 1.2], uptake: [1.1, 0.4, -0.3], action: [0.0001, 0.0002, 0.0] },
  flux_extremes: { appearance: [1.2, 2.0], uptake: [-0.3, 1.1], action: [0.0, 0.0002] },
  stride: 1,
  step_count: 3,
};

function makeElement(tag = "div", id = "") {
  return {
    tagName: tag,
    id,
    children: [],
    attributes: {},
    style: {},
    dataset: {},
    listeners: {},
    value: "",
    max: "",
    min: "",
    checked: false,
    hidden: false,
    textContent: "",
    innerHTML: "",
    className: "",
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    setAttribute(key, value) {
      this.attributes[key] = String(value);
    },
    getAttribute(key) {
      return this.attributes[key];
    },
    removeAttribute(key) {
      delete this.attributes[key];
    },
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    append(...kids) {
      this.children.push(...kids);
    },
    replaceChildren(...kids) {
      this.children = kids;
    },
    remove() {},
    addEventListener(type, fn) {
      (this.listeners[type] ||= []).push(fn);
    },
    removeEventListener() {},
    querySelector: () => null,
    querySelectorAll: () => [],
    closest: () => null,
    focus() {},
    click() {},
    scrollIntoView() {},
    getBoundingClientRect: () => ({ width: 760, height: 420, top: 0, left: 0 }),
    insertAdjacentHTML() {},
  };
}

const registry = new Map();
const unknownIds = new Set();
function byId(id) {
  if (!declaredIds.has(id)) unknownIds.add(id);
  if (!registry.has(id)) registry.set(id, makeElement("div", id));
  return registry.get(id);
}

globalThis.window = {
  __TAURI__: {
    core: {
      invoke: async (command) => {
        if (command === "preview_results") {
          return {
            csv_path: "/fixture/results.csv",
            row_count: timeline.step_count,
            columns: ["time_minutes", "glucose_actual_mgdl"],
            rows: [["0", "120"]],
            metrics: {},
            graph_path: null,
          };
        }
        if (command === "compartment_timeline") return timeline;
        return {};
      },
    },
  },
  addEventListener() {},
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  location: { href: "" },
};
globalThis.localStorage = window.localStorage;
globalThis.navigator = { clipboard: { writeText: async () => {} }, userAgent: "node" };
globalThis.document = {
  getElementById: byId,
  createElement: (tag) => makeElement(tag),
  createElementNS: (_ns, tag) => makeElement(tag),
  createTextNode: (text) => ({ textContent: text }),
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener() {},
  body: makeElement("body"),
  documentElement: makeElement("html"),
};
globalThis.requestAnimationFrame = (fn) => fn();
// The module fetches bundled evidence files with document-relative URLs on
// load. Node has no document base, so answer with a clean 404 instead of
// letting an unresolvable URL print a stack trace over the check output.
globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({}), text: async () => "" });

await import(`${join(appRoot, "frontend/main.js")}`);

byId("csv-path").value = "/fixture/results.csv";
const previewClicks = registry.get("preview-btn")?.listeners.click || [];
check(previewClicks.length > 0, "No click handler is bound to preview-btn.");
await Promise.all(previewClicks.map((fn) => fn()));

const diagram = byId("compartment-diagram");
const caption = () => String(byId("compartment-caption").textContent);
const arrowsOf = () => diagram.children.filter((node) => node.tagName === "line");
const boxesOf = () => diagram.children.filter((node) => node.tagName === "g");

check(!byId("compartment-viewer").hidden, "Viewer stayed hidden for an available timeline.");
check(unknownIds.size === 0, `JavaScript touched HTML IDs that do not exist: ${[...unknownIds].join(", ")}`);
check(byId("compartment-time").max === "2", `Slider max should span the samples, got ${byId("compartment-time").max}.`);

// Default view: contents only, effect state hidden.
check(boxesOf().length === 2, `Default view should draw the 2 contents, got ${boxesOf().length}.`);
check(
  /1 signal or effect states hidden/.test(caption()),
  "Caption does not disclose the hidden effect state."
);
check(
  /1 fluxes that touch them/.test(caption()),
  "Caption does not disclose the flux dropped with the hidden state."
);
check(
  /Not drawn as a rate: Carbohydrate ingestion/.test(caption()),
  "Caption does not name the flux that carries no numeric rate."
);
check(/normalised per variable/.test(caption()), "Caption lost the per-variable normalisation caveat.");
check(/instantaneous rates/.test(caption()), "Caption lost the instantaneous-rate caveat.");

// With every state shown, all recorded fluxes must be drawn.
byId("compartment-show-signals").checked = true;
(registry.get("compartment-show-signals").listeners.change || []).forEach((fn) => fn());
const recorded = timeline.schema.fluxes.filter((flux) => timeline.fluxes[flux.key]).length;
check(boxesOf().length === 3, `All states shown should draw 3 boxes, got ${boxesOf().length}.`);
check(
  arrowsOf().length === recorded,
  `All states shown should draw every recorded flux (${recorded}), got ${arrowsOf().length}.`
);

function degenerateArrows() {
  return arrowsOf().filter((line) => {
    const dx = Number(line.attributes.x2) - Number(line.attributes.x1);
    const dy = Number(line.attributes.y2) - Number(line.attributes.y1);
    return !Number.isFinite(dx) || !Number.isFinite(dy) || Math.hypot(dx, dy) < 1;
  });
}
check(
  degenerateArrows().length === 0,
  `${degenerateArrows().length} arrows have zero length; a boundary flux collapsed onto its own box.`
);

// Two distinct fluxes must not be drawn as the same line: plasma glucose
// carries several boundary terms, and stacking them reads as one flux.
const geometries = arrowsOf().map((line) =>
  ["x1", "y1", "x2", "y2"].map((key) => Number(line.attributes[key]).toFixed(1)).join(",")
);
const duplicated = geometries.filter((g, i) => geometries.indexOf(g) !== i);
check(
  duplicated.length === 0,
  `${duplicated.length} arrows share their geometry with another flux; boundary stubs must occupy separate slots.`
);

// Scrub to the sample where uptake is negative: the arrow reverses.
const slider = byId("compartment-time");
const arrowOrientation = () => {
  const line = arrowsOf().find((node) => String(node.children[0]?.textContent || "").startsWith("Glucose uptake"));
  return line ? `${line.attributes.y1}->${line.attributes.y2}` : null;
};
const forward = arrowOrientation();
slider.value = "2";
(registry.get("compartment-time").listeners.input || []).forEach((fn) => fn());
const reversed = arrowOrientation();
check(forward !== null && reversed !== null, "Could not locate the uptake arrow to check its direction.");
check(
  forward !== reversed,
  "A flux that turned negative kept its original direction; the arrow must reverse."
);
check(degenerateArrows().length === 0, "An arrow collapsed to zero length after reversing.");
check(
  byId("compartment-time-label").textContent === "10 min",
  `Time label should follow the slider, got ${byId("compartment-time-label").textContent}.`
);

// An unavailable timeline must state a reason instead of showing an empty view.
const unavailable = { ...timeline, available: false, reason: "run predates the compartment export" };
Object.assign(timeline, unavailable);
await Promise.all(previewClicks.map((fn) => fn()));
check(byId("compartment-viewer").hidden, "Viewer stayed visible for an unavailable timeline.");
check(
  /predates/.test(String(byId("compartment-status").textContent)),
  "Unavailable timeline did not surface its reason."
);

if (failures.length) {
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log("IINTS compartment view OK");
