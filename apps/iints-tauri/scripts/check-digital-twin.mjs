// Headless check of the illustrated torso diagram in frontend/main.js.
//
// Two layers, same split as check-compartment-view.mjs uses for the SVG
// "Diagram" tab: pure data-layer tests against digital-twin-data.js
// (interpolation, normalization, hypo threshold, organ layout shape), then a
// DOM-driven integration check against the real main.js loaded in a stubbed
// Node DOM -- there is no WebGL/canvas involved any more, so unlike the
// three.js scene this replaced, the actual rendering logic is fully
// headless-testable here, not just its pure math.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  HYPO_THRESHOLD_MGDL,
  HYPO_VASCULAR_COLOR,
  LIVER_CARD_CAPTION,
  ORGAN_LAYOUT,
  averageFillLevel,
  compartmentRange,
  interpolateSeries,
  lookupEquation,
  normalizeFillLevel,
  resolveHypoState,
} from "../frontend/digital-twin-data.js";

const appRoot = join(dirname(fileURLToPath(import.meta.url)), "..");

const failures = [];
function check(condition, message) {
  if (!condition) failures.push(message);
}

// --- interpolateSeries ----------------------------------------------------

const times = [0, 5, 10, 20];
const values = [100, 120, 80, 200];

check(interpolateSeries(times, values, 0) === 100, "interpolateSeries should return the exact value at the first sample.");
check(interpolateSeries(times, values, 5) === 120, "interpolateSeries should return the exact value at an interior sample.");
check(interpolateSeries(times, values, 20) === 200, "interpolateSeries should return the exact value at the last sample.");
check(
  interpolateSeries(times, values, 7.5) === 100,
  `interpolateSeries should interpolate linearly between 120 and 80 at t=7.5 (expected 100, got ${interpolateSeries(times, values, 7.5)}).`
);
check(
  interpolateSeries(times, values, -50) === 100,
  "interpolateSeries should clamp to the first value below the first sample time."
);
check(
  interpolateSeries(times, values, 1440) === 200,
  "interpolateSeries should clamp to the last value above the last sample time."
);
check(Number.isNaN(interpolateSeries([], [], 5)), "interpolateSeries should return NaN for an empty series rather than throw.");

// --- compartmentRange / normalizeFillLevel / averageFillLevel --------------

check(
  JSON.stringify(compartmentRange([3, 1, NaN, 5])) === JSON.stringify({ low: 1, high: 5 }),
  "compartmentRange should ignore non-finite values and return the min/max of the rest."
);
check(compartmentRange([]) === null, "compartmentRange should return null for an empty series rather than throw.");
check(normalizeFillLevel(5, { low: 0, high: 10 }) === 0.5, "normalizeFillLevel should map the midpoint of a range to 0.5.");
check(normalizeFillLevel(-5, { low: 0, high: 10 }) === 0, "normalizeFillLevel should clamp below the range to 0.");
check(normalizeFillLevel(15, { low: 0, high: 10 }) === 1, "normalizeFillLevel should clamp above the range to 1.");
check(normalizeFillLevel(5, { low: 10, high: 10 }) === 0.5, "normalizeFillLevel should fall back to 0.5 for a degenerate (zero-span) range.");
check(Math.abs(averageFillLevel([0.2, 0.4, 0.6]) - 0.4) < 1e-9, "averageFillLevel should average its inputs.");
check(averageFillLevel([]) === 0.5, "averageFillLevel should fall back to 0.5 for an empty input.");
check(averageFillLevel([NaN, 0.8]) === 0.8, "averageFillLevel should ignore non-finite inputs rather than propagate NaN.");

// --- resolveHypoState -------------------------------------------------------

check(HYPO_THRESHOLD_MGDL === 70, "The hypoglycemia threshold must be 70 mg/dL.");
check(resolveHypoState(69.9) === true, "69.9 mg/dL must resolve as hypoglycemic.");
check(resolveHypoState(70.0) === false, "70.0 mg/dL must resolve as NOT hypoglycemic (threshold is exclusive).");
check(resolveHypoState(70.1) === false, "70.1 mg/dL must resolve as not hypoglycemic.");
check(resolveHypoState(NaN) === false, "A missing/NaN glucose value must not be treated as hypoglycemic.");

// --- ORGAN_LAYOUT ------------------------------------------------------------

const expectedIds = ["gut", "subcutaneous", "plasma", "periphery", "liver"];
const actualIds = ORGAN_LAYOUT.map((organ) => organ.id);
check(
  expectedIds.every((id) => actualIds.includes(id)) && actualIds.length === expectedIds.length,
  `ORGAN_LAYOUT must declare exactly the 5 organs ${JSON.stringify(expectedIds)}, got ${JSON.stringify(actualIds)}.`
);

const liver = ORGAN_LAYOUT.find((organ) => organ.id === "liver");
check(liver?.kind === "flux-proxy", "The liver organ must be a flux-proxy, not a compartment-group -- it has no ODE state of its own.");
check(
  liver?.boundFluxKey === "endogenous_production",
  `The liver organ must bind to the endogenous_production flux (Hovorka's EGP term), got ${liver?.boundFluxKey}.`
);

for (const organ of ORGAN_LAYOUT) {
  if (organ.id === "liver") continue;
  check(organ.kind === "compartment-group", `Organ "${organ.id}" should be a compartment-group (it is backed by a real ODE site).`);
  check(typeof organ.site === "string" && organ.site.length > 0, `Organ "${organ.id}" must declare which schema site it renders.`);
}

for (const organ of ORGAN_LAYOUT) {
  check(typeof organ.elementId === "string" && organ.elementId.length > 0, `Organ "${organ.id}" must declare the SVG element id it drives.`);
  check(/^#[0-9a-f]{6}$/i.test(organ.color), `Organ "${organ.id}" must declare a CSS hex color string, got ${organ.color}.`);
}
check(/^#[0-9a-f]{6}$/i.test(HYPO_VASCULAR_COLOR), "HYPO_VASCULAR_COLOR must be a CSS hex color string.");

// --- lookupEquation + liver card copy ---------------------------------------

const fixtureSchema = {
  fluxes: [
    {
      key: "endogenous_production",
      source: null,
      target: "Q1",
      label: "Endogenous glucose production",
      unit: "mg/min",
      rate_expression: "EGP_0 * max(0, 1 - x3 + x_gluc)",
    },
  ],
};
check(
  lookupEquation(fixtureSchema, "endogenous_production") === "EGP_0 * max(0, 1 - x3 + x_gluc)",
  "lookupEquation must return the flux's rate_expression verbatim."
);
check(lookupEquation(fixtureSchema, "not_a_real_flux") === null, "lookupEquation must return null for an unknown flux key, not throw.");
check(
  /not a stored compartment/i.test(LIVER_CARD_CAPTION) && /source flux/i.test(LIVER_CARD_CAPTION),
  "The liver's inspection-card caption must explicitly state it is a source flux, not a storage compartment."
);

// --- DOM integration: drive the real main.js against a fixture timeline ----

const html = readFileSync(join(appRoot, "frontend/index.html"), "utf8");
const declaredIds = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
for (const organ of ORGAN_LAYOUT) {
  check(declaredIds.has(organ.elementId), `index.html must declare an element with id="${organ.elementId}" for organ "${organ.id}".`);
}

// A payload shaped like the bridge output, with two sub-compartments at the
// gut site (mirroring Hovorka's D1/D2) so averageFillLevel actually has more
// than one input to average, and a hypoglycemic sample at t=10 so the plasma
// organ's hypo-color path gets exercised.
const timeline = {
  available: true,
  reason: "",
  schema: {
    model_key: "fixture",
    model_label: "Fixture model",
    compartments: [
      { key: "D1", symbol: "D1", label: "Gut carbs 1", unit: "mg", site: "gut", kind: "pool" },
      { key: "D2", symbol: "D2", label: "Gut carbs 2", unit: "mg", site: "gut", kind: "pool" },
      { key: "S1", symbol: "S1", label: "SC insulin", unit: "mU", site: "subcutaneous", kind: "pool" },
      { key: "Q1", symbol: "Q1", label: "Plasma glucose", unit: "mg", site: "plasma", kind: "pool" },
      { key: "Q2", symbol: "Q2", label: "Peripheral glucose", unit: "mg", site: "periphery", kind: "pool" },
    ],
    fluxes: [
      { key: "endogenous_production", source: null, target: "Q1", label: "EGP", unit: "mg/min", rate_expression: "EGP_0 * max(0, 1 - x3)" },
      { key: "glucose_appearance", source: "D2", target: "Q1", label: "Glucose appearance", unit: "mg/min", rate_expression: "D2 / tmax" },
      { key: "glucose_to_periphery", source: "Q1", target: "Q2", label: "Peripheral uptake", unit: "mg/min", rate_expression: "x1 * Q1" },
    ],
  },
  times: [0, 5, 10],
  compartments: {
    D1: [1000, 600, 200],
    D2: [500, 800, 950],
    S1: [10, 8, 6],
    Q1: [180, 150, 60],
    Q2: [90, 95, 100],
  },
  fluxes: {
    endogenous_production: [0.1, 0.2, 0.3],
    glucose_appearance: [2.0, 2.4, 2.8],
    glucose_to_periphery: [0.5, 0.6, 0.7],
  },
  flux_extremes: {
    endogenous_production: [0.1, 0.3],
    glucose_appearance: [2.0, 2.8],
    glucose_to_periphery: [0.5, 0.7],
  },
  plasma_glucose_mgdl: [180, 150, 60],
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
function byId(id) {
  if (!registry.has(id)) registry.set(id, makeElement("div", id));
  return registry.get(id);
}

globalThis.window = {
  __TAURI__: {
    core: {
      invoke: async (command) => {
        if (command === "preview_results") {
          return { csv_path: "/fixture/results.csv", row_count: 3, columns: ["time_minutes"], rows: [["0"]], metrics: {}, graph_path: null };
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
Object.defineProperty(globalThis, "navigator", {
  value: { clipboard: { writeText: async () => {} }, userAgent: "node" },
  configurable: true,
  writable: true,
});
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
globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({}), text: async () => "" });

await import(pathToFileURL(join(appRoot, "frontend/main.js")));

byId("csv-path").value = "/fixture/results.csv";
const previewClicks = registry.get("preview-btn")?.listeners.click || [];
check(previewClicks.length > 0, "No click handler is bound to preview-btn.");
await Promise.all(previewClicks.map((fn) => fn()));

check(
  byId("digital-twin-time").max === "10",
  `Illustrated-view time bounds should end at the run's last sample (10), got ${byId("digital-twin-time").max}.`
);
check(
  /FIXTURE/.test(String(byId("digital-twin-model-label").textContent)),
  "Illustrated-view model tag should be derived from the loaded schema instead of naming a fixed model."
);
check(
  /Fixture model/.test(String(byId("digital-twin-description").textContent)),
  "Illustrated-view description should identify the loaded model."
);

// renderDigitalTwinDiagram() doesn't gate on which compartment-view tab is
// active, so driving the time input alone is enough to exercise it here --
// same as scrubbing while the "Illustrated view" tab happens to be open.
const timeInput = byId("digital-twin-time");
check(!!(timeInput.listeners.input && timeInput.listeners.input.length), "No input handler is bound to digital-twin-time.");
timeInput.value = "5";
timeInput.listeners.input.forEach((fn) => fn({ target: timeInput }));

for (const organ of ORGAN_LAYOUT) {
  const el = byId(organ.elementId);
  check(typeof el.style.stroke === "string" && el.style.stroke.length > 0, `Organ "${organ.id}" should have its stroke set after a render.`);
  check(
    Number(el.style.opacity) >= 0.55 && Number(el.style.opacity) <= 1,
    `Organ "${organ.id}" opacity should stay within the documented [0.55, 1] band, got ${el.style.opacity}.`
  );
  check(
    Number(el.style.strokeWidth) >= 1.4 && Number(el.style.strokeWidth) <= 2.6,
    `Organ "${organ.id}" stroke-width should stay within the documented [1.4, 2.6] band, got ${el.style.strokeWidth}.`
  );
}

const plasmaAtT5 = byId("twin-organ-plasma").style.stroke;
check(plasmaAtT5.toLowerCase() !== HYPO_VASCULAR_COLOR.toLowerCase(), "Plasma should not show the hypo color at t=5 (180 mg/dL, not hypoglycemic).");

timeInput.value = "10";
timeInput.listeners.input.forEach((fn) => fn({ target: timeInput }));
const plasmaAtT10 = byId("twin-organ-plasma").style.stroke;
check(
  plasmaAtT10.toLowerCase() === HYPO_VASCULAR_COLOR.toLowerCase(),
  `Plasma should switch to the hypo color at t=10 (60 mg/dL, hypoglycemic), got ${plasmaAtT10}.`
);
const liverAtT10 = byId("twin-organ-liver").style.stroke;
check(liverAtT10.toLowerCase() !== HYPO_VASCULAR_COLOR.toLowerCase(), "The liver's own steady red must stay visually distinct from the hypo-alarm color.");

// --- HUD badges: real flux/compartment keys, schema-driven units ----------
// t=10 uses glucose_appearance=2.8, glucose_to_periphery=0.7, endogenous_production=0.3,
// S1=6 -- values chosen to be distinguishable from each other and from zero,
// so a badge stuck on a stale/placeholder value would show up as a mismatch.
check(byId("twin-val-liver").textContent === "0.30 mg/min", `Liver HUD badge should read the endogenous_production flux, got "${byId("twin-val-liver").textContent}".`);
check(
  byId("twin-val-gut").textContent === "2.80 mg/min",
  `Gut HUD badge should read the glucose_appearance flux (not a nonexistent "appearance" key), got "${byId("twin-val-gut").textContent}".`
);
check(
  byId("twin-val-periphery").textContent === "0.70 mg/min",
  `Periphery HUD badge should read the glucose_to_periphery flux (not a nonexistent "uptake" key), got "${byId("twin-val-periphery").textContent}".`
);
check(
  byId("twin-val-subcutaneous").textContent === "6.0 mU",
  `Subcutaneous HUD badge should read S1 with its real schema unit (mU), got "${byId("twin-val-subcutaneous").textContent}".`
);
check(
  byId("twin-val-plasma").textContent === "60.0 mg/dL",
  `Plasma HUD badge should track the interpolated glucose value, got "${byId("twin-val-plasma").textContent}".`
);

// --- Status pill tracks the same hypo/hyper thresholds as the organ color -
check(
  byId("twin-status-text").textContent === "ALERT: HYPOGLYCEMIA (<70)",
  `Status pill should read the hypoglycemia alert at 60 mg/dL, got "${byId("twin-status-text").textContent}".`
);
timeInput.value = "0"; // 180 mg/dL: above the 70-180 band's ceiling is not exercised by this fixture, but 180 itself is not > 180
timeInput.listeners.input.forEach((fn) => fn({ target: timeInput }));
check(
  byId("twin-status-text").textContent === "EUGLYCEMIA (STABLE)",
  `Status pill should read stable at 180 mg/dL (not strictly above the high threshold), got "${byId("twin-status-text").textContent}".`
);
timeInput.value = "10";
timeInput.listeners.input.forEach((fn) => fn({ target: timeInput }));

// Clicking an organ should populate the inspection card.
const gutClicks = byId("twin-organ-gut").listeners.click || [];
check(gutClicks.length > 0, "No click handler is bound to the gut organ shape.");
gutClicks.forEach((fn) => fn());
const card = byId("digital-twin-card");
check(card.hidden === false, "Clicking an organ should reveal the inspection card.");
check(/Gut carbs 1/.test(card.innerHTML) && /Gut carbs 2/.test(card.innerHTML), "Clicking the gut organ should list both of its sub-compartments in the card.");

const liverClicks = byId("twin-organ-liver").listeners.click || [];
liverClicks.forEach((fn) => fn());
check(/EGP/.test(card.innerHTML), "Clicking the liver should show the endogenous_production flux's label.");
check(/EGP_0/.test(card.innerHTML), "Clicking the liver should show the flux's literal rate_expression.");
check(new RegExp(LIVER_CARD_CAPTION.slice(0, 30).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).test(card.innerHTML), "Clicking the liver should show its fixed caption.");

if (failures.length) {
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log("IINTS illustrated digital twin OK");
