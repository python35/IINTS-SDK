// Headless check of the 3D Virtual Patient Digital Twin's pure data layer.
//
// Only frontend/digital-twin-data.js is imported here -- never
// digital-twin-scene.js or the vendored three.js file. That module needs a
// real WebGL context (a `<canvas>.getContext("webgl2")`), which a stubbed
// Node DOM (as check-compartment-view.mjs uses for main.js) cannot provide;
// mesh rendering, glow/transparency, particle motion, raycast picking, and
// camera feel are therefore explicitly NOT covered here and must be verified
// manually via `npm run dev`, the same way the existing hand-rolled
// "molecule viewer" canvas has no headless visual test either.
//
// What *is* checked here is exactly the part that can regress silently
// without ever touching a GPU: interpolation math, the flux-direction and
// particle-density formulas (which must stay in parity with the 2D SVG
// diagram's own formulas in main.js, see the parity check below), the hypo
// threshold, and the organ layout's shape.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  HYPO_THRESHOLD_MGDL,
  LIVER_CARD_CAPTION,
  ORGAN_LAYOUT,
  computeParticleParams,
  interpolateSeries,
  lookupEquation,
  resolveFluxDirection,
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

// --- resolveFluxDirection --------------------------------------------------

check(resolveFluxDirection(1.5) === 1, "A positive flux should resolve to the schema's declared direction (+1).");
check(resolveFluxDirection(0) === 1, "A zero flux should resolve to the declared direction, not reversed.");
check(resolveFluxDirection(-0.01) === -1, "A negative flux should resolve to the reversed direction (-1), matching the 2D diagram's arrow-flip rule.");

// --- computeParticleParams: parity with the 2D diagram's arrow-strength formula ---
// main.js's renderCompartmentDiagram computes, per flux:
//   scale = max(|extreme[0]|, |extreme[1]|); strength = scale > 0 ? |rate| / scale : 0
// This is deliberately re-derived here (not imported from main.js, which is
// not a module main.js's non-DOM parts can be cleanly imported from) and
// checked for exact numeric parity, so the two views can never silently
// disagree about how "strong" a flux looks at a given moment.
function svgArrowStrength(rate, extreme) {
  const scale = Math.max(Math.abs(extreme[0]), Math.abs(extreme[1]));
  return scale > 0 ? Math.abs(rate) / scale : 0;
}

for (const [rate, extreme] of [
  [1.2, [-0.3, 1.1]],
  [-0.3, [-0.3, 1.1]],
  [0, [-0.3, 1.1]],
  [0.55, [0, 2.0]],
  [3, [0, 0]], // degenerate: no recorded range yet
]) {
  const expected = svgArrowStrength(rate, extreme);
  const { strength } = computeParticleParams(rate, extreme);
  check(
    Math.abs(strength - expected) < 1e-9,
    `computeParticleParams(${rate}, [${extreme}]).strength should match the 2D diagram's formula (expected ${expected}, got ${strength}).`
  );
}

const weak = computeParticleParams(0.1, [0, 1]);
const strong = computeParticleParams(0.9, [0, 1]);
check(strong.count > weak.count, "A stronger flux should produce a denser particle stream than a weaker one.");
check(strong.speed > weak.speed, "A stronger flux should produce a faster particle stream than a weaker one.");
check(weak.count >= 8 && strong.count <= 120, "Particle count should stay within the documented [minCount, maxCount] bounds.");

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
  check(
    Array.isArray(organ.position) && organ.position.length === 3 && organ.position.every(Number.isFinite),
    `Organ "${organ.id}" must declare a finite 3D position.`
  );
  check(Number.isInteger(organ.color), `Organ "${organ.id}" must declare a numeric hex color.`);
}

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

// --- three.js vendoring sanity (cheap, no WebGL needed) ---------------------

const vendoredThree = readFileSync(join(appRoot, "frontend/vendor/three/three.module.js"), "utf8");
check(
  !/^\s*import\s+.*from\s+["'](?!\.)/m.test(vendoredThree),
  "The vendored three.js build must not contain a bare-specifier import (this app's CSP has no import map)."
);

if (failures.length) {
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log("IINTS digital twin data layer OK");
