// Pure data helpers for the 3D Virtual Patient Digital Twin viewer.
//
// Deliberately zero imports, no THREE, no DOM access: this file is safe to
// import from a headless Node script (scripts/check-digital-twin.mjs) the
// same way scripts/check-compartment-view.mjs already imports parts of
// main.js in a stubbed DOM. digital-twin-scene.js (the WebGL half) imports
// this module; this module never imports it back.
//
// House rule, same as the existing SVG compartment diagram
// (main.js's renderCompartmentDiagram): nothing about the physiology is
// re-implemented here. Every number these functions consume is a value the
// simulator already computed (compartmentTimeline.compartments/fluxes); this
// file only interpolates, normalizes, and looks values up for rendering.

// Hovorka v1 scope only (see the implementation plan): Bergman's `periphery`
// site has no dedicated pool compartment the way Hovorka's Q2 does, so a
// second, model-aware layout is a later iteration, not this one.
export const HYPO_THRESHOLD_MGDL = 70;

// Per-organ 3D placement/color/role. This is the 3D analogue of main.js's
// COMPARTMENT_SITE_ORDER/COMPARTMENT_SITE_LABELS -- a hardcoded, client-side
// layout table, not something the backend describes. `kind` distinguishes
// the four organs backed by a real ODE site ("compartment-group") from the
// liver, which has no compartment of its own and is fed purely by the live
// value of a boundary flux ("flux-proxy") -- see the plan's decision on how
// to represent hepatic glucose production (EGP) without a liver state.
export const ORGAN_LAYOUT = [
  {
    id: "gut",
    label: "Gut",
    kind: "compartment-group",
    site: "gut",
    position: [-3, 0.5, 0],
    color: 0xff9f43, // orange
  },
  {
    id: "subcutaneous",
    label: "Subcutaneous pump site",
    kind: "compartment-group",
    site: "subcutaneous",
    position: [-1.4, 1.6, 0.5],
    color: 0x8b5cf6, // purple/blue
  },
  {
    id: "plasma",
    label: "Vascular / plasma",
    kind: "compartment-group",
    site: "plasma",
    position: [0, 0.4, 0],
    color: 0x38bdf8, // cyan
  },
  {
    id: "periphery",
    label: "Peripheral muscle/fat",
    kind: "compartment-group",
    site: "periphery",
    position: [1.4, -1.2, 0.3],
    color: 0x22c55e, // green
  },
  {
    id: "liver",
    label: "Liver (EGP source)",
    kind: "flux-proxy",
    boundFluxKey: "endogenous_production",
    position: [0.6, 0.9, -0.4],
    color: 0xef4444, // red
  },
];

// The fixed, alarm-state red for the vascular organ during hypoglycemia,
// kept visually distinct from the liver's steady red (0xef4444 above) so the
// two don't read as the same signal when both are on screen at once.
export const HYPO_VASCULAR_COLOR = 0xdc2626;

/**
 * Min/max of a compartment's own series over the whole run, used to
 * normalize its fill level the same way the 2D SVG diagram normalizes tank
 * fill bars (main.js's compartmentRange) -- each compartment is scaled
 * against its own range, not a shared one, since a 5 mU insulin depot and a
 * 10,000 mg glucose pool are not comparable in absolute terms.
 *
 * @param {number[]} series
 * @returns {{low: number, high: number} | null} null if the series has no finite values
 */
export function compartmentRange(series) {
  let low = Infinity;
  let high = -Infinity;
  for (const value of series || []) {
    if (!Number.isFinite(value)) continue;
    if (value < low) low = value;
    if (value > high) high = value;
  }
  if (!Number.isFinite(low)) return null;
  return { low, high };
}

/**
 * Normalize a value into [0, 1] against a compartment's own run range, for
 * driving a mesh's fill scale. Falls back to a fixed mid-level when the
 * range is degenerate (constant series, or no data yet) rather than
 * dividing by zero.
 *
 * @param {number} value
 * @param {{low: number, high: number} | null} range
 * @returns {number}
 */
export function normalizeFillLevel(value, range) {
  if (!range || !Number.isFinite(value)) return 0.5;
  const span = range.high - range.low;
  if (span <= 0) return 0.5;
  return Math.max(0, Math.min(1, (value - range.low) / span));
}

/**
 * Linearly interpolate a downsampled timeline series at an arbitrary time.
 *
 * The backend timeline is a fetch-once, stride-downsampled array (up to
 * ~400 points for a 1440-minute run), not continuous data -- see
 * load_compartment_timeline's docstring in results.py. This is what makes
 * scrubbing/playback look smooth despite the underlying ~5-minute step
 * granularity: every animation frame re-interpolates at the current
 * simMinutes rather than snapping to the nearest sample.
 *
 * @param {number[]} times - strictly non-decreasing minutes, aligned to values
 * @param {number[]} values - same length as times
 * @param {number} minutes - arbitrary time to sample at
 * @returns {number} interpolated value, or NaN if times/values are empty
 */
export function interpolateSeries(times, values, minutes) {
  if (!Array.isArray(times) || !Array.isArray(values) || times.length === 0 || values.length === 0) {
    return NaN;
  }
  const n = Math.min(times.length, values.length);
  if (minutes <= times[0]) return values[0];
  if (minutes >= times[n - 1]) return values[n - 1];
  // Linear scan is fine here: n is at most a few hundred points, and this
  // runs once per bound series per animation frame, not per particle.
  for (let i = 1; i < n; i += 1) {
    if (minutes <= times[i]) {
      const t0 = times[i - 1];
      const t1 = times[i];
      const v0 = values[i - 1];
      const v1 = values[i];
      if (t1 === t0) return v1;
      const fraction = (minutes - t0) / (t1 - t0);
      return v0 + (v1 - v0) * fraction;
    }
  }
  return values[n - 1];
}

/**
 * A flux's sign determines which way particles should travel: a negative
 * rate means the ODE term is running in reverse (e.g. Q2->Q1 dominating
 * over Q1->Q2), which the 2D diagram already handles by flipping the arrow
 * rather than clamping the value to zero (see main.js's renderCompartmentDiagram,
 * "A negative rate means the term runs the other way"). Same invariant here.
 *
 * @param {number} rate
 * @returns {1 | -1} +1 for the schema's declared source->target direction, -1 reversed
 */
export function resolveFluxDirection(rate) {
  return rate < 0 ? -1 : 1;
}

/**
 * Map a flux's instantaneous rate to particle stream density/speed, using
 * the exact same normalization the 2D SVG diagram uses for arrow
 * stroke-width (main.js's renderCompartmentDiagram: strength = |rate| / scale,
 * scale = max(|extreme_lo|, |extreme_hi|)) -- so the two views never disagree
 * about how "strong" a flux looks at a given moment.
 *
 * @param {number} rate - instantaneous flux value at the sampled/interpolated time
 * @param {[number, number]} extreme - [min, max] of this flux over the whole run
 * @param {{minCount?: number, maxCount?: number, minSpeed?: number, maxSpeed?: number}} [options]
 */
export function computeParticleParams(rate, extreme, options = {}) {
  const { minCount = 8, maxCount = 120, minSpeed = 0.15, maxSpeed = 1.0 } = options;
  const [lo, hi] = Array.isArray(extreme) ? extreme : [0, 0];
  const scale = Math.max(Math.abs(lo), Math.abs(hi));
  // Exact parity with the 2D diagram's formula, unclamped: flux_extremes is
  // computed over the whole run and interpolateSeries never produces a value
  // outside the two samples it interpolates between, so `strength` exceeding
  // 1 should not happen for real data. Clamp only where it actually matters
  // -- turning strength into a visual count/speed -- so a fixture or a
  // future edge case can't silently break the parity this is checked
  // against in scripts/check-digital-twin.mjs.
  const strength = scale > 0 && Number.isFinite(rate) ? Math.abs(rate) / scale : 0;
  const clamped = Math.min(1, strength);
  return {
    strength,
    count: Math.round(minCount + clamped * (maxCount - minCount)),
    speed: minSpeed + clamped * (maxSpeed - minSpeed),
  };
}

/**
 * @param {number} glucoseMgdl
 * @returns {boolean} true when the plasma glucose value is in the hypoglycemic range
 */
export function resolveHypoState(glucoseMgdl) {
  return Number.isFinite(glucoseMgdl) && glucoseMgdl < HYPO_THRESHOLD_MGDL;
}

/**
 * Look up a flux's declared rate_expression (the literal ODE term, e.g.
 * "EGP_0 * max(0, 1 - x3 + x_gluc)") from the run's own schema -- the same
 * source the plan's inspection cards use so the equation shown is never
 * hand-copied/out of sync with what the backend actually integrated.
 *
 * @param {{fluxes?: Array<{key: string, rate_expression?: string}>}} schema
 * @param {string} fluxKey
 * @returns {string | null}
 */
export function lookupEquation(schema, fluxKey) {
  const flux = (schema?.fluxes || []).find((entry) => entry.key === fluxKey);
  return flux?.rate_expression ?? null;
}

/** Fixed caption for the liver's inspection card (user decision: UI-only
 * organ, not a storage compartment) -- kept as a named export, not inlined
 * in the scene module, so the headless check can assert its wording. */
export const LIVER_CARD_CAPTION =
  "Endogenous glucose production is an external source flux in this model, " +
  "not a stored compartment -- this backend does not track hepatic glycogen mass.";
