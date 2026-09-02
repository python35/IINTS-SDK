/**
 * Scientific primitives for the desktop workbench.
 *
 * These are deliberate ports of the Python implementations so the desktop app
 * and the SDK cannot report different numbers for the same data:
 *   - clarkeZones / clarkeErrorGrid  <-  src/iints/analysis/error_grid.py
 *   - glycemicBands                  <-  international consensus (Battelino 2019)
 *
 * Rule for every chart in this app: a displayed percentage must be counted
 * from the points that are actually plotted. If the points are synthetic, the
 * chart must say so. Never write a percentage into a label by hand.
 *
 * References
 * ----------
 * Clarke WL, Cox D, Gonder-Frederick LA, Carter W, Pohl SL. Evaluating
 *   clinical accuracy of systems for self-monitoring of blood glucose.
 *   Diabetes Care. 1987;10(5):622-628.
 * Battelino T, et al. Clinical targets for continuous glucose monitoring data
 *   interpretation: recommendations from the international consensus on time
 *   in range. Diabetes Care. 2019;42(8):1593-1603.
 */

export const ZONES = ["A", "B", "C", "D", "E"];

/**
 * Deterministic pseudo-random generator (mulberry32).
 *
 * Demonstration traces must be reproducible. With Math.random() the plotted
 * points change on every repaint, so any percentage shown alongside them is
 * describing a picture that no longer exists.
 */
export function seededRandom(seed = 42) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Classify one (reference, predicted) pair in mg/dL into a Clarke zone. */
export function clarkeZone(ref, pred) {
  if (!Number.isFinite(ref) || !Number.isFinite(pred) || ref <= 0) return null;

  // Zone A: within 20% of reference, or both in the hypo range where a
  // relative band is not meaningful.
  if ((ref <= 70 && pred <= 70) || Math.abs(pred - ref) / ref <= 0.2) return "A";

  // Zone E: treatment would be the opposite of what is required.
  if ((ref >= 180 && pred <= 70) || (ref <= 70 && pred >= 180)) return "E";

  // Zone C: would prompt over-correction of a value needing none.
  if ((ref >= 70 && ref <= 290 && pred >= ref + 110) ||
      (ref >= 130 && ref <= 180 && pred <= (7 / 5) * ref - 182)) return "C";

  // Zone D: dangerous failure to detect a true excursion.
  if ((ref >= 240 && pred >= 70 && pred <= 180) ||
      (ref <= 175 / 3 && pred >= 70 && pred <= 180) ||
      (ref >= 175 / 3 && ref <= 70 && pred >= (6 / 5) * ref)) return "D";

  return "B";
}

/**
 * Count Clarke zones over paired arrays.
 * @returns {{counts: Object, percentages: Object, nPairs: number,
 *            hazardousPct: number}}
 * @throws if no valid pair is supplied - there is no synthetic fallback.
 */
export function clarkeErrorGrid(reference, predicted) {
  if (reference.length !== predicted.length) {
    throw new Error(`shape mismatch: ${reference.length} vs ${predicted.length}`);
  }
  const counts = { A: 0, B: 0, C: 0, D: 0, E: 0 };
  let n = 0;
  for (let i = 0; i < reference.length; i++) {
    const z = clarkeZone(reference[i], predicted[i]);
    if (z === null) continue;
    counts[z] += 1;
    n += 1;
  }
  if (n === 0) {
    throw new Error("Clarke EGA requires at least one valid (reference, predicted) pair");
  }
  const percentages = {};
  for (const z of ZONES) percentages[z] = (100 * counts[z]) / n;
  const hazardousPct = percentages.C + percentages.D + percentages.E;
  return { counts, percentages, nPairs: n, hazardousPct };
}

/**
 * Time in the international consensus glucose bands, as percentages.
 * Bands: <54, 54-69, 70-180, 181-250, >250 mg/dL.
 *
 * Note this is time-in-band by reading count, which equals time only on an
 * even sampling grid. On gappy CGM data, resample before calling this.
 */
export function glycemicBands(glucose) {
  const g = glucose.filter((v) => Number.isFinite(v) && v > 0);
  if (g.length === 0) throw new Error("no valid glucose values supplied");
  const b = { vlow: 0, low: 0, tir: 0, high: 0, vhigh: 0 };
  for (const v of g) {
    if (v < 54) b.vlow += 1;
    else if (v < 70) b.low += 1;
    else if (v <= 180) b.tir += 1;
    else if (v <= 250) b.high += 1;
    else b.vhigh += 1;
  }
  const out = { n: g.length };
  for (const k of ["vlow", "low", "tir", "high", "vhigh"]) {
    out[k] = (100 * b[k]) / g.length;
  }
  return out;
}

/** Coefficient of variation (%), sample standard deviation (n-1). */
export function coefficientOfVariation(glucose) {
  const g = glucose.filter((v) => Number.isFinite(v) && v > 0);
  if (g.length < 2) throw new Error("need at least two values");
  const mean = g.reduce((s, v) => s + v, 0) / g.length;
  const varr = g.reduce((s, v) => s + (v - mean) ** 2, 0) / (g.length - 1);
  return (100 * Math.sqrt(varr)) / mean;
}

/**
 * Draw the banner that must accompany any chart built from generated data.
 * Keeping this in one place means a synthetic chart cannot quietly lose its
 * label during a refactor.
 */
export function drawSyntheticBanner(ctx, x, y, text = "SYNTHETIC DEMONSTRATION DATA - not a measurement") {
  ctx.save();
  ctx.font = "bold 10px system-ui, sans-serif";
  const padX = 6;
  const wTxt = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(217, 48, 37, 0.10)";
  ctx.strokeStyle = "#d93025";
  ctx.lineWidth = 1;
  ctx.beginPath();
  // roundRect is unavailable in older webviews; the banner must never be the
  // reason a chart throws, so fall back to a plain rectangle.
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, wTxt + 2 * padX, 18, 4);
  } else {
    ctx.rect(x, y, wTxt + 2 * padX, 18);
  }
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#a50e0e";
  ctx.fillText(text, x + padX, y + 13);
  ctx.restore();
}
