# Local AI Safety Gates

The local-AI pipeline is intentionally critical. A controller can train successfully and still be blocked from research promotion.

## What Gets Checked

The training gate reviews:

- minimum training rows;
- maximum teacher insulin label;
- unsafe insulin proposals during hypoglycemia;
- proposals above 5 U;
- whether the training target came from `reference_teacher_insulin_units`;
- whether deterministic supervisor limits are still required.

The held-out closed-loop gate reviews every candidate against `clinical_baseline`:

- no early terminations;
- completion percentage;
- time below 54 mg/dL regression;
- time below 70 mg/dL regression;
- supervisor-intervention regression;
- large TIR drop.

## Why This Matters

For diabetes technology research, a model that improves average glucose but increases severe hypoglycemia is not an improvement. The SDK therefore treats hypo regression and safety-supervisor burden as blocking issues.

## Output

`iints research local-ai-lab` now writes:

- `training_safety_gate` in `LOCAL_AI_RESEARCH_SUMMARY.json`;
- `closed_loop_evaluation.safety_gate` when evaluation is enabled;
- a report section explaining why a local model is blocked, review-only, or research-ready.

## Boundary

These gates are pre-clinical research gates. They do not approve a model for pump control or patient dosing.
