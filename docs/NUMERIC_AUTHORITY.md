# Numeric Authority and Reproducibility

IINTS-AF is a research and educational simulator, not a medical device. This
document defines which component is allowed to produce numerical results.

## Authority order

1. Mechanistic patient models calculate physiological state with explicit code.
2. Deterministic controllers calculate research candidates.
3. The independent supervisor clamps or rejects unsafe candidates.
4. ML predictors may estimate future glucose, but cannot replace physiology or
   bypass the supervisor.
5. Language models may explain supplied evidence, but have no numerical,
   diagnostic, or actuator authority.

## Deterministic guarantees

- The same dose payload and safety configuration produces the same result.
- Safety configurations carry a formula version and SHA-256 fingerprint.
- Dose calculations carry a calculation version and input fingerprint.
- Core physiological and sensor equations are registered in
  `docs/FORMULA_REGISTRY.md` and `src/iints/core/formula_registry.py`.
- Missing model checkpoints, scaler metadata, non-finite predictions, and model
  exceptions fail closed to a deterministic fallback.
- Benchmark improvements are calculated only from measured output traces.

## Controlled stochasticity

Randomness is allowed only where scientifically useful, such as synthetic test
generation, model training, and uncertainty estimation. Every such path must:

- use an explicit seed;
- avoid global random-state mutation;
- record the seed in its artifact or report;
- remain outside the final dose authority path.

MC-dropout uncertainty uses an isolated seeded generator. Its mean or standard
deviation can trigger a fallback, but a stochastic sample is never delivered as
an insulin result.

## Model and data provenance

- PyTorch checkpoints are loaded with `weights_only=True`.
- Training reports record dataset, configuration, warm-start, and model hashes.
- Training enables deterministic PyTorch algorithms and deterministic
  DataLoader shuffling.
- Golden vectors and property tests protect formula behaviour against drift.

## AI boundary

The local Mistral/Ministral assistant receives deterministic metrics and may
summarize them. It must not calculate missing values, alter reported numbers,
diagnose a patient, or generate insulin/glucagon decisions. Automated tests scan
the dose path and prompt templates for violations of this contract.

The assistant also receives a compact static formula registry. That registry is
context only: it lets the model explain which equation was used, but never asks
the model to derive, solve, or correct the equation.
