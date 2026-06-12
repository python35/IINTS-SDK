# AI Red-Team Realism Auditor

The AI Realism Auditor is a post-run verification workflow for IINTS-AF physiological simulations. It is designed for one clear question:

> Did the simulator produce a biologically plausible extreme, or did the math/data pipeline produce an impossible state?

It is not a clinical decision tool. It is an engineering and education tool for finding simulator bugs, broken assumptions, and unrealistic edge cases before a model is shown as evidence.

## What It Does

The workflow has two layers:

1. **Deterministic filter**: scans every row in a CSV for hard violations such as negative glucose, impossible rate of change, explosive ketones, or negative pump delivery.
2. **Optional local AI verdict**: sends only the small window around a flagged event to local Ollama/Ministral so the explanation stays local and context-bounded.

This avoids sending a full 14-day run to an LLM. The LLM explains flagged evidence; it does not decide insulin and it does not replace the deterministic filter.

## Install Requirements

For the full Jetson/research workflow, install the SDK with report dependencies:

```bash
python3 -m pip install -e ".[full]"
```

For local AI explanations, Ollama must be running locally:

```bash
ollama serve
ollama pull ministral-3:8b
```

On small edge devices, use the deterministic mode first:

```bash
python3 -m iints.tools.ai_realism_auditor results/red_team/endurance_data.csv --no-ai
```

## Path A: Official Jetson Endurance Output

Run the official endurance workflow:

```bash
iints jetson endurance start \
  --algo algorithms/example_algorithm.py \
  --duration 14d \
  --output-dir results/jetson_14day \
  --profile mixed_adversarial \
  --seed 42
```

Audit the raw step CSV:

```bash
python3 -m iints.tools.ai_realism_auditor \
  results/jetson_14day/raw/steps.csv \
  --report results/jetson_14day/final/AI_REALISM_AUDIT.md \
  --no-ai
```

With local Ollama enabled:

```bash
python3 -m iints.tools.ai_realism_auditor \
  results/jetson_14day/raw/steps.csv \
  --report results/jetson_14day/final/AI_REALISM_AUDIT.md \
  --model ministral-3:8b
```

## Path B: Advanced Metabolic Model Stress CSV

For an educational DKA/lipotoxicity stress demo, generate a CSV directly from the `AdvancedMetabolicModel`:

```bash
PYTHONPATH=src python3 examples/jetson_endurance_test.py \
  --days 14 \
  --output results/red_team/endurance_data.csv \
  --inject-demo-glitch
```

Then audit it:

```bash
PYTHONPATH=src python3 -m iints.tools.ai_realism_auditor \
  results/red_team/endurance_data.csv \
  --report results/red_team/AI_REALISM_AUDIT.md \
  --no-ai
```

The `--inject-demo-glitch` flag writes one deliberately corrupted row. Use it only for a self-test or booth explanation. It proves the auditor catches impossible states; it is not evidence that the physiological model naturally produced that bug.

## Red-Team Thresholds

The deterministic filter flags:

| Check | Threshold |
|---|---:|
| Negative glucose | `< 0 mg/dL` |
| Extreme hypoglycemia | `< 20 mg/dL` |
| Extreme hyperglycemia | `> 800 mg/dL` |
| Impossible glucose velocity | `abs(dG/dt) > 15 mg/dL/min` |
| Explosive ketones | `> 15 mmol/L` |
| Negative insulin delivery | `< 0 U` |

## Mathematical Model Summary

The current advanced model extends the Bergman-style state vector to 18 states:

\[
y = [G, X, I, Q_{sto1}, Q_{sto2}, Q_{gut}, S_1, S_2, Y_1, Y_2, \Gamma, x_{gluc}, HAAF, F, K, \beta, Q_{fat}, Q_{prot}]
\]

where:

- \(G\): plasma glucose in mg/dL
- \(X\): remote insulin action
- \(I\): plasma insulin
- \(F\): free fatty acids (FFA)
- \(K\): ketones
- \(\beta\): residual beta-cell mass fraction
- \(Q_{fat}\): slow fat stomach pool
- \(Q_{prot}\): slow protein pool

### Free Fatty Acids

Insulin suppresses lipolysis. The implemented FFA dynamics are:

\[
\frac{dF}{dt} = \ell_0 e^{-\ell_1 I} - k_f F
\]

with the current parameters:

\[
\ell_0 = 0.2, \quad \ell_1 = 0.23, \quad k_f = 0.1
\]

### Ketones

Ketone production is driven by high FFA and low insulin:

\[
\frac{dK}{dt} = k_0 F e^{-k_1 I} - k_2 K
\]

with:

\[
k_0 = 0.125, \quad k_1 = 0.33, \quad k_2 = 0.05
\]

### Lipotoxic Insulin Resistance

High FFA reduces insulin sensitivity through:

\[
L(F) = \frac{0.4}{\max(0.4, F)}
\]

and the effective insulin-action gain becomes:

\[
p_{3,eff} = p_3 \cdot M_{exercise} \cdot M_{stress} \cdot L(F)
\]

### Beta-Cell Decay

Residual beta-cell mass decays exponentially:

\[
\frac{d\beta}{dt} = -a \beta
\]

where \(a\) is `autoimmune_aggressiveness`.

### Endogenous Glucose Production

The T1D instability upgrade removes the old automatic pull back to basal glucose and models glucose as:

\[
\frac{dG}{dt} = -XG + EGP + R_a + D - U_{exercise} - F_R
\]

where:

\[
EGP = p_{1,eff}G_{b,eff}
\]

and starvation/hepatic resistance scales the effective basal term:

\[
S = e^{-0.4I}\frac{\max(F, 0.4)}{0.4}
\]

\[
G_{b,eff} = G_b \cdot M_{stress} \cdot M_{rescue} \cdot \max(0,1+x_{gluc}) \cdot (1 + 3S)
\]

### Safety Guard

The implementation still applies simulator guards after ODE solving:

\[
G_{t+1} = \max(20, G_t + \mathrm{clip}(G^*_{t+1}-G_t, -r_{max}\Delta t, r_{max}\Delta t))
\]

This means a negative glucose value should not appear in normal model output. If it appears in a CSV, the auditor should treat it as data corruption or a mathematical bug.

## What To Say At A Booth

Use this phrasing:

> We run the digital patient for many simulated days. A fast red-team filter scans every row for impossible biology. If something suspicious appears, a local AI explains the small evidence window in plain language. The AI explains the bug; it does not control the pump.

That distinction is important and credible.
