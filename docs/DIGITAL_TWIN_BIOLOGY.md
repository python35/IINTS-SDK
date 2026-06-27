# Digital Twin Biology Integrations

IINTS-AF includes optional biology helpers that connect simulator concepts to public biomedical databases.

These tools are for explanation, research context, and education. They do not replace the deterministic simulator, do not make clinical decisions, and do not provide patient-specific interpretation.

## Why This Exists

The SDK uses mathematical diabetes models. Those models contain ideas such as insulin sensitivity, insulin delay, glucagon rescue, muscle glucose uptake, and tissue compartments.

The biology helpers make those ideas easier to inspect:

- protein structure context from AlphaFold
- pathway context from STRING
- public pharmacology context from ChEMBL
- tissue-expression context from GTEx
- variant-summary context from ClinVar

The output is useful for reports, teaching, presentations, and sanity-checking model assumptions.

## 1. Structural Biology: AlphaFold

The SDK can download public AlphaFold protein predictions and render explanatory protein images.

```bash
iints render-molecules --target insulin-mutation
```

Common targets include:

- `insulin-mutation`
- `glucagon`
- `glut4`
- `insulin-receptor`
- `all`

The rendered structures are not used by the simulator. They help explain why a model might include delayed insulin action, receptor-level effects, or glucagon rescue.

## 2. PAE Heatmaps

AlphaFold Predicted Aligned Error (PAE) matrices show predicted relative-position uncertainty between residues.

```bash
iints render-pae --target all
```

The SDK writes interactive Plotly HTML files such as:

```text
results/structural/glucagon_pae.html
```

PAE is structural prediction context only. It is not a glucose metric, dosing metric, safety score, or treatment signal.

## 3. Physiological Pathways: STRING

The SDK can fetch high-resolution STRING protein-interaction network images for curated diabetes pathways.

```bash
iints render-pathways --network insulin-cascade
iints render-pathways --network glucagon-rescue
```

The pathway images help users understand how insulin signalling, GLUT4 translocation, and glucagon rescue relate to known biological networks.

They do not calibrate simulator equations automatically.

## 4. Pharmacology Context: ChEMBL

The SDK can look up public ChEMBL molecule records for insulin-related drugs and display the SDK's fixed pharmacokinetic mapping.

```bash
iints analyze-insulin --drug lispro
iints analyze-insulin --drug glargine
```

Important distinction:

- ChEMBL provides public molecule context.
- The SDK absorption values are deterministic model defaults.
- The command does not calculate patient dosing and does not infer treatment advice.

## 5. Anatomy Context: GTEx

The SDK can render interactive tissue-expression charts from GTEx v8.

```bash
iints render-expression --gene GLUT4
```

`GLUT4` is mapped to the official gene symbol `SLC2A4`. The output helps explain why muscle compartments and exercise-related glucose uptake matter in diabetes simulations.

The chart is not patient calibration.

## 6. Genetic Variant Context: ClinVar

The SDK can fetch public ClinVar summaries and show curated simulator stress-test mappings for selected genes.

```bash
iints simulate-mutation --gene INSR
iints simulate-mutation --gene INS
```

These mappings create educational virtual-patient edge cases, such as severe insulin resistance or insulin-deficiency stress tests.

They are not diagnostic genetics and should not be interpreted for a real person.

## Outputs

Most commands write files under:

```text
results/structural/
```

Examples:

```text
results/structural/insulin-cascade_string_network.png
results/structural/glucagon_pae.html
results/structural/SLC2A4_expression.html
```

## Safety Boundary

These integrations are deliberately separated from the simulator and safety supervisor.

They are allowed to help explain biological context. They are not allowed to:

- dose insulin
- diagnose a patient
- override safety rules
- alter patient-specific parameters without explicit simulator configuration
- act as clinical evidence

For deterministic formulas and model authority, see the SDK formula and numeric-authority documentation.
