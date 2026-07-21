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

The SDK can download public AlphaFold protein predictions and render explanatory protein images. The examples below are generated directly from the bundled AlphaFold DB mmCIF snapshots and coloured by per-residue pLDDT confidence.

<div class="protein-gallery">
  <figure class="protein-card">
    <a class="protein-card__media" href="https://alphafold.ebi.ac.uk/entry/P01308">
      <img src="../assets/alphafold/insulin-precursor-p01308.png" alt="AlphaFold prediction of the human insulin precursor P01308 coloured by pLDDT confidence" width="1200" height="900" loading="lazy" decoding="async">
    </a>
    <figcaption><span class="protein-card__title">Insulin precursor · INS · P01308</span>The full 110-residue precursor prediction, not an isolated mature insulin molecule. Local snapshot: AF-P01308-F1 model v4.</figcaption>
  </figure>
  <figure class="protein-card">
    <a class="protein-card__media" href="https://alphafold.ebi.ac.uk/entry/P01275">
      <img src="../assets/alphafold/proglucagon-p01275.png" alt="AlphaFold prediction of human proglucagon P01275 coloured by pLDDT confidence" width="1200" height="900" loading="lazy" decoding="async">
    </a>
    <figcaption><span class="protein-card__title">Proglucagon precursor · GCG · P01275</span>The full precursor prediction includes more than the mature 29-residue glucagon peptide. Local snapshot: AF-P01275-F1 model v4.</figcaption>
  </figure>
  <figure class="protein-card">
    <a class="protein-card__media" href="https://alphafold.ebi.ac.uk/entry/P06213">
      <img src="../assets/alphafold/insulin-receptor-p06213.png" alt="AlphaFold monomer prediction of the human insulin receptor P06213 coloured by pLDDT confidence" width="1200" height="900" loading="lazy" decoding="async">
    </a>
    <figcaption><span class="protein-card__title">Insulin receptor · INSR · P06213</span>A single-chain prediction. It does not depict the activated receptor dimer, bound insulin, membrane environment, or signalling kinetics. Local snapshot: model v6.</figcaption>
  </figure>
  <figure class="protein-card">
    <a class="protein-card__media" href="https://alphafold.ebi.ac.uk/entry/P14672">
      <img src="../assets/alphafold/glut4-p14672.png" alt="AlphaFold monomer prediction of the human GLUT4 transporter P14672 coloured by pLDDT confidence" width="1200" height="900" loading="lazy" decoding="async">
    </a>
    <figcaption><span class="protein-card__title">GLUT4 transporter · SLC2A4 · P14672</span>A monomer prediction without membrane, glucose, or transport dynamics. It provides structural context for the SDK's exercise and muscle-uptake abstractions. Local snapshot: model v6.</figcaption>
  </figure>
  <figure class="protein-card">
    <a class="protein-card__media" href="https://alphafold.ebi.ac.uk/entry/P47871">
      <img src="../assets/alphafold/glucagon-receptor-p47871.png" alt="AlphaFold monomer prediction of the human glucagon receptor P47871 coloured by pLDDT confidence" width="1200" height="900" loading="lazy" decoding="async">
    </a>
    <figcaption><span class="protein-card__title">Glucagon receptor · GCGR · P47871</span>A monomer prediction without glucagon, membrane, G protein, or downstream liver response. It is explanatory context only. Local snapshot: model v6.</figcaption>
  </figure>
</div>

<div class="alphafold-confidence" aria-label="AlphaFold pLDDT confidence colour key">
  <span class="alphafold-confidence__item"><span class="alphafold-confidence__swatch alphafold-confidence__swatch--very-high"></span>Very high: pLDDT &gt; 90</span>
  <span class="alphafold-confidence__item"><span class="alphafold-confidence__swatch alphafold-confidence__swatch--confident"></span>Confident: 70–90</span>
  <span class="alphafold-confidence__item"><span class="alphafold-confidence__swatch alphafold-confidence__swatch--low"></span>Low: 50–70</span>
  <span class="alphafold-confidence__item"><span class="alphafold-confidence__swatch alphafold-confidence__swatch--very-low"></span>Very low: &lt; 50</span>
</div>

<p class="figure-note">Structure data: <a href="https://alphafold.ebi.ac.uk/">AlphaFold Protein Structure Database</a>, data provided by Google DeepMind and EMBL-EBI under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. IINTS-AF rendered the local mmCIF snapshots with PyMOL; the images are predictions, not experimentally determined structures.</p>

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

!!! important "Read structure confidence separately from physiology"
    pLDDT describes local confidence in a predicted structure. PAE describes confidence in the relative placement of residues or domains. Neither score measures biological activity, mutation severity, insulin sensitivity, glucose effect, PK/PD, or clinical safety. IINTS-AF never converts these values into a patient parameter or treatment action.

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

The SDK can fetch public ClinVar classification context through the MyVariant.info
annotation API. The query is bounded to an exact gene symbol and exact protein
change; approximate search hits are discarded. The exported context preserves
the reported condition, classification, accession, and review status where
available.

This is an evidence lookup, not a functional-effect calculator:

- ClinVar classifications are variant-condition classifications and may evolve or conflict.
- MyVariant.info transports aggregated ClinVar annotations; it is not the classification authority.
- `Pathogenic` does **not** mean "20% receptor function" or any other quantitative scalar.
- `Benign` and absence from ClinVar do not become patient-model parameters.
- AlphaFold pLDDT remains local structure-prediction confidence only.

The multi-scale simulator therefore stays blocked unless a versioned SDK demo
assumption or separately reviewed quantitative functional measurement supplies
the scalar and provenance. ClinVar and AlphaFold context are still included in
the rejection metadata so researchers can see why no simulation was generated.

```bash
iints simulate-mutation --gene INSR
iints simulate-mutation --gene INS
```

Curated mappings create explicitly labelled educational virtual-patient edge
cases, such as severe insulin resistance or insulin-deficiency stress tests.
They are scenario assumptions, not effect estimates calculated from ClinVar.

They are not diagnostic genetics and should not be interpreted for a real person.

## 7. From Structure To Measured Chemistry And Dynamics

AlphaFold remains one structural layer. The SDK now keeps three additional reference layers separate:

- BindingDB exports measured `Ki`, `Kd`, `IC50`, and related assay records without treating them as equivalent.
- COPASI inspects and explicitly runs researcher-configured sensitivity or parameter-analysis tasks.
- CellML/OpenCOR validates independent equation models, while FMI/FMPy handles reviewed physical-device models.

```bash
iints research binding query --uniprot P06213 --output-dir results/bindingdb
iints research copasi inspect path/to/analysis.cps
iints research cellml validate path/to/model.cellml --output-dir results/cellml
iints research fmi inspect path/to/device_model.fmu
```

No structure score, assay value, fitted parameter, CellML variable, or FMU output is mapped into the virtual patient automatically. See [Cross-scale Reference Labs](CROSS_SCALE_REFERENCE_LABS.md) for execution and security boundaries.

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
