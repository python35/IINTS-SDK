# Mechanistic Reference Model Lab

AlphaFold helps inspect **molecular structure**. It does not calculate how glucose, insulin, metabolites, pumps, or sensors change over time. The Mechanistic Reference Model Lab adds that missing mathematical layer by running an external model independently from the IINTS patient simulator.

!!! warning "Research boundary"
    Successful execution means only that a numerical engine could run the supplied equations. It is not evidence that the equations, parameters, population, units, or outcomes are biologically correct. This workflow is for educational and pre-clinical research only and must not be used for treatment or insulin dosing.

## Structure Is Not Dynamics

These two AlphaFold predictions make the boundary concrete: a structural model can show a plausible fold and its local confidence, but it does not calculate absorption, receptor activation, insulin action, glucose transport, or a time-course response.

<div class="protein-gallery protein-gallery--pair">
  <figure class="protein-card">
    <a class="protein-card__media" href="https://alphafold.ebi.ac.uk/entry/P01308">
      <img src="../assets/alphafold/insulin-precursor-p01308.png" alt="AlphaFold prediction of the human insulin precursor P01308 coloured by pLDDT confidence" width="1200" height="900" loading="lazy" decoding="async">
    </a>
    <figcaption><span class="protein-card__title">Predicted molecular structure</span>Human insulin precursor, UniProt P01308. AlphaFold pLDDT reports local structural confidence; it does not provide an insulin absorption or action curve.</figcaption>
  </figure>
  <figure class="protein-card">
    <a class="protein-card__media" href="https://alphafold.ebi.ac.uk/entry/P06213">
      <img src="../assets/alphafold/insulin-receptor-p06213.png" alt="AlphaFold monomer prediction of the human insulin receptor P06213 coloured by pLDDT confidence" width="1200" height="900" loading="lazy" decoding="async">
    </a>
    <figcaption><span class="protein-card__title">Predicted receptor structure</span>Human insulin receptor, UniProt P06213. This is a monomer prediction, not a ligand-bound receptor complex or signalling simulation.</figcaption>
  </figure>
</div>

<p class="figure-note">The Mechanistic Reference Model Lab starts where these pictures stop: it executes explicit equations over time. Structure and dynamics remain separate evidence layers, and neither is clinical validation.</p>

## What Is Implemented

The current workflow supports local [SBML](https://sbml.org/) `.xml` and `.sbml` files.

| Capability | Available | Meaning |
| --- | --- | --- |
| Size-limited, entity-safe XML inspection | yes | reads model structure without executing equations |
| SBML level/version and namespace summary | yes | records basic interchange metadata |
| Species, compartments, global parameters, units, rules, reactions, and event counts | yes | creates a reviewable structural inventory |
| SHA-256 model fingerprint | yes | identifies the exact local model file used |
| Independent time-course execution | optional | uses [libRoadRunner](https://libroadrunner.readthedocs.io/) when installed |
| Isolated CSV, manifest, model summary, and review report | yes | preserves provenance and limitations per execution |
| Full SBML schema validation | no | use a dedicated SBML validator before publication |
| Automatic unit conversion to mg/dL or minutes | no | mappings must be explicit and reviewed |
| Automatic calibration of IINTS parameters | no | external models never silently tune the digital patient |

The IINTS Hovorka, Bergman, and other patient models remain Python implementations. Loading an external SBML file does not make those implementations SBML-compatible and does not replace them.

## Install The Optional Engine

Inspection works with the normal SDK. Time-course execution requires libRoadRunner:

```bash
python -m pip install -U "iints-sdk-python35[mechanistic]"
```

libRoadRunner distributes platform-specific binaries. If no compatible package exists for an edge platform such as a Jetson, inspect the model there but execute it on a supported workstation. Do not copy an unverified binary into a research environment.

## Inspect Before Running

```bash
iints research mechanistic inspect path/to/model.sbml \
  --output-json results/model_inspection.json
```

Review at least:

1. the exact model identifier, source, licence, and hash
2. declared time, substance, concentration, and compartment units
3. warnings about missing units, identifiers, dynamics, or SBML packages
4. whether the model population and assumptions match the research question
5. whether the selected variables represent amounts, concentrations, rates, or dimensionless states

Inspection is deliberately separate from execution so malformed or poorly documented files can be rejected before a numerical engine sees them.

## Run An Independent Reference

```bash
iints research mechanistic run path/to/model.sbml \
  --output-dir results/mechanistic_reference \
  --start 0 \
  --end 1440 \
  --points 289 \
  --variable glucose \
  --variable insulin \
  --source-url https://example.org/exact-model-record \
  --model-license NOASSERTION
```

`start` and `end` stay in the model's own time units. The SDK does not assume they are minutes. Use `[G]` or `concentration:G` to request a species concentration and `amount:G` to request an amount. An unqualified species follows its SBML `hasOnlySubstanceUnits` declaration. Global parameters can also be selected explicitly. The manifest records the semantic choice and declared units for every output.

Each run receives its own timestamped folder:

```text
sbml_model_timestamp_hash/
|-- reference_results.csv
|-- sbml_model_summary.json
|-- mechanistic_run_manifest.json
`-- REFERENCE_MODEL_REPORT.md
```

The manifest records the model hash, source, licence, SBML version, engine and integrator versions, simulation interval, selections, row count, output names, and simple finite summary statistics. Absolute local model paths are not written into the shareable run artifacts.

## Desktop Workflow

In the Tauri research workbench, open **Biology and evidence** and use **Mechanistic Reference Model Lab**:

1. select or paste a local SBML file path
2. select **Inspect SBML**
3. review units, counts, readiness, and warnings
4. install the optional engine only if execution is needed
5. enter model-time bounds and explicit variables
6. record an exact HTTPS source and the model-artifact licence
7. run the reference and review the CSV, manifest, and report together

The app calls the same Python SDK functions as the CLI. Rust validates the bounded request but does not reimplement scientific equations.

## How This Improves Realism

Use the reference model to challenge an IINTS assumption, not to declare either model correct. Useful comparisons include:

- glucose and insulin response timing under matched initial conditions
- sensitivity of a mechanism to one parameter at a time
- conservation, positivity, steady-state, and finite-value behaviour
- disagreement caused by different meal, insulin, hepatic, or counter-regulatory assumptions
- numerical stability across step sizes and solvers

Before plotting two outputs together, create a reviewed mapping table:

| Item | External model | IINTS | Required decision |
| --- | --- | --- | --- |
| Time | declared model unit | minutes | conversion and origin |
| Glucose | amount or concentration | mg/dL | compartment and molecular conversion |
| Insulin | amount, concentration, or action state | delivered/active/effect states | physiological interpretation |
| Population | model-specific | selected patient profile | comparability and exclusions |

Never fit a conversion merely because it makes two curves look similar.

## Cross-scale Toolchain

| Tool or standard | Scientific role | IINTS maturity |
| --- | --- | --- |
| [libRoadRunner](https://libroadrunner.readthedocs.io/) + SBML | independent biochemical ODE and reaction-network execution | integrated optional backend |
| [COPASI](https://copasi.org/) | sensitivity, configured parameter estimation, scans, and stochastic kinetics | integrated inspection and explicit CopasiSE task execution |
| [OpenCOR](https://opencor.ws/) + [Physiome CellML](https://models.physiomeproject.org/cellml) | independent CellML structure and standards validation | integrated inspection and OpenCOR validation; no automatic coupling |
| [FMI](https://fmi-standard.org/docs/3.0.2/) + [FMPy](https://fmpy.readthedocs.io/) | pump, motor, fluid-flow, occlusion, and sensor co-simulation | integrated static inspection and explicitly trusted native execution |
| [BindingDB](https://www.bindingdb.org/rwd/bind/index.jsp) | measured molecular affinity context beside predicted structure | integrated read-only UniProt evidence export |

The detailed commands, security boundaries, and output bundles are documented in [Cross-scale Reference Labs](CROSS_SCALE_REFERENCE_LABS.md). The integrations preserve source/version metadata, units, local artifacts, limitations, and a no-silent-calibration boundary. They do not claim automatic model equivalence or clinical validation.

## Interpretation Rules

1. Structure confidence, binding affinity, biochemical kinetics, whole-body physiology, and clinical outcomes are different evidence levels.
2. AlphaFold pLDDT or PAE must never become insulin sensitivity or mutation severity automatically.
3. `Kd`, `Ki`, and `IC50` are assay-specific and are not interchangeable PK/PD parameters.
4. A model repository label does not prove suitability for type 1 diabetes or a specific population.
5. Agreement between two models can reflect shared assumptions; disagreement can be scientifically useful.
6. Preserve failed executions and rejected mappings in the research record.
7. Use measured data and preregistered evaluation criteria for calibration and validation.

For the broader reproducibility package, continue with [Academic Research Workbench](ACADEMIC_RESEARCH_WORKBENCH.md).
