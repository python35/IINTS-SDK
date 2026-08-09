# Claim-to-Evidence Map

This page tells a reviewer where to verify each important statement. A claim is
strong only when its implementation, test, output and scientific basis are
kept separate.

## Core project claims

| Claim | Implementation | Verification or output | Boundary |
| --- | --- | --- | --- |
| The SDK simulates virtual diabetes scenarios | `src/iints/core/simulator.py`, `src/iints/core/patient/` | Core, scenario and preset tests; `results.csv` | Research approximation |
| Candidate actions pass a separate safety layer | `src/iints/core/supervisor.py`, `src/iints/core/safety/` | Safety tests, audit events and safety report | Not proof of clinical safety |
| Physiology is deterministic code, not LLM output | `src/iints/core/formula_registry.py`, patient model `_ode` methods | Formula registry tests and numeric-authority checks | Parameter validity still matters |
| CGM can differ from latent glucose | `src/iints/core/devices/models.py` | Sensor tests and separate result columns | Generic, not vendor-equivalent |
| Runs produce inspectable evidence | `src/iints/highlevel.py`, `src/iints/analysis/`, `src/iints/validation/` | CSV, metadata, manifests, reports | Artifact set varies by command |
| Data can be contract-checked | `src/iints/data/` | MDMP certificate and quality report | Certification is not clinical approval |
| AI is advisory in explanation mode | `src/iints/ai/`, `docs/NUMERIC_AUTHORITY.md` | Prompt/policy tests and separate AI artifacts | Human review remains required |
| Desktop app calls the same SDK | `apps/iints-tauri/`, `src/iints_desktop/` | Rust checks, bridge tests and desktop smoke tests | Packaging/security need maintenance |
| Edge and FPGA paths are bench-only | `src/iints/live_patient/`, `src/iints/jetson/` | Mock, protocol and hardware-adapter tests | No medication actuation |

## Formula evidence map

Runtime paths in this table are relative to `src/iints/core/`.

| ID | Runtime authority | Scientific basis |
| --- | --- | --- |
| F01 Bergman glucose | `patient/bergman_model.py` | Bergman 1979 plus declared extensions |
| F02 remote insulin | `patient/bergman_model.py` | Bergman 1979 |
| F03 plasma insulin | `patient/bergman_model.py` | Bergman balance plus disabled-by-default research secretion |
| F04 subcutaneous insulin | Bergman and Hovorka patient models | Hovorka 2004 and PK abstraction |
| F05 meal absorption | Bergman and Hovorka patient models | Published Hovorka two-compartment chain; explicitly adapted three-stage Bergman branch |
| F06 Hovorka glucose mass | `patient/hovorka_model.py` | Hovorka 2004 plus declared extensions |
| F07 insulin-action channels | `patient/hovorka_model.py` | Hovorka 2004 action channels plus heuristic molecular/tissue sensitivity scalars |
| F08 stress/exercise | `patient/hovorka_model.py` | Research pseudo-hormone abstraction |
| F09 GLUT4/NIMGU | `patient/hovorka_model.py` | Exercise/GLUT4 physiology context |
| F10 circadian EGP | `patient/hovorka_model.py` | Dawn-phenomenon context, gated approximation |
| F11 hypo rescue | Bergman and Hovorka patient models | Counterregulation and HAAF context |
| F12 HAAF memory | Bergman and Hovorka patient models | Cryer 2013, experimental memory state |
| F13 glucagon PK/PD | Bergman and Hovorka patient models | Published exogenous-glucagon context |
| F14 renal clearance | `patient/physiology.py` and both ODE models | Renal threshold/splay context |
| F15 CGM observation | `devices/models.py`, `SensorModel.read` | Blood-to-ISF lag and sensor context |

Canonical source:
`src/iints/core/formula_registry.py`. Generated human reference:
`docs/FORMULA_REGISTRY.md`.

## Benchmark evidence map

Paths in this table are relative to `research/eucys_pack/`, except where noted.
The named study runner lives in `tools/research/`.

| Evidence | Repository path | Review question |
| --- | --- | --- |
| Aggregate arm table | `assets/EUCYS_RESULTS_TABLE.csv` | Do arm counts and metrics match the report? |
| Algorithm figure data | `assets/EUCYS_MAIN_FIGURE.csv` | Can every plotted bar be reconstructed? |
| Main figure | `assets/EUCYS_MAIN_FIGURE.png` | Does visual labelling match the table? |
| Full report source | `../EUCYS_REPORT.md` | Are protocol, results and limitations stated? |
| Final workflow | `../EUCYS_FINAL_WORKFLOW.md` | Can the benchmark and report be regenerated? |
| Study runner | `run_eucys_final.sh` | Is the command path explicit? |

## Scientific source map

| Topic | Primary source used in SDK documentation |
| --- | --- |
| Glycaemic targets and hypoglycaemia | ADA Standards of Care 2026, DOI [10.2337/dc26-S006](https://doi.org/10.2337/dc26-S006) |
| Time in range | Battelino et al. 2019, DOI [10.2337/dci19-0028](https://doi.org/10.2337/dci19-0028) |
| Bergman minimal model | Bergman et al. 1979, DOI [10.1152/ajpendo.1979.236.6.E667](https://doi.org/10.1152/ajpendo.1979.236.6.E667) |
| Hovorka model | Hovorka et al. 2004, DOI [10.1088/0967-3334/25/4/010](https://doi.org/10.1088/0967-3334/25/4/010) |
| Meal absorption | Dalla Man et al. 2007, DOI [10.1109/TBME.2007.893506](https://doi.org/10.1109/TBME.2007.893506) |
| CGM lag | Wentholt et al. 2004, DOI [10.1089/dia.2004.6.615](https://doi.org/10.1089/dia.2004.6.615) |
| Exercise in T1D | Riddell et al. 2017, DOI [10.1016/S2213-8587(17)30014-1](https://doi.org/10.1016/S2213-8587(17)30014-1) |
| Exercise and GLUT4 | Richter and Hargreaves 2013, DOI [10.1152/physrev.00038.2012](https://doi.org/10.1152/physrev.00038.2012) |
| HAAF | Cryer 2013, DOI [10.1056/NEJMra1215228](https://doi.org/10.1056/NEJMra1215228) |
| Renal glucose handling | Hummel et al. 2018, DOI [10.1007/s00125-018-4656-5](https://doi.org/10.1007/s00125-018-4656-5) |
| OhioT1DM | Marling and Bunescu 2020, [CEUR paper](http://ceur-ws.org/Vol-2675/paper2.pdf) |
| AGP interpretation | International Diabetes Center guide and TIR consensus |

The complete maintained source list is in `docs/EVIDENCE_BASE.md` and
`docs/SOURCE_LIBRARY.md`.

## Mermaid diagram sources

The diagrams in this dossier are both embedded in the Markdown pages and stored
as reusable Mermaid source:

| Diagram | Source |
| --- | --- |
| System architecture | `docs/eucys/diagrams/system-architecture.mmd` |
| Simulation step | `docs/eucys/diagrams/simulation-step.mmd` |
| Numeric authority | `docs/eucys/diagrams/numeric-authority.mmd` |
| Evidence lifecycle | `docs/eucys/diagrams/evidence-lifecycle.mmd` |
| Desktop bridge | `docs/eucys/diagrams/desktop-bridge.mmd` |
| AI boundary | `docs/eucys/diagrams/ai-boundary.mmd` |
| Data lifecycle | `docs/eucys/diagrams/data-lifecycle.mmd` |
| Cross-scale evidence | `docs/eucys/diagrams/cross-scale-evidence.mmd` |
| Validation ladder | `docs/eucys/diagrams/validation-ladder.mmd` |

## Reproduction commands

Build this dossier:

```bash
tools/research/build_eucys_dossier.sh
```

Build the existing evidence PDFs:

```bash
tools/research/build_eucys_pack.sh
```

Run the final benchmark workflow:

```bash
tools/research/run_eucys_final.sh \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_2026 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --no-prepare-ai
```

Build and validate the documentation:

```bash
mkdocs build --strict
```

Run the principal software checks:

```bash
python3 tools/ci/check_architecture_boundaries.py
python3 -m pytest tests/ -q
mypy src/iints/
```

## Review checklist

A reviewer should be able to trace:

1. A sentence in the report to a table or raw trace.
2. A table value to deterministic metric code.
3. A simulated state to a registered equation and parameter set.
4. A safety intervention to a candidate, accepted action and reason.
5. A dataset to a source and transformation manifest.
6. An AI statement to supplied evidence.
7. A biological context view to its public source and interpretation boundary.
8. A release artifact to a tagged software version.
