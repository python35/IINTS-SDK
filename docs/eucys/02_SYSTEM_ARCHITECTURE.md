# System Architecture

## Architectural principle

The most important separation in IINTS-AF is:

> The patient model calculates state, a candidate algorithm proposes an action,
> deterministic safety logic decides what is allowed in the simulation, and
> reporting consumes the resulting evidence afterward.

No language model is part of the numerical physiology solver or final simulated
action authority.

## End-to-end architecture

<!-- diagram:system-architecture -->
```mermaid
flowchart LR
    A["Scenario, patient profile<br/>dataset and seed"] --> B["Data and configuration<br/>validation"]
    B --> C["Simulation orchestrator"]
    C --> D["Virtual patient<br/>Custom / Bergman / Hovorka"]
    D --> E["CGM-like sensor<br/>lag, drift, seeded noise"]
    E --> F["Candidate algorithm"]
    F --> G["Deterministic safety<br/>supervisor"]
    G -->|approved / reduced / blocked| D
    C --> H["Run recorder"]
    G --> H
    H --> I["CSV + JSON manifests<br/>audit and validation"]
    I --> J["Metrics, AGP-style report<br/>poster and evidence bundle"]
    I -. evidence only .-> K["Optional local AI<br/>explanation and review"]
```

## One simulation step

<!-- diagram:simulation-step -->
```mermaid
sequenceDiagram
    participant S as Scenario
    participant P as Patient model
    participant C as CGM sensor
    participant A as Candidate algorithm
    participant V as Safety supervisor
    participant R as Recorder

    S->>P: Apply meal, exercise, stress or device event
    P->>P: Integrate deterministic state equations
    P->>C: Provide latent blood glucose
    C->>A: Provide CGM-like reading and context
    A->>V: Propose insulin or research glucagon action
    V->>V: Validate values, limits, trend and active insulin
    V->>P: Apply approved, reduced or blocked action
    P->>R: Record next physiological state
    V->>R: Record decision, reason and safety event
```

The exact ordering depends on the selected workflow, but the authority boundary
does not change: an experimental proposal cannot bypass deterministic
validation.

## Numeric authority

<!-- diagram:numeric-authority -->
```mermaid
flowchart TD
    P["Mechanistic patient model<br/>calculates physiological state"] --> C["Deterministic controller or<br/>research model proposes candidate"]
    C --> S["Independent deterministic supervisor<br/>checks and constrains candidate"]
    S --> O["Recorded simulated output"]
    O --> M["Deterministic metrics and validation"]
    O -. read-only artifacts .-> L["Local language model explanation"]
    L -. no numerical or actuator authority .-> M
```

Authority order:

1. Explicit model code calculates physiological state.
2. Deterministic or learned algorithms may calculate a candidate.
3. The supervisor validates, clamps or rejects the candidate.
4. Metric code calculates results from recorded traces.
5. A language model may explain supplied values but may not replace them.

## Source-layer architecture

```mermaid
flowchart TD
    UI["Interfaces<br/>CLI and Rust/Tauri desktop"] --> APP["Application workflows<br/>highlevel, analysis, research, AI, edge"]
    APP --> DATA["Adapters and evidence<br/>data and validation"]
    APP --> DOMAIN["Domain core<br/>simulator, patient, safety, API"]
    DATA --> DOMAIN
    DOMAIN --> CONTRACT["Small contracts<br/>algorithm API, units, formula registry"]
```

| Layer | Important paths | Responsibility |
| --- | --- | --- |
| Domain core | `src/iints/core/`, `src/iints/api/` | Patient state, simulator, algorithm contract and deterministic safety |
| Data and validation | `src/iints/data/`, `src/iints/validation/` | Imports, contracts, realism, replay and evidence checks |
| Application workflows | `src/iints/highlevel.py`, `src/iints/analysis/`, `src/iints/research/`, `src/iints/ai/` | Run orchestration, reports, studies and optional research AI |
| Edge adapters | `src/iints/live_patient/`, `src/iints/jetson/` | Bench-only hardware and endurance workflows |
| Interfaces | `src/iints/cli/`, `apps/iints-tauri/`, `src/iints_desktop/` | User interaction without redefining domain rules |

The core is intentionally dependency-light. It must not import the CLI,
reporting UI, training orchestration or hardware presentation layer. The
repository checks these boundaries with
`tools/ci/check_architecture_boundaries.py`.

## Evidence lifecycle

<!-- diagram:evidence-lifecycle -->
```mermaid
flowchart LR
    A["Protocol<br/>question, matrix, seed policy"] --> B["Execution<br/>fixed configuration"]
    B --> C["Raw trace<br/>results.csv"]
    C --> D["Integrity<br/>metadata and manifest"]
    C --> E["Validation<br/>safety, realism, contracts"]
    C --> F["Analysis<br/>metrics and comparisons"]
    D --> G["Evidence bundle"]
    E --> G
    F --> G
    G --> H["Human review<br/>report, poster, jury dossier"]
```

Typical artifacts:

| Artifact | What it proves |
| --- | --- |
| `results.csv` | The recorded state, observations, events and actions |
| `run_metadata.json` | Patient, scenario, algorithm, duration, step and seed |
| `run_manifest.json` | File inventory and integrity information |
| `audit/` or safety report | Candidate versus accepted action and intervention reason |
| `validation_report.json` | Deterministic checks and warnings |
| `realism_report.json` | Plausibility comparison against a documented reference |
| `report.pdf` / AGP-style assets | Human-readable interpretation of the same trace |
| `sources_manifest.json` | Scientific and data-source context |

## Desktop application boundary

<!-- diagram:desktop-bridge -->
```mermaid
flowchart LR
    U["Researcher"] --> T["Rust/Tauri desktop shell"]
    T --> R["Allowlisted Rust commands<br/>path and argument validation"]
    R --> P["Private Python SDK engine"]
    P --> O["Workspace outputs"]
    O --> T
    P -. optional localhost only .-> L["Ollama"]
    P -. explicit research request .-> X["Public scientific APIs"]
```

The desktop application is an interface to the same Python SDK. It should not
contain a second physiology implementation. Rust/Tauri narrows desktop command
execution and file access; Python remains the research engine.

## Hardware boundary

```mermaid
flowchart LR
    S["IINTS simulation and test scenario"] --> B["Bench adapter"]
    B --> H["Pi / UNO Q / Jetson / FPGA mock or hardware"]
    H --> C["Software-versus-hardware comparison"]
    C --> E["Latency, mismatch and evidence report"]
```

All hardware paths are research and bench-only. They may demonstrate protocols,
timing, deterministic safety logic or edge execution. They are not authorised
to deliver medication to a person.

## Failure containment

| Failure | Required response |
| --- | --- |
| Missing or invalid model checkpoint | Fail closed or use a documented deterministic fallback |
| Non-finite physiology or prediction | Reject, terminate or flag; never silently continue |
| Sensor dropout or invalid glucose | Mark observation invalid and invoke configured safety behaviour |
| AI unavailable | Preserve the run; omit optional explanation |
| AI contradicts a metric | Deterministic artifact remains authoritative |
| Optional scientific API unavailable | Preserve cached/local work and report the missing context |
| Report generation fails | Keep raw trace, metadata and validation artifacts |
