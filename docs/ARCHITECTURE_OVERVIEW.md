# Visual Architecture

This page gives code contributors the shortest possible map of how the IINTS-AF SDK fits together before they edit a subsystem.

## End-To-End Flow

```mermaid
flowchart LR
    A["Datasets, scenarios, and patient profiles"] --> B["Data layer\nsrc/iints/data"]
    B --> C["Simulation orchestration\nsrc/iints/highlevel.py"]
    C --> D["Algorithm API\nsrc/iints/api"]
    C --> E["Virtual patient + simulator\nsrc/iints/core"]
    D --> F["Candidate insulin decision"]
    E --> F
    F --> G["Deterministic safety supervisor\nsrc/iints/core/safety"]
    G --> H["Run bundle\nCSV + metadata + audit + report"]
    H --> I["Analysis and posters\nsrc/iints/analysis"]
    H --> J["Optional AI explanation\nsrc/iints/ai"]
    H --> K["Validation + replay\nsrc/iints/validation"]
```

The important design rule is that a candidate algorithm may propose insulin, but the deterministic safety supervisor remains a separate hard gate before outputs are accepted.

## Source Layout

```mermaid
flowchart TD
    CLI["cli/\nPublic commands"] --> HIGH["highlevel.py\nWorkflow orchestration"]
    CLI --> LIVE["live_patient/ + jetson/\nEdge runtime"]
    HIGH --> API["api/\nAlgorithm contract"]
    HIGH --> CORE["core/\nSimulator, patient, safety"]
    HIGH --> DATA["data/\nImport, registry, MDMP, realism"]
    HIGH --> ANALYSIS["analysis/\nMetrics, reports, studies"]
    ANALYSIS --> AI["ai/\nLocal explanation layer"]
    CORE --> VALIDATION["validation/\nReplay + safety contracts"]
    LIVE --> CORE
    LIVE --> DATA
    RESEARCH["research/\nPredictor training helpers"] --> CORE
```

## One Simulation Step

```mermaid
sequenceDiagram
    participant Scenario
    participant Patient
    participant Algorithm
    participant Supervisor
    participant Recorder

    Scenario->>Patient: meal / exercise / sensor event
    Patient->>Algorithm: glucose, IOB, carbs, context
    Algorithm->>Supervisor: candidate insulin decision
    Supervisor->>Supervisor: validate bounds, trends, IOB, contract
    Supervisor->>Patient: accepted or overridden dose
    Patient->>Recorder: next simulated state
    Supervisor->>Recorder: intervention reason if any
```

## Edge And Hardware Boundary

```mermaid
flowchart LR
    A["Linux-side SDK runtime\nPi / UNO Q Linux side / Jetson"] --> B["Patient runtime + safety logic"]
    B --> C["HTTP / serial bridge"]
    C --> D["UNO Q microcontroller sketch"]
    B --> E["Snapshots, archives, reports"]
```

The microcontroller bridge is intentionally thin. The main simulation state, safety policy, replay data, and reporting stay on the Linux-side SDK runtime.

## Where To Edit What

| If You Need To Change | Start In |
| --- | --- |
| dose validation, safety limits, override reasons | `src/iints/core/safety/`, `src/iints/core/supervisor.py` |
| simulation timing, patient updates, event handling | `src/iints/core/simulator.py`, `src/iints/core/patient/` |
| algorithm plugin contract | `src/iints/api/` |
| command behavior or user-facing flows | `src/iints/cli/` |
| dataset imports, realism, certification | `src/iints/data/` |
| reports, posters, study summaries | `src/iints/analysis/` |
| Pi / UNO Q / Jetson runtime | `src/iints/live_patient/`, `src/iints/jetson/` |
| local AI summaries and guardrails | `src/iints/ai/` |

## Read Next

- [Developer Portal](DEVELOPER_PORTAL.md)
- [API Reference](API_REFERENCE.md)
- [Architecture Hardening Plan](ARCHITECTURE_HARDENING.md)
- [Architecture & Module Guide](COMPREHENSIVE_GUIDE.md)
- [Contribute Safely](CONTRIBUTING_SAFELY.md)
