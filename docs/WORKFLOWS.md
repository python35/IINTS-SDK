# Workflow Hub

This is the central page for people who already have the SDK running and now need to do real work with it.

## Choose A Workflow

| Workflow | Use it when you need to... | Start with |
| --- | --- | --- |
| Research study | compare algorithms, generate benchmark evidence, or prepare a study bundle | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) |
| Study review | inspect completed runs, compare arms, or summarize findings | [Study Analysis](STUDY_ANALYSIS.md) |
| Data quality | certify a dataset, inspect realism, or defend a data-quality claim | [MDMP Quickstart](MDMP_QUICKSTART.md) |
| Local AI review | prepare explanations, reviews, and reports from validated outputs | [AI Assistant](AI_ASSISTANT.md) |
| Presentation | show code plus results during a meeting or booth demo | [Booth Demo Guide](BOOTH_DEMO.md) |

## Recommended Research Route

```text
Getting Started
  -> Scientific Workflow
  -> Study Analysis
  -> Evidence Base
```

## Recommended Data Route

```text
MDMP Quickstart
  -> MDMP Guide
  -> Evidence Base
```

## Recommended Presentation Route

```text
Booth Demo Guide
  -> iints demo-live
  -> poster + run bundle + talk track
```

## Common Commands

```bash
iints study-ready --algo algorithms/example_algorithm.py --output-dir results/study_ready
iints data certify contracts/clinical_mdmp_contract.yaml data/my_trace.csv --output-json results/certification.json
iints ai report results/<run_id>
iints demo-live --output-dir results/live_demo
```

## Read Next

- [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)
- [Study Analysis](STUDY_ANALYSIS.md)
- [MDMP Quickstart](MDMP_QUICKSTART.md)
- [AI Assistant](AI_ASSISTANT.md)
- [Booth Demo Guide](BOOTH_DEMO.md)
