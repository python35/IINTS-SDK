# Workflow Hub

Use this page after your first successful run, when the question changes from “does the SDK work?” to “what do I want to do with it?”

## Choose The Job

| If you need to... | Start with | Typical output |
| --- | --- | --- |
| compare algorithms in a reproducible study | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) | protocol bundle, study runs, comparisons |
| summarize a completed batch of runs | [Study Analysis](STUDY_ANALYSIS.md) | aggregate metrics, evidence table, poster-ready figures |
| prove whether input data is trustworthy | [MDMP Quickstart](MDMP_QUICKSTART.md) | certification JSON, dashboard, trust grade |
| explain validated outputs locally | [AI Assistant](AI_ASSISTANT.md) | guarded summaries and review notes |
| present the SDK live | [Booth Demo Guide](BOOTH_DEMO.md) | showable code, poster, talk track |

## Recommended Routes

### Research

```text
Getting Started
  -> Scientific Workflow
  -> Study Analysis
  -> Evidence Base
```

### Data Quality

```text
MDMP Quickstart
  -> MDMP Full Guide
  -> Evidence Base
```

### Presentation

```text
Booth Demo Guide
  -> iints demo-live
  -> poster + talk track + proof bundle
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
