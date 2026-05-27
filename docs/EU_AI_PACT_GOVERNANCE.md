# EU AI Pact Readiness

IINTS includes a research-only MDMP governance checklist inspired by the EU AI Pact and high-risk AI Act readiness themes.

It is **not** legal advice, not a conformity assessment, and not medical-device certification. It is a stricter internal checklist so IINTS evidence bundles do not overclaim.

## Core AI Pact Themes

The readiness review checks that a project records evidence for:

- AI governance strategy;
- high-risk system mapping;
- AI literacy.

## High-Risk Readiness Themes

For stricter research bundles, the review also checks:

- risk management;
- data governance and provenance;
- technical documentation;
- logging and traceability;
- human oversight;
- transparency and limitations;
- accuracy, robustness, and cybersecurity;
- post-deployment monitoring.

## Example

```python
from iints.mdmp.eu_ai_pact import review_eu_ai_pact_readiness

result = review_eu_ai_pact_readiness({
    "mdmp_grade": "research_grade",
    "compliance_score": 0.99,
    "dataset_fingerprint_sha256": "...",
    "contract_fingerprint_sha256": "...",
    "row_count": 288,
    "governance": {
        "ai_governance_strategy": True,
        "high_risk_mapping": True,
        "ai_literacy": True,
        "risk_management": True,
        "data_governance_and_provenance": True,
        "technical_documentation": True,
        "logging_and_traceability": True,
        "human_oversight": True,
        "transparency_and_limitations": True,
        "accuracy_robustness_cybersecurity": True,
        "post_deployment_monitoring": True,
    },
})

print(result.status)
```

## Interpretation

- `research_ready`: the evidence bundle is complete enough for internal research review.
- `needs_review`: no hard blocker, but the bundle still needs explanation or cleanup.
- `blocked`: do not use the bundle for public AI claims until the missing controls are resolved.

## Official Sources

- European Commission: [AI Pact](https://digital-strategy.ec.europa.eu/en/policies/ai-pact)
- European Commission: [AI Act](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
- European Commission: [Navigating the AI Act](https://digital-strategy.ec.europa.eu/en/faqs/navigating-ai-act)
