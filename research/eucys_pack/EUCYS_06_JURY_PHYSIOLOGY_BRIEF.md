# IINTS-AF Jury Physiology Brief

**Purpose:** A short, judge-friendly answer to the question: “What is the SDK actually simulating?”

## The 30-Second Answer

IINTS-AF simulates **virtual-patient glucose physiology**, not just lines on a graph. It represents meals, insulin action, insulin-on-board, carbohydrates-on-board, exercise, sensor imperfections, and safety supervision as separate pieces so they can be tested and explained.

## The Three Numbers Judges Should Remember

| Number | Meaning |
|---|---|
| `70-180 mg/dL` | standard glucose target range used for Time in Range |
| `<70 mg/dL` | low-glucose exposure |
| `<54 mg/dL` | clinically more serious hypoglycemia threshold |

These values make a simulated run interpretable, but they do not turn a simulator into a clinical device.

## The Three Layers Judges Should Not Confuse

| Layer | Example |
|---|---|
| physiology | true glucose rises after a meal |
| measurement | the CGM signal is noisy or delayed |
| protection | the supervisor blocks an excessive insulin request |

This is one of the SDK’s main strengths: it does not hide those ideas inside one black-box score.

## One Meaningful Example

| Event | Example |
|---|---|
| breakfast | 48 g carbs |
| lunch | 62 g carbs |
| exercise | added glucose-lowering pressure |
| dinner | 74 g carbs |

A believable glucose graph should react to events like these. A perfectly smooth curve can be easier to code and less realistic scientifically.

## Why Data Quality Matters

Bad CGM data can quietly mislead an algorithm. Missing insulin, duplicated timestamps, impossible glucose jumps, and sensor artifacts can make a model look better or worse for the wrong reason. That is why IINTS-AF includes MDMP data checks before evidence is trusted.

## What The SDK Does Not Claim

- it is not a medical device
- it does not dose real patients
- it is not a personalized digital twin
- simulation does not replace clinical evidence

## Best One-Sentence Claim

**IINTS-AF makes insulin-algorithm research more reproducible by keeping physiology, data quality, and safety behavior visible instead of hiding them inside one polished demo graph.**
