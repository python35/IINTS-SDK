# Intended Use & Claims Control

This document controls what IINTS-AF may and may not claim.

## Approved Intended Use

IINTS-AF is an open-source research and education SDK for pre-clinical simulation of diabetes technology concepts. It can be used to:

- simulate virtual glucose-insulin scenarios;
- test candidate algorithms in a sandbox;
- generate reproducible research artifacts;
- explore physiology-inspired models and glucose forecasting;
- inspect data quality using MDMP-style contracts;
- run local AI review of simulation outputs for research discussion.

## Excluded Intended Use

IINTS-AF must not be used for:

- diagnosis;
- treatment decisions;
- insulin or glucagon dosing for a person;
- real-time patient care;
- clinical monitoring of a patient;
- autonomous or semi-autonomous therapy;
- claims of CE marking, MDR conformity, clinical safety, or clinical performance.

## Approved Public Claim Pattern

Use wording like:

> IINTS-AF is a research-only educational simulator for diabetes technology experimentation. It helps generate reproducible simulations, reports, and evidence artifacts. It is not a medical device and must not be used for treatment decisions.

## Forbidden Claim Patterns

Do not use wording like:

- "clinically validated insulin dosing";
- "safe for patient use";
- "approved medical AI";
- "CE marked";
- "doctor replacement";
- "predicts patient glucose for treatment";
- "connect this to a real pump for therapy".

## Review Triggers

Stop and perform a formal regulatory review before any change that:

- connects SDK output to real insulin or glucagon delivery;
- markets the SDK for a clinical workflow;
- adds patient-specific recommendations;
- uses identifiable patient data outside local research processing;
- enables remote cloud processing of health data;
- adds automatic app updates without signed release verification;
- adds broad Tauri filesystem, shell, network, or updater permissions.

## Maintainer Checklist

Before release, confirm:

- README contains the research-only boundary.
- Desktop app header contains the medical-device boundary.
- Reports and AI outputs do not produce treatment advice.
- Release notes do not claim regulatory approval.
- Screenshots and demo scripts avoid suggesting clinical deployment.
