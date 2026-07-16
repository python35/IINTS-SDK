# Security Policy

IINTS-AF is research and education software. It is not a medical device and must not be used for treatment decisions.

## Reporting a Vulnerability

Please report security issues privately when possible by contacting the maintainer or opening a GitHub security advisory if available. Do not include private patient, CGM, pump, genetic, or clinical data in public issues.

If a private channel is not available, open a minimal public issue that describes the affected component without exploit details or sensitive data.

## What to Report

Please report:

- arbitrary command execution;
- unsafe desktop app permissions;
- dependency or packaging vulnerabilities;
- path traversal or unsafe file handling;
- accidental telemetry or upload behavior;
- exposure of private health data;
- flaws that could make research output look like treatment advice.

## Supported Scope

Security fixes are prioritized for the latest release and the current `main` branch. Older research releases are best-effort only.

## Data Handling

Never attach real patient data to a vulnerability report. Use synthetic examples or small mock CSVs.

## Maintainer Response Goals

- Acknowledge serious reports as soon as practical.
- Triage severity and affected versions.
- Patch and test before public disclosure when feasible.
- Credit reporters if they want credit.
