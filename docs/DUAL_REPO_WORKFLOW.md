# Dual Repo Workflow (IINTS-SDK + MDMP)

This project now runs with two repositories:

- SDK: `python35/IINTS-SDK`
- MDMP: `python35/MDMP`

## Goal
Keep simulation/runtime concerns in SDK and protocol/provenance concerns in MDMP, while shipping both in lockstep.

## CI sync gate

SDK now contains a dedicated workflow:

- `.github/workflows/mdmp-sync.yml`

It installs the standalone MDMP package from `python35/MDMP` and runs:

- `tools/ci/check_mdmp_sync.py`

The gate fails if SDK MDMP backend behavior diverges from standalone `mdmp_core`
(grade order, validation result surface, fingerprints, or dashboard generation contract).

## Local layout

```text
IINTS-SDK-main/
  local/mdmp-private/   # standalone MDMP repo clone/worktree
```

## Daily workflow

1. Check both repos:

```bash
tools/local/dual_repo_status.sh
```

2. Implement SDK changes and/or MDMP changes.
3. Run tests in both repos.
4. Commit/push both:

```bash
tools/local/dual_repo_commit_push.sh \
  --sdk-msg "SDK: <change summary>" \
  --mdmp-msg "MDMP: <change summary>"
```

## MDMP backend in SDK

SDK MDMP commands can use either backend:

- `iints` (default): built-in SDK MDMP runtime
- `mdmp_core` (optional): standalone MDMP package

Set backend explicitly:

```bash
export IINTS_MDMP_BACKEND=mdmp_core
```

Then run:

```bash
iints mdmp validate mdmp_contract.yaml data/my_cgm.csv --output-json results/mdmp_report.json
```

The command summary prints the active backend so provenance is explicit.

For stale lineage handling with standalone MDMP:

```bash
mdmp fingerprint-record data/my_cgm.csv --output-json results/fingerprint.json --expires-days 365
mdmp fingerprint-check results/fingerprint.json data/my_cgm.csv
mdmp lineage-card-refresh results/mdmp_model_card.yaml
mdmp registry init --registry registry/mdmp_registry.json
mdmp registry push --registry registry/mdmp_registry.json --report results/mdmp_report.json
```
