# Troubleshooting

**This page is for:** when a command failed and you want the shortest path to a fix.

## First Recovery Commands

```bash
iints doctor --full --suggest
iints demo --dry-run
iints run --dry-run --preset baseline_t1d
```

## Common Problems

### `Algorithm file not found`

What it usually means:
- the path is wrong
- you are not in the project folder you expected

Good fixes:

```bash
pwd
ls algorithms
iints demo
```

If the CLI shows `Did you mean ...`, use that path directly.

### `Scenario file not found`

Good fixes:

```bash
ls scenarios
iints scenarios generate --output-path scenarios/generated_scenario.json
```

Or skip the scenario file and use a preset:

```bash
iints run --preset baseline_t1d
```

### `Patient config file not found`

Good fixes:

```bash
iints profiles create --name my_profile
ls patient_profiles
```

Or use a built-in preset:

```bash
iints run --preset baseline_t1d
```

### `No such command ...`

This usually means the installed CLI is older than the docs or repo checkout.

Good fixes:

```bash
python -m pip install -U pip
python -m pip install -U -e ".[full,mdmp]"
hash -r
iints --help
```

### `demo-booth`, AI, or reporting commands are missing

Install the reporting stack:

```bash
python -m pip install -U "iints-sdk-python35[reports]"
```

Or everything:

```bash
python -m pip install -U "iints-sdk-python35[full]"
```

### Public dataset fetch says verification failed

That is often expected for public packs without a pinned hash yet.

Use:

```bash
iints data fetch aide_t1d --no-verify --output-dir data_packs/public/aide_t1d
```

Only do this when you trust the source and understand that verification is relaxed.

### Raspberry Pi / Maker Faire setup is confusing

Use these in order:
- [Maker Faire Pi Mode](MAKERFAIRE_PI.md)
- [Maker Faire Pi Checklist](MAKERFAIRE_PI_CHECKLIST.md)
- [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md)

## Before Filing A Bug

Collect these:

```bash
iints doctor --full --suggest
iints --help
python -V
```

And include:
- the exact command you ran
- the full error text
- your OS and Python version
