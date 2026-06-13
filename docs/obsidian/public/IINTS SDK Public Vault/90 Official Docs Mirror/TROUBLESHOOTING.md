# Troubleshooting

Use this page when a command failed and you want the shortest safe path back to a working state.

**Read before:** [Installation](INSTALLATION.md) if you are not sure the SDK is installed correctly.

**Read next:** [Command Reference](COMMAND_REFERENCE.md) when you need to confirm exact command syntax.

## First Recovery Pass

Run these before changing anything else:

```bash
iints doctor --full --suggest
iints demo --dry-run
iints run --dry-run --preset baseline_t1d
```

If `pip` only offers old `0.1.x` releases, check the active interpreter first:

```bash
python --version
python -m pip install -U "iints-sdk-python35[full,mdmp]"
```

Current SDK releases require Python `>=3.10`. A Python `3.8` virtual environment will keep resolving legacy releases and will not contain newer command groups such as `jetson`.

If one of these commands already tells you what is wrong, fix that first. It is usually faster than guessing.

## Common Failures

### `Algorithm file not found`

Usually means the path is wrong or you are not inside the project folder you expected.

```bash
pwd
ls algorithms
iints demo
```

If the CLI prints `Did you mean ...`, use that path directly.

### `Scenario file not found`

```bash
ls scenarios
iints scenarios generate --output-path scenarios/generated_scenario.json
```

Or skip the file and use a preset:

```bash
iints run --preset baseline_t1d
```

### `Patient config file not found`

```bash
iints profiles create --name my_profile
ls patient_profiles
```

Or use a built-in preset:

```bash
iints run --preset baseline_t1d
```

### `No such command ...`

This normally means the installed CLI is older than the docs or older than your repository checkout.

```bash
python -m pip install -U pip
python -m pip install -U -e ".[full,mdmp]"
hash -r
iints --help
```

### AI, reporting, or booth commands are missing

Install the reporting extras:

```bash
python -m pip install -U "iints-sdk-python35[reports]"
```

Or install the full profile:

```bash
python -m pip install -U "iints-sdk-python35[full]"
```

### Dataset fetch verification failed

Some public sources still do not publish a pinned SHA-256. If you trust the source and deliberately accept the weaker verification mode:

```bash
iints data fetch aide_t1d --no-verify --output-dir data_packs/public/aide_t1d
```

Treat `--no-verify` as an explicit trust decision, not as a normal success path.

### Raspberry Pi or Maker Faire setup feels unclear

Read these in order:

1. [Hardware Hub](HARDWARE.md)
2. [Maker Faire Pi Mode](MAKERFAIRE_PI.md)
3. [Maker Faire Pi Checklist](MAKERFAIRE_PI_CHECKLIST.md)
4. [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md)

## Before Filing A Bug

Collect:

```bash
iints doctor --full --suggest
iints --help
python -V
```

Then include:

- the command you ran
- the exact error output
- whether you installed from PyPI or from source
- the working directory where you ran it

## Where To Go Next

| If you want to... | Continue with |
| --- | --- |
| reinstall cleanly | [Installation](INSTALLATION.md) |
| finish the first workflow | [Getting Started](GETTING_STARTED.md) |
| confirm a command shape | [Command Reference](COMMAND_REFERENCE.md) |
| choose a docs route again | [Choose Your Path](USER_GUIDE_MAP.md) |
