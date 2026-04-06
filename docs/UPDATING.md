# Updating The SDK

Use this guide when you already have IINTS installed and want to move to the latest release cleanly.

If you are not sure which folder you should be in, read [Installation And Paths](INSTALLATION.md) first.

This page assumes the full workstation install. If you are updating a Raspberry Pi or UNO Q runtime, substitute `iints-sdk-python35[edge,mdmp]` instead.

## The safest upgrade path

Always upgrade inside the virtual environment you actually use for IINTS:

```bash
source .venv/bin/activate
python --version
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[full,mdmp]"
hash -r
```

Then confirm the installed version:

```bash
python -c "import iints; print(iints.__version__)"
iints --help
```

If you want a reproducible environment for a paper, demo, or audit, you can still pin an exact version explicitly, for example:

```bash
python -m pip install -U "iints-sdk-python35[full,mdmp]==1.5.2"
```

## If you installed from source

If you work from a local checkout, update the repo and reinstall editable mode:

```bash
cd /path/to/IINTS-SDK
git pull
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U -e ".[full,mdmp]"
hash -r
```

If you see:

```text
ERROR: file:///home/your-user does not appear to be a Python project
```

then you ran `pip install -e ".[full,mdmp]"` from the wrong folder. Move into the SDK repository root first, where `pyproject.toml` lives.

The repository root is the folder containing:

- `pyproject.toml`
- `src/`
- `scripts/`
- `examples/`

## If `iints ai` is missing after upgrading

The most common cause is a legacy `iints` package shadowing `iints-sdk-python35`.

Check the environment:

```bash
iints-sdk-doctor
```

If it reports a conflict, repair it with:

```bash
python -m pip uninstall -y iints iints-sdk-python35
python -m pip install -U "iints-sdk-python35[full,mdmp]"
hash -r
```

## If you are on the wrong Python version

Current releases require Python `>=3.10`.

Check it:

```bash
python --version
```

If you are on Python `3.8` or `3.9`, create a fresh environment with Python `3.10` or newer:

```bash
python3.11 -m venv .venv-iints
source .venv-iints/bin/activate
python -m pip install -U "iints-sdk-python35[full,mdmp]"
```

## Recommended post-upgrade checks

For a normal SDK environment:

```bash
iints doctor --smoke-run
```

For local AI features:

```bash
iints ai models
iints ai local-check --model ministral-3:3b
```

For the booth demo:

```bash
iints demo-booth --output-dir results/booth_demo
```

## Quick checklist

- activated the correct virtual environment
- upgraded `pip`
- installed the latest `iints-sdk-python35[full,mdmp]` release
- ran `hash -r`
- confirmed `python -c "import iints; print(iints.__version__)"`
- ran `iints --help`

If all six are true, the SDK should be up to date on that machine.
