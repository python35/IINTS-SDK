# Updating The SDK

Use this guide when you already have IINTS installed and want to move to the latest release cleanly.

## The safest upgrade path

Always upgrade inside the virtual environment you actually use for IINTS:

```bash
source .venv/bin/activate
python --version
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[mdmp]==1.3.0"
hash -r
```

Then confirm the installed version:

```bash
python -c "import iints; print(iints.__version__)"
iints --help
```

## If you installed from source

If you work from a local checkout, update the repo and reinstall editable mode:

```bash
git pull
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U -e ".[mdmp]"
hash -r
```

## If `iints ai` is missing after upgrading

The most common cause is a legacy `iints` package shadowing `iints-sdk-python35`.

Check the environment:

```bash
iints-sdk-doctor
```

If it reports a conflict, repair it with:

```bash
python -m pip uninstall -y iints iints-sdk-python35
python -m pip install -U "iints-sdk-python35[mdmp]==1.3.0"
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
python -m pip install -U "iints-sdk-python35[mdmp]==1.3.0"
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
- installed `iints-sdk-python35[mdmp]==1.3.0`
- ran `hash -r`
- confirmed `python -c "import iints; print(iints.__version__)"`
- ran `iints --help`

If all six are true, the SDK should be up to date on that machine.
