# Updating The SDK

Use this guide when you already have IINTS installed and want to move to the latest release cleanly.

If you are not sure which folder you should be in, read [Installation And Paths](INSTALLATION.md) first.

This page assumes the full workstation install. If you are updating a Raspberry Pi or UNO Q runtime, substitute `iints-sdk-python35[edge,mdmp]` instead.

## Fast path: `iints update`

If IINTS is already installed in the environment you are using, run:

```bash
source .venv/bin/activate
iints update
hash -r
iints --version
```

By default `iints update` uses `--source auto`: it tries PyPI first and falls back to GitHub `main` if the release has not propagated yet.

For the newest GitHub version directly:

```bash
iints update --source github --yes
hash -r
iints --version
```

Use a dry run when you want to see exactly what will happen:

```bash
iints update --dry-run
```

If your terminal still sees an older command after updating, repair the environment:

```bash
iints update --repair --force-reinstall --yes
hash -r
iints --version
```

If pip seems to reuse stale wheels/caches:

```bash
iints update --no-cache-dir --force-reinstall --yes
```

The updater uses the same Python executable that launched `iints`, so it updates the virtual environment you are currently inside.

## Clean removal: `iints delete`

Use this when you want to remove IINTS from the active Python environment and clean user-level IINTS files. The default command uninstalls the SDK package itself from the Python environment that launched `iints`.

```bash
iints delete --dry-run
iints delete --yes
```

Default behavior:

- uninstalls `iints` and `iints-sdk-python35` from the active Python environment
- removes user-level IINTS config, plugin, and cache folders such as `~/.iints`
- refuses dangerous targets such as home, root, or the current working directory
- does not remove source checkouts, private datasets, or generated `results/` folders by default

To remove **everything IINTS-owned that the command can safely identify**, including generated output folders in the current directory and a detected local `IINTS-SDK` source checkout:

```bash
iints delete --everything --dry-run
iints delete --everything --yes
```

This is the closest command to "remove the whole SDK from this machine." It still does not guess private datasets, external-drive research archives, or unrelated virtual environments.

To also remove generated IINTS output folders in the current project directory:

```bash
iints delete --local-outputs --yes
```

To explicitly remove a detected source checkout without using `--everything`:

```bash
cd /path/to/IINTS-SDK
iints delete --source-checkout --yes
```

To remove one extra IINTS-owned folder:

```bash
iints delete --no-packages --no-user-data --path results/old_iints_run --yes
```

Always run `--dry-run` first when cleaning research machines or external drives.

## Manual upgrade path

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
python -m pip install -U "iints-sdk-python35[full,mdmp]==1.5.17"
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
