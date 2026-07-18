# First Run

Use this page to verify the SDK and create one small, deterministic simulation.

**Time:** about five minutes after Python is installed.

**Result:** a folder containing a time series, report, metadata, and reproducibility information.

## 1. Create An Environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2. Install IINTS-AF

```bash
python -m pip install --upgrade pip
python -m pip install "iints-sdk-python35[full,mdmp]"
```

## 3. Verify The Installation

```bash
iints --version
iints doctor --smoke-run
```

Resolve blocking errors before continuing. Warnings about optional hardware or AI components are expected when those components are not installed.

## 4. Run The Demo

```bash
iints demo quick --output-dir results/first_run
```

The command uses a bundled patient, scenario, clinical baseline algorithm, and deterministic seed. It does not connect to a person or medical device.

## 5. Inspect The Result

List the generated files:

```bash
ls results/first_run
```

On Windows:

```powershell
Get-ChildItem results\first_run
```

You should see at least:

- `results.csv`
- `report.pdf`
- `run_manifest.json`

The command may also create metadata, an audit trail, an evidence bundle, and an editable example algorithm.

Open `report.pdf`, but also inspect `results.csv` and the metadata. A graph is a summary, not the complete evidence.

## 6. Confirm Reproducibility

Run the same preset with an explicit seed in a second folder:

```bash
iints demo quick --seed 42 --output-dir results/repeat_run
```

The run metadata should record the same controlled inputs. See [Understand A Run](RUN_OUTPUTS.md) for the correct review order.

## If It Fails

```bash
iints doctor --full --suggest
iints demo quick --dry-run --output-dir results/first_run
```

Then use [Troubleshooting](TROUBLESHOOTING.md). Include the command, SDK version, Python version, operating system, and complete error when reporting a problem.

## Next Step

Read [Core Concepts](CORE_CONCEPTS.md), then complete the [Complete First Workflow](GETTING_STARTED.md) with your own project folder.
