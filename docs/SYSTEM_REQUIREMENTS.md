# System Requirements

This page separates **declared compatibility** from **capacity planning**. The first is enforced by package metadata and release builds. The second is practical guidance for choosing hardware; it is not a claim that every workload was benchmarked on exactly that configuration.

## Supported Software

| Component | Requirement | Notes |
| --- | --- | --- |
| Python SDK | Python 3.10 through 3.14 | Python 3.15 is not supported by the current package metadata. |
| Operating system | 64-bit Windows, macOS, or Linux | The Python package is cross-platform; availability of optional scientific wheels can differ by OS and CPU architecture. |
| Linux desktop beta | x86_64 Linux | The current AppImage is built on Ubuntu 22.04 and is not an ARM AppImage. |
| Windows desktop beta | Windows x64 | Distributed as an `.exe` installer. |
| macOS desktop beta | macOS build from the current GitHub `macos-latest` runner | The beta is not advertised as a universal binary; inspect the release notes when CPU architecture matters. |
| Internet access | Required for initial installation | Also required for public biological APIs, Hugging Face downloads, Ollama model pulls, and update checks. Simulations can run offline after dependencies and inputs are present. |
| User storage | Writable home and output directories | The SDK writes environments, caches, run bundles, reports, models, and datasets outside the source tree when configured to do so. |

No GPU is required for ordinary simulation, validation, MDMP certification, or report generation.

## Capacity Planning

These figures are conservative starting points. Study duration, cohort size, report resolution, dataset size, model size, and parallelism can increase demand substantially.

| Workload | Practical minimum | Recommended | Free storage to reserve |
| --- | --- | --- | --- |
| CLI, small simulations, validation | 2 CPU cores, 4 GB RAM | 4 CPU cores, 8 GB RAM | 2-5 GB |
| Reports and desktop workbench | 4 CPU cores, 8 GB RAM | 4-8 CPU cores, 16 GB RAM | 5-10 GB |
| Local Ollama explanation with a small model | 4 CPU cores, 8 GB RAM | 8 CPU cores or supported GPU, 16 GB RAM | 10-30 GB |
| Glucose-model training and large study matrices | 8 CPU cores, 16 GB RAM | 8+ CPU cores, 32 GB RAM, optional supported GPU | 25-100+ GB |
| Long-running Jetson or edge research | board-specific | active cooling and monitored storage | depends on checkpoints and telemetry |

Storage estimates exclude private datasets, Ollama models, model checkpoints, exported figures, and accumulated `results/` folders. Those artifacts usually dominate long-term storage.

## Installation Profiles

| Profile | Python package | Main use |
| --- | --- | --- |
| Standard | `iints-sdk-python35[full,mdmp]` | simulation, reports, imports, and certification |
| Research | `iints-sdk-python35[full,mdmp,research]` | Torch, ONNX, Parquet/HDF5, and interactive research plots |
| Edge | `iints-sdk-python35[edge,mdmp]` | serial bridges and supported hardware workflows |
| Maintained desktop engine | `iints-sdk-python35[tauri-engine]` | Tauri/Python bridge, reports, interactive plots, SBML, and FMI support |
| Legacy Qt/development bundle | `iints-sdk-python35[desktop-all]` | compatibility testing for the former PySide interface plus ML/packaging dependencies |

The `research` and legacy `desktop-all` profiles are much larger because they include machine-learning or GUI-packaging libraries. The maintained app uses the smaller `tauri-engine` profile; install a training profile only on machines that actually train models.

## External Tools

The following tools are optional and are not silently installed with the normal SDK package:

- Ollama and its model files
- COPASI
- OpenCOR
- external FMUs used through FMPy
- private or licensed datasets
- pretrained Hugging Face model weights

The desktop workbench detects missing optional tools and should keep unrelated workflows available.

## Desktop-Specific Notes

The Linux AppImage bundles the application shell, but the scientific Python engine remains a private environment under `~/.iints-af/python-engine`. On Omarchy, the supported installer prepares both layers and installs `fuse2` for normal AppImage startup.

Tauri documents that AppImage compatibility depends on the GNU C Library baseline used for the build. The IINTS-AF Linux beta is built on Ubuntu 22.04, while current rolling-release Omarchy systems provide a newer userspace. See the [Tauri AppImage guide](https://v2.tauri.app/distribute/appimage/) for the underlying compatibility model.

## Check A Machine

Before installing:

=== "Linux or macOS"

    ```bash
    uname -m
    python3 --version
    python3 -c "import platform; print(platform.platform())"
    df -h "$HOME"
    ```

=== "Windows PowerShell"

    ```powershell
    py --version
    Get-CimInstance Win32_OperatingSystem | Select-Object OSArchitecture, TotalVisibleMemorySize
    Get-PSDrive -PSProvider FileSystem
    ```

After installing:

```bash
iints --version
iints version --refresh
iints doctor --smoke-run --suggest
```

Use `iints doctor --full --suggest` before a long study, edge deployment, or AI workflow.

## Platform Guides

- [Installation](INSTALLATION.md)
- [Omarchy Linux](OMARCHY_INSTALL.md)
- [Desktop App Installation](APP_INSTALL.md)
- [Hardware Hub](HARDWARE.md)
- [Troubleshooting](TROUBLESHOOTING.md)

## Basis For These Requirements

- Python and dependency compatibility come from `pyproject.toml` in the released SDK.
- Desktop architectures and build baselines come from `.github/workflows/tauri-desktop-beta.yml`.
- Omarchy package and runtime handling follows the [Omarchy manual](https://omarchy.org/manual/), its [development tools guide](https://omarchy.org/manual/development-tools/), and its [update procedure](https://omarchy.org/manual/updates/).
- Linux desktop build dependencies follow the [official Tauri prerequisites](https://v2.tauri.app/start/prerequisites/).

Last reviewed: 2026-08-22.
