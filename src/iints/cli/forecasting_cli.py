"""Typer CLI subcommands for `iints forecasting` (Phases 1–5: Data, Baselines, Neural, Robustness & Jetson Edge)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
import typer

from iints.forecasting import (
    ForecastingModelRegistry,
    NeuralTrainingConfig,
    WindowSpec,
    benchmark_edge_inference,
    build_partitioned_windows,
    export_model_for_edge,
    load_carelink_benchmark_dataset,
    load_ohiot1dm_dataset,
    load_virtual_benchmark_dataset,
    run_cgm_robustness_benchmark,
    run_phase2_baseline_benchmark,
    train_and_compare_all_models,
)

app = typer.Typer(
    help="IINTS Neural Forecasting v1 — OhioT1DM/Virtual pipelines, GRU/TCN training, robustness & Jetson Edge."
)
console = Console()


def _load_dataset_helper(dataset: str, data_path: Optional[Path], virtual_subjects: int, seed: int):
    if dataset == "ohiot1dm":
        if data_path is None:
            raise typer.BadParameter("--data-path pointing to OhioT1DM XML directory is required.")
        return load_ohiot1dm_dataset(data_path)
    if dataset == "carelink":
        if data_path is None:
            raise typer.BadParameter("--data-path pointing to CareLink CSV is required.")
        return load_carelink_benchmark_dataset(data_path)
    if dataset == "iints_virtual":
        return load_virtual_benchmark_dataset(n_subjects=virtual_subjects, seed=seed)
    raise typer.BadParameter(f"Unsupported dataset '{dataset}'.")


@app.command("validate-data")
def cmd_validate_data(
    dataset: str = typer.Option("ohiot1dm", "--dataset", "-d"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", "-p"),
    virtual_subjects: int = typer.Option(12, "--virtual-subjects"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Validate dataset schema, timestamps, duplicates, out-of-bounds values, and gaps."""
    _, report = _load_dataset_helper(dataset, data_path, virtual_subjects, seed)
    table = Table(title=f"Data Quality Audit — {report.dataset_id} (synthetic={report.is_synthetic})")
    table.add_column("Subject", style="cyan")
    table.add_column("Split")
    table.add_column("Raw", justify="right")
    table.add_column("Valid", justify="right")
    table.add_column("Dups", justify="right")
    table.add_column("OOB", justify="right")
    table.add_column("Gaps>6.5m", justify="right")
    table.add_column("MaxGap(m)", justify="right")
    for s in report.subjects:
        table.add_row(
            s.subject_id,
            s.source_split,
            str(s.raw_rows),
            str(s.valid_rows),
            str(s.duplicate_timestamps),
            str(s.out_of_bounds_rows),
            str(s.gaps_over_threshold),
            f"{s.max_gap_minutes:.1f}",
        )
    console.print(table)
    console.print(f"[bold]Provenance:[/bold] {report.source_provenance}")


@app.command("evaluate-baselines")
def cmd_evaluate_baselines(
    dataset: str = typer.Option("ohiot1dm", "--dataset", "-d"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", "-p"),
    out_dir: Path = typer.Option(Path("results/neural_forecasting_phase2"), "--out-dir", "-o"),
    virtual_subjects: int = typer.Option(12, "--virtual-subjects"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Run Phase 2 Persistence + Ridge baseline evaluation with embargoed splits & registry."""
    res = run_phase2_baseline_benchmark(
        dataset=dataset,
        data_path=data_path,
        output_dir=out_dir,
        virtual_subjects=virtual_subjects,
        seed=seed,
    )
    console.print(res["pooled_summary"].to_string(index=False))
    console.print(f"\n[bold green]Baseline report written to:[/bold green] {res['report_path'].resolve()}")


@app.command("train-all")
def cmd_train_all(
    dataset: str = typer.Option("iints_virtual", "--dataset", "-d"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", "-p"),
    out_dir: Path = typer.Option(Path("results/neural_forecasting_v1"), "--out-dir", "-o"),
    seeds: str = typer.Option("11,23,42,67,89", "--seeds", help="Comma-separated neural training seeds."),
    epochs: int = typer.Option(45, "--epochs"),
    virtual_subjects: int = typer.Option(12, "--virtual-subjects"),
) -> None:
    """Train Persistence, Ridge, 5-seed GRU, and 5-seed TCN under identical embargoed windows."""
    seed_list = tuple(int(s.strip()) for s in seeds.split(",") if s.strip())
    canonical, quality = _load_dataset_helper(dataset, data_path, virtual_subjects, seed_list[0])
    part = build_partitioned_windows(canonical, WindowSpec())
    cfg = NeuralTrainingConfig(seeds=seed_list, max_epochs=epochs)
    res = train_and_compare_all_models(part, quality, out_dir, train_cfg=cfg)
    console.print(res["summary_table"].to_string(index=False))
    console.print(f"\n[bold green]Models & seed sensitivity saved in:[/bold green] {out_dir.resolve()}")


@app.command("robustness")
def cmd_robustness(
    dataset: str = typer.Option("iints_virtual", "--dataset", "-d"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", "-p"),
    registry_dir: Path = typer.Option(Path("results/neural_forecasting_v1/model_registry"), "--registry-dir"),
    out_dir: Path = typer.Option(Path("results/neural_forecasting_v1/robustness"), "--out-dir", "-o"),
    virtual_subjects: int = typer.Option(12, "--virtual-subjects"),
) -> None:
    """Run Phase 4 CGM Robustness Benchmark (Clean, Dropout+LOCF, Noise, Delay) on registered models."""
    canonical, quality = _load_dataset_helper(dataset, data_path, virtual_subjects, 42)
    part = build_partitioned_windows(canonical, WindowSpec())
    reg = ForecastingModelRegistry(registry_dir)
    models = {e.model_id: reg.load_model(e.model_id, e.version) for e in reg.list_entries()}
    if not models:
        raise typer.BadParameter(f"No registered models found in {registry_dir}. Run `train-all` first.")
    rob = run_cgm_robustness_benchmark(
        part.test,
        models,
        is_synthetic=quality.is_synthetic,
        dataset_id=quality.dataset_id,
        output_dir=out_dir,
    )
    console.print(rob["paired_degradation"].to_string(index=False))


@app.command("export-edge")
def cmd_export_edge(
    model_id: str = typer.Argument(..., help="Registered model ID (e.g. iints-gru-glucose-v1 or iints-tcn-glucose-v1)."),
    registry_dir: Path = typer.Option(Path("results/neural_forecasting_v1/model_registry"), "--registry-dir"),
    out_dir: Path = typer.Option(Path("results/neural_forecasting_v1/edge_exports"), "--out-dir", "-o"),
) -> None:
    """Export a registered model to `.iints_edge.json` (+ `.onnx` when PyTorch/ONNX available) & check parity."""
    reg = ForecastingModelRegistry(registry_dir)
    model = reg.load_model(model_id)
    res = export_model_for_edge(model, out_dir)
    console.print(
        f"[bold green]Exported:[/bold green] {res['edge_bundle_path']} | runtime={res['active_runtime']} | "
        f"max_diff={res['parity']['max_abs_diff_mgdl']:.2e} mg/dL (passed={res['parity']['passed']})"
    )


@app.command("edge-benchmark")
def cmd_edge_benchmark(
    bundle_path: Path = typer.Argument(..., help="Path to `.iints_edge.json` bundle."),
    warmup: int = typer.Option(50, "--warmup"),
    runs: int = typer.Option(500, "--runs"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
) -> None:
    """Benchmark batch-size-1 latency, throughput, Peak RAM & Jetson `tegrastats` telemetry."""
    rep = benchmark_edge_inference(bundle_path, n_warmup=warmup, n_runs=runs, output_json_path=out_json)
    console.print_json(data=rep)


@app.command("serve-jetson-worker")
def cmd_serve_jetson_worker(
    bundle_path: Path = typer.Argument(..., help="Path to `.iints_edge.json` bundle."),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8770, "--port"),
) -> None:
    """Launch the local NVIDIA Jetson Orin Nano HTTP inference + telemetry worker."""
    import uvicorn
    from iints.forecasting.deployment import create_jetson_worker_app

    console.print(f"[bold cyan]Starting IINTS Jetson Edge Worker on http://{host}:{port}[/bold cyan]")
    uvicorn.run(create_jetson_worker_app(bundle_path), host=host, port=port)


@app.command("registry-list")
def cmd_registry_list(
    registry_dir: Path = typer.Option(Path("results/neural_forecasting_v1/model_registry"), "--registry-dir"),
) -> None:
    """List registered forecasting models and checkpoint SHA-256 checksums."""
    reg = ForecastingModelRegistry(registry_dir)
    entries = reg.list_entries()
    if not entries:
        console.print(f"[yellow]No models registered in {registry_dir}[/yellow]")
        return
    table = Table(title="IINTS Forecasting Model Registry")
    table.add_column("Model ID", style="cyan")
    table.add_column("Arch")
    table.add_column("Dataset")
    table.add_column("Synthetic")
    table.add_column("Span/Horiz")
    table.add_column("Val MAE")
    table.add_column("Checkpoint SHA-256")
    for e in entries:
        v_mae = e.validation_metrics.get("mae")
        table.add_row(
            e.model_id,
            e.architecture,
            e.dataset_id,
            str(e.is_synthetic),
            f"{e.elapsed_input_span_minutes:.0f}m -> {e.prediction_horizon_minutes}m",
            f"{v_mae:.2f}" if v_mae is not None else "—",
            e.checkpoint_sha256[:16] + "...",
        )
    console.print(table)
