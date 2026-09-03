use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

static WORKFLOW_JOBS: OnceLock<Mutex<HashMap<String, WorkflowJobRecord>>> = OnceLock::new();
static NEXT_WORKFLOW_JOB_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct WorkflowJobRecord {
    status: String,
    phase: String,
    progress_percent: f64,
    message: String,
    cancel_requested: bool,
    result: Option<Value>,
    error: Option<String>,
    started: Instant,
    progress_path: PathBuf,
    cancel_path: PathBuf,
}

fn workflow_jobs() -> &'static Mutex<HashMap<String, WorkflowJobRecord>> {
    WORKFLOW_JOBS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn workflow_job_payload(job_id: &str, record: &WorkflowJobRecord) -> Value {
    json!({
        "job_id": job_id,
        "status": record.status,
        "phase": record.phase,
        "progress_percent": record.progress_percent,
        "message": record.message,
        "cancel_requested": record.cancel_requested,
        "elapsed_seconds": record.started.elapsed().as_secs_f64(),
        "result": record.result,
        "error": record.error,
    })
}

#[derive(Debug, Clone)]
struct PythonCandidate {
    program: String,
    prefix_args: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MechanisticRunRequest {
    model: String,
    output_dir: String,
    start: Option<f64>,
    end: Option<f64>,
    points: Option<i64>,
    variables: Option<Vec<String>>,
    source_url: Option<String>,
    model_license: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CopasiRunRequest {
    model: String,
    output_dir: String,
    task: Option<String>,
    timeout_seconds: Option<i64>,
    allow_external_execution: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CellmlValidationRequest {
    model: String,
    output_dir: String,
    timeout_seconds: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FmiRunRequest {
    model: String,
    output_dir: String,
    start: Option<f64>,
    end: Option<f64>,
    output_interval: Option<f64>,
    variables: Option<Vec<String>>,
    timeout_seconds: Option<i64>,
    trust_native_code: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BindingQueryRequest {
    uniprot: String,
    output_dir: String,
    cutoff_nm: Option<i64>,
    max_records: Option<i64>,
    timeout_seconds: Option<i64>,
}

fn python_candidates() -> Vec<PythonCandidate> {
    let mut candidates = Vec::new();
    if let Ok(value) = env::var("IINTS_PYTHON") {
        if !value.trim().is_empty() {
            candidates.push(PythonCandidate {
                program: value,
                prefix_args: Vec::new(),
            });
        }
    }
    if let Some(path) = managed_python_engine_path() {
        if path.is_file() {
            candidates.push(PythonCandidate {
                program: path.to_string_lossy().into_owned(),
                prefix_args: Vec::new(),
            });
        }
    }
    if cfg!(windows) {
        candidates.push(PythonCandidate {
            program: "py".to_string(),
            prefix_args: vec!["-3".to_string()],
        });
        candidates.push(PythonCandidate {
            program: "python".to_string(),
            prefix_args: Vec::new(),
        });
    } else {
        if cfg!(target_os = "macos") {
            for path in [
                "/opt/homebrew/bin/python3",
                "/usr/local/bin/python3",
                "/opt/local/bin/python3",
                "/Library/Frameworks/Python.framework/Versions/Current/bin/python3",
                "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3",
                "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
                "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
                "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
                "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3",
            ] {
                if Path::new(path).is_file() {
                    candidates.push(PythonCandidate {
                        program: path.to_string(),
                        prefix_args: Vec::new(),
                    });
                }
            }
        }
        candidates.push(PythonCandidate {
            program: "python3".to_string(),
            prefix_args: Vec::new(),
        });
        candidates.push(PythonCandidate {
            program: "python".to_string(),
            prefix_args: Vec::new(),
        });
    }
    candidates
}

fn bridge_working_dir() -> Option<PathBuf> {
    let home = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"))?;
    let path = PathBuf::from(home);
    path.is_dir().then_some(path)
}

fn python_bridge_timeout(args: &[String]) -> Duration {
    let command = args.first().map(String::as_str).unwrap_or_default();
    let seconds = match command {
        "status"
        | "workflows"
        | "diagnostics"
        | "update-info"
        | "molecules"
        | "evidence-connectors"
        | "mechanistic-status"
        | "cross-scale-status"
        | "history"
        | "ai-check"
        | "ai-models" => 20,
        "preview" | "mechanistic-inspect" | "copasi-inspect" | "cellml-inspect" | "fmi-inspect" => {
            60
        }
        "molecule-pae" | "binding-query" | "cellml-validate" | "mdmp-certify"
        | "academic-bundle" => 300,
        "genomics-sim" | "tissue-stress" | "mechanistic-run" | "copasi-run" | "fmi-run" => 900,
        "ai-start" | "ai-ask" => 1_200,
        "run" => 1_800,
        _ => 120,
    };
    Duration::from_secs(seconds)
}

fn collect_pipe<T>(mut pipe: T) -> Result<Vec<u8>, String>
where
    T: Read,
{
    let mut bytes = Vec::new();
    pipe.read_to_end(&mut bytes)
        .map_err(|error| format!("Could not read Python bridge output: {error}"))?;
    Ok(bytes)
}

fn join_pipe_reader(
    handle: thread::JoinHandle<Result<Vec<u8>, String>>,
) -> Result<Vec<u8>, String> {
    handle
        .join()
        .map_err(|_| "Python bridge output reader stopped unexpectedly.".to_string())?
}

fn command_output_with_timeout(
    mut command: Command,
    timeout: Duration,
    label: &str,
) -> Result<Output, String> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("Could not start {label}: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| format!("Could not capture stdout from {label}."))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| format!("Could not capture stderr from {label}."))?;
    let stdout_reader = thread::spawn(move || collect_pipe(stdout));
    let stderr_reader = thread::spawn(move || collect_pipe(stderr));
    let deadline = Instant::now() + timeout;

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_pipe_reader(stdout_reader);
                let _ = join_pipe_reader(stderr_reader);
                return Err(format!(
                    "{label} timed out after {} seconds and was stopped.",
                    timeout.as_secs()
                ));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_pipe_reader(stdout_reader);
                let _ = join_pipe_reader(stderr_reader);
                return Err(format!("Could not monitor {label}: {error}"));
            }
        }
    };

    Ok(Output {
        status,
        stdout: join_pipe_reader(stdout_reader)?,
        stderr: join_pipe_reader(stderr_reader)?,
    })
}

fn run_python_bridge(args: &[String]) -> Result<Value, String> {
    let mut attempts = Vec::new();
    let timeout = python_bridge_timeout(args);
    for candidate in python_candidates() {
        let mut command = Command::new(&candidate.program);
        command.args(&candidate.prefix_args);
        command.args(["-m", "iints_desktop.tauri_bridge"]);
        command.args(args);
        command.env("PYTHONUTF8", "1");
        command.env("PYTHONSAFEPATH", "1");
        if let Some(working_dir) = bridge_working_dir() {
            // Avoid importing relative to an app bundle, mounted DMG, or removable
            // research volume. Installed packages remain available via site-packages.
            command.current_dir(working_dir);
        }

        let label = format!("Python bridge via {}", candidate.program);
        match command_output_with_timeout(command, timeout, &label) {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                if !output.status.success() && stdout.is_empty() {
                    attempts.push(format!(
                        "{} {:?}: exit {}; {}",
                        candidate.program, candidate.prefix_args, output.status, stderr
                    ));
                    continue;
                }
                let envelope: Value = serde_json::from_str(&stdout).map_err(|error| {
                    format!(
                        "Python bridge returned invalid JSON via {}: {error}. stdout={stdout:?} stderr={stderr:?}",
                        candidate.program
                    )
                })?;
                if envelope.get("ok").and_then(Value::as_bool) == Some(true) {
                    return Ok(envelope.get("data").cloned().unwrap_or_else(|| json!({})));
                }
                let message = envelope
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("Python bridge reported an error");
                let details = envelope
                    .get("details")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if !details.is_empty() {
                    eprintln!("IINTS Python bridge diagnostic:\n{details}");
                }
                return Err(message.to_string());
            }
            Err(error) => {
                if error.contains("timed out after") {
                    return Err(format!(
                        "The IINTS Python research engine did not respond in time. Open Settings and run 'Refresh versions'. If this repeats, use 'Install or update Python SDK' to repair the private engine.\n{error}"
                    ));
                }
                attempts.push(format!(
                    "{} {:?}: {error}",
                    candidate.program, candidate.prefix_args
                ));
            }
        }
    }
    Err(format!(
        "The IINTS Python research engine is not installed or could not be found. Open Settings and choose 'Install or update Python SDK'. The app will create a private engine under ~/.iints-af when Python 3.10-3.14 is available. Advanced users may instead set IINTS_PYTHON.\nAttempts:\n{}",
        attempts.join("\n")
    ))
}

async fn run_python_bridge_async(args: Vec<String>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || run_python_bridge(&args))
        .await
        .map_err(|error| format!("Python bridge task failed: {error}"))?
}

#[tauri::command]
async fn desktop_status() -> Result<Value, String> {
    run_python_bridge_async(vec!["status".to_string()]).await
}

#[tauri::command]
fn desktop_app_info() -> Value {
    json!({
        "app_version": env!("CARGO_PKG_VERSION"),
        "product_name": "IINTS-AF Research Workbench",
        "platform": env::consts::OS,
        "architecture": env::consts::ARCH,
        "release_url": "https://github.com/python35/IINTS-SDK/releases/tag/tauri-beta-latest",
        "guide_url": "https://python35.github.io/IINTS-SDK/RESEARCH_WORKBENCH_GUIDE/"
    })
}

#[tauri::command]
async fn list_workflows() -> Result<Value, String> {
    run_python_bridge_async(vec!["workflows".to_string()]).await
}

#[tauri::command]
async fn desktop_diagnostics() -> Result<Value, String> {
    run_python_bridge_async(vec!["diagnostics".to_string()]).await
}

#[tauri::command]
async fn desktop_update_info(refresh: Option<bool>) -> Result<Value, String> {
    let mut args = vec![
        "update-info".to_string(),
        "--app-version".to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
    ];
    if refresh.unwrap_or(false) {
        args.push("--refresh".to_string());
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn list_molecule_assets(output_dir: Option<String>) -> Result<Value, String> {
    let mut args = vec!["molecules".to_string()];
    if let Some(output) = output_dir {
        if !output.trim().is_empty() {
            args.push("--output-dir".to_string());
            args.push(output);
        }
    }
    run_python_bridge_async(args).await
}

fn is_allowed_molecule_target(target: &str) -> bool {
    matches!(
        target,
        "insulin-mutation" | "glucagon" | "glut4" | "insulin-receptor" | "glucagon-receptor"
    )
}

#[tauri::command]
async fn generate_molecule_pae(target: String, output_dir: String) -> Result<Value, String> {
    if !is_allowed_molecule_target(target.trim()) {
        return Err("Unsupported AlphaFold PAE target.".to_string());
    }
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    run_python_bridge_async(vec![
        "molecule-pae".to_string(),
        "--target".to_string(),
        target,
        "--output-dir".to_string(),
        output_dir,
    ])
    .await
}

#[tauri::command]
async fn list_evidence_connectors() -> Result<Value, String> {
    run_python_bridge_async(vec!["evidence-connectors".to_string()]).await
}

#[tauri::command]
async fn run_genomics_simulation(
    gene: String,
    variant: String,
    output_dir: String,
    duration_minutes: Option<i64>,
) -> Result<Value, String> {
    if variant.trim().is_empty() {
        return Err("variant is required".to_string());
    }
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    let duration = duration_minutes.unwrap_or(360).clamp(60, 24 * 60);
    run_python_bridge_async(vec![
        "genomics-sim".to_string(),
        "--gene".to_string(),
        if gene.trim().is_empty() {
            "INSR".to_string()
        } else {
            gene
        },
        "--variant".to_string(),
        variant,
        "--output-dir".to_string(),
        output_dir,
        "--duration-minutes".to_string(),
        duration.to_string(),
    ])
    .await
}

#[tauri::command]
async fn run_tissue_stress(
    muscle_percent: Option<f64>,
    liver_percent: Option<f64>,
    output_dir: String,
) -> Result<Value, String> {
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    let muscle = muscle_percent.unwrap_or(30.0).clamp(0.0, 100.0);
    let liver = liver_percent.unwrap_or(100.0).clamp(0.0, 100.0);
    run_python_bridge_async(vec![
        "tissue-stress".to_string(),
        "--muscle-percent".to_string(),
        muscle.to_string(),
        "--liver-percent".to_string(),
        liver.to_string(),
        "--output-dir".to_string(),
        output_dir,
    ])
    .await
}

#[tauri::command]
async fn mechanistic_engine_status() -> Result<Value, String> {
    run_python_bridge_async(vec!["mechanistic-status".to_string()]).await
}

#[tauri::command]
async fn inspect_mechanistic_model(model: String) -> Result<Value, String> {
    if model.trim().is_empty() {
        return Err("model path is required".to_string());
    }
    run_python_bridge_async(vec![
        "mechanistic-inspect".to_string(),
        "--model".to_string(),
        model,
    ])
    .await
}

#[tauri::command]
async fn run_mechanistic_model(request: MechanisticRunRequest) -> Result<Value, String> {
    let MechanisticRunRequest {
        model,
        output_dir,
        start,
        end,
        points,
        variables,
        source_url,
        model_license,
    } = request;
    if model.trim().is_empty() {
        return Err("model path is required".to_string());
    }
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    let start_value = start.unwrap_or(0.0);
    let end_value = end.unwrap_or(1440.0);
    if !start_value.is_finite() || !end_value.is_finite() || end_value <= start_value {
        return Err("end must be finite and greater than start".to_string());
    }
    let bounded_points = points.unwrap_or(289);
    if !(2..=1_000_001).contains(&bounded_points) {
        return Err("points must be between 2 and 1,000,001".to_string());
    }
    let mut args = vec![
        "mechanistic-run".to_string(),
        "--model".to_string(),
        model,
        "--output-dir".to_string(),
        output_dir,
        "--start".to_string(),
        start_value.to_string(),
        "--end".to_string(),
        end_value.to_string(),
        "--points".to_string(),
        bounded_points.to_string(),
    ];
    let requested_variables = variables.unwrap_or_default();
    if requested_variables.len() > 256 {
        return Err("at most 256 variables may be selected".to_string());
    }
    for variable in requested_variables {
        if !variable.trim().is_empty() {
            args.push("--variable".to_string());
            args.push(variable);
        }
    }
    for (flag, value) in [
        ("--source-url", source_url),
        ("--model-license", model_license),
    ] {
        if let Some(text) = value {
            if !text.trim().is_empty() {
                args.push(flag.to_string());
                args.push(text);
            }
        }
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn cross_scale_engine_status() -> Result<Value, String> {
    run_python_bridge_async(vec!["cross-scale-status".to_string()]).await
}

#[tauri::command]
async fn inspect_copasi_model(model: String) -> Result<Value, String> {
    if model.trim().is_empty() {
        return Err("COPASI model path is required".to_string());
    }
    run_python_bridge_async(vec![
        "copasi-inspect".to_string(),
        "--model".to_string(),
        model,
    ])
    .await
}

#[tauri::command]
async fn run_copasi_analysis(request: CopasiRunRequest) -> Result<Value, String> {
    if request.model.trim().is_empty() || request.output_dir.trim().is_empty() {
        return Err("COPASI model and output directory are required".to_string());
    }
    if request.allow_external_execution != Some(true) {
        return Err(
            "Review the COPASI tasks and explicitly allow external execution first".to_string(),
        );
    }
    let timeout = request.timeout_seconds.unwrap_or(900);
    if !(1..=86_400).contains(&timeout) {
        return Err("COPASI timeout must be between 1 and 86,400 seconds".to_string());
    }
    let mut args = vec![
        "copasi-run".to_string(),
        "--model".to_string(),
        request.model,
        "--output-dir".to_string(),
        request.output_dir,
        "--timeout-seconds".to_string(),
        timeout.to_string(),
        "--allow-external-execution".to_string(),
    ];
    if let Some(task) = request.task {
        if !task.trim().is_empty() {
            args.push("--task".to_string());
            args.push(task);
        }
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn inspect_cellml_reference(model: String) -> Result<Value, String> {
    if model.trim().is_empty() {
        return Err("CellML model path is required".to_string());
    }
    run_python_bridge_async(vec![
        "cellml-inspect".to_string(),
        "--model".to_string(),
        model,
    ])
    .await
}

#[tauri::command]
async fn validate_cellml_reference(request: CellmlValidationRequest) -> Result<Value, String> {
    if request.model.trim().is_empty() || request.output_dir.trim().is_empty() {
        return Err("CellML model and output directory are required".to_string());
    }
    let timeout = request.timeout_seconds.unwrap_or(120);
    if !(1..=86_400).contains(&timeout) {
        return Err("OpenCOR timeout must be between 1 and 86,400 seconds".to_string());
    }
    run_python_bridge_async(vec![
        "cellml-validate".to_string(),
        "--model".to_string(),
        request.model,
        "--output-dir".to_string(),
        request.output_dir,
        "--timeout-seconds".to_string(),
        timeout.to_string(),
    ])
    .await
}

#[tauri::command]
async fn inspect_fmu_model(model: String) -> Result<Value, String> {
    if model.trim().is_empty() {
        return Err("FMU path is required".to_string());
    }
    run_python_bridge_async(vec![
        "fmi-inspect".to_string(),
        "--model".to_string(),
        model,
    ])
    .await
}

#[tauri::command]
async fn run_fmi_model(request: FmiRunRequest) -> Result<Value, String> {
    if request.model.trim().is_empty() || request.output_dir.trim().is_empty() {
        return Err("FMU and output directory are required".to_string());
    }
    if request.trust_native_code != Some(true) {
        return Err(
            "Review the FMU publisher/hash and explicitly trust native code first".to_string(),
        );
    }
    let start = request.start.unwrap_or(0.0);
    let end = request.end.unwrap_or(60.0);
    let interval = request.output_interval.unwrap_or(0.1);
    if !start.is_finite()
        || !end.is_finite()
        || !interval.is_finite()
        || end <= start
        || interval <= 0.0
    {
        return Err(
            "FMI timing values must be finite, with end > start and interval > 0".to_string(),
        );
    }
    if (end - start) / interval > 1_000_000.0 {
        return Err("FMI run exceeds the 1,000,001-row safety limit".to_string());
    }
    let timeout = request.timeout_seconds.unwrap_or(300);
    if !(1..=86_400).contains(&timeout) {
        return Err("FMI timeout must be between 1 and 86,400 seconds".to_string());
    }
    let mut args = vec![
        "fmi-run".to_string(),
        "--model".to_string(),
        request.model,
        "--output-dir".to_string(),
        request.output_dir,
        "--start".to_string(),
        start.to_string(),
        "--end".to_string(),
        end.to_string(),
        "--output-interval".to_string(),
        interval.to_string(),
        "--timeout-seconds".to_string(),
        timeout.to_string(),
        "--trust-native-code".to_string(),
    ];
    let variables = request.variables.unwrap_or_default();
    if variables.len() > 256 {
        return Err("At most 256 FMU variables may be selected".to_string());
    }
    for variable in variables {
        if !variable.trim().is_empty() {
            args.push("--variable".to_string());
            args.push(variable);
        }
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn query_bindingdb_evidence(request: BindingQueryRequest) -> Result<Value, String> {
    if request.uniprot.trim().is_empty() || request.output_dir.trim().is_empty() {
        return Err("UniProt accession and output directory are required".to_string());
    }
    let cutoff = request.cutoff_nm.unwrap_or(10_000);
    if !(1..=1_000_000_000).contains(&cutoff) {
        return Err("BindingDB cutoff must be between 1 and 1,000,000,000 nM".to_string());
    }
    let max_records = request.max_records.unwrap_or(5_000);
    if !(1..=100_000).contains(&max_records) {
        return Err("BindingDB max records must be between 1 and 100,000".to_string());
    }
    let timeout = request.timeout_seconds.unwrap_or(30);
    if !(1..=300).contains(&timeout) {
        return Err("BindingDB timeout must be between 1 and 300 seconds".to_string());
    }
    run_python_bridge_async(vec![
        "binding-query".to_string(),
        "--uniprot".to_string(),
        request.uniprot,
        "--output-dir".to_string(),
        request.output_dir,
        "--cutoff-nm".to_string(),
        cutoff.to_string(),
        "--max-records".to_string(),
        max_records.to_string(),
        "--timeout-seconds".to_string(),
        timeout.to_string(),
    ])
    .await
}

#[tauri::command]
async fn run_workflow(
    workflow_key: String,
    output_dir: String,
    seed: i64,
) -> Result<Value, String> {
    if workflow_key.trim().is_empty() {
        return Err("workflow_key is required".to_string());
    }
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    run_python_bridge_async(vec![
        "run".to_string(),
        "--workflow-key".to_string(),
        workflow_key,
        "--output-dir".to_string(),
        output_dir,
        "--seed".to_string(),
        seed.to_string(),
    ])
    .await
}

#[tauri::command]
fn start_workflow_job(
    workflow_key: String,
    output_dir: String,
    seed: i64,
) -> Result<Value, String> {
    if workflow_key.trim().is_empty() {
        return Err("workflow_key is required".to_string());
    }
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    if !(0..=i32::MAX as i64).contains(&seed) {
        return Err("seed must be between 0 and 2147483647".to_string());
    }

    let sequence = NEXT_WORKFLOW_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let job_id = format!("workflow-{}-{sequence}", std::process::id());
    let job_dir = env::temp_dir().join("iints-af-workflow-jobs").join(&job_id);
    fs::create_dir_all(&job_dir)
        .map_err(|error| format!("Could not create workflow job directory: {error}"))?;
    let progress_path = job_dir.join("progress.json");
    let cancel_path = job_dir.join("cancel.requested");

    {
        let mut jobs = workflow_jobs()
            .lock()
            .map_err(|_| "Workflow job registry is unavailable".to_string())?;
        if jobs.len() >= 64 {
            if let Some(stale_id) = jobs
                .iter()
                .find(|(_, job)| {
                    matches!(job.status.as_str(), "completed" | "failed" | "cancelled")
                })
                .map(|(id, _)| id.clone())
            {
                if let Some(stale) = jobs.remove(&stale_id) {
                    let _ =
                        fs::remove_dir_all(stale.progress_path.parent().unwrap_or(Path::new("")));
                }
            }
        }
        jobs.insert(
            job_id.clone(),
            WorkflowJobRecord {
                status: "queued".to_string(),
                phase: "queued".to_string(),
                progress_percent: 0.0,
                message: "Workflow queued".to_string(),
                cancel_requested: false,
                result: None,
                error: None,
                started: Instant::now(),
                progress_path: progress_path.clone(),
                cancel_path: cancel_path.clone(),
            },
        );
    }

    let thread_job_id = job_id.clone();
    thread::spawn(move || {
        if let Ok(mut jobs) = workflow_jobs().lock() {
            if let Some(job) = jobs.get_mut(&thread_job_id) {
                job.status = "running".to_string();
                job.phase = "preparing".to_string();
                job.message = "Starting Python research engine".to_string();
            }
        }
        let args = vec![
            "run".to_string(),
            "--workflow-key".to_string(),
            workflow_key,
            "--output-dir".to_string(),
            output_dir,
            "--seed".to_string(),
            seed.to_string(),
            "--progress-file".to_string(),
            progress_path.to_string_lossy().into_owned(),
            "--cancel-file".to_string(),
            cancel_path.to_string_lossy().into_owned(),
        ];
        let result = run_python_bridge(&args);
        if let Ok(mut jobs) = workflow_jobs().lock() {
            if let Some(job) = jobs.get_mut(&thread_job_id) {
                match result {
                    Ok(value) => {
                        job.status = "completed".to_string();
                        job.phase = "complete".to_string();
                        job.progress_percent = 100.0;
                        job.message = "Workflow and artifacts completed".to_string();
                        job.result = Some(value);
                    }
                    Err(error)
                        if job.cancel_requested || error.to_lowercase().contains("cancel") =>
                    {
                        job.status = "cancelled".to_string();
                        job.phase = "cancelled".to_string();
                        job.message = "Workflow cancelled; partial artifacts may remain for audit"
                            .to_string();
                        job.error = Some(error);
                    }
                    Err(error) => {
                        job.status = "failed".to_string();
                        job.phase = "failed".to_string();
                        job.message = "Workflow failed".to_string();
                        job.error = Some(error);
                    }
                }
            }
        }
    });

    Ok(json!({"job_id": job_id, "status": "queued"}))
}

#[tauri::command]
fn workflow_job_status(job_id: String) -> Result<Value, String> {
    if job_id.trim().is_empty() {
        return Err("job_id is required".to_string());
    }
    let progress_path = {
        let jobs = workflow_jobs()
            .lock()
            .map_err(|_| "Workflow job registry is unavailable".to_string())?;
        jobs.get(&job_id)
            .map(|job| job.progress_path.clone())
            .ok_or_else(|| "Unknown workflow job".to_string())?
    };

    let progress = fs::read_to_string(&progress_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok());
    let mut jobs = workflow_jobs()
        .lock()
        .map_err(|_| "Workflow job registry is unavailable".to_string())?;
    let job = jobs
        .get_mut(&job_id)
        .ok_or_else(|| "Unknown workflow job".to_string())?;
    if matches!(job.status.as_str(), "queued" | "running" | "cancelling") {
        if let Some(progress) = progress {
            if let Some(phase) = progress.get("phase").and_then(Value::as_str) {
                job.phase = phase.to_string();
            }
            if let Some(percent) = progress.get("progress_percent").and_then(Value::as_f64) {
                job.progress_percent = percent.clamp(0.0, 100.0);
            }
            if let Some(message) = progress.get("message").and_then(Value::as_str) {
                job.message = message.to_string();
            }
        }
    }
    Ok(workflow_job_payload(&job_id, job))
}

#[tauri::command]
fn cancel_workflow_job(job_id: String) -> Result<Value, String> {
    let mut jobs = workflow_jobs()
        .lock()
        .map_err(|_| "Workflow job registry is unavailable".to_string())?;
    let job = jobs
        .get_mut(&job_id)
        .ok_or_else(|| "Unknown workflow job".to_string())?;
    if matches!(job.status.as_str(), "completed" | "failed" | "cancelled") {
        return Ok(workflow_job_payload(&job_id, job));
    }
    fs::write(&job.cancel_path, b"cancel\n")
        .map_err(|error| format!("Could not request workflow cancellation: {error}"))?;
    job.cancel_requested = true;
    job.status = "cancelling".to_string();
    job.phase = "cancelling".to_string();
    job.message = "Cancellation requested; waiting for a safe simulation boundary".to_string();
    Ok(workflow_job_payload(&job_id, job))
}

#[tauri::command]
async fn preview_results(csv: String, max_rows: Option<i64>) -> Result<Value, String> {
    if csv.trim().is_empty() {
        return Err("csv path is required".to_string());
    }
    let bounded_rows = max_rows.unwrap_or(80).clamp(1, 200);
    run_python_bridge_async(vec![
        "preview".to_string(),
        "--csv".to_string(),
        csv,
        "--max-rows".to_string(),
        bounded_rows.to_string(),
    ])
    .await
}

#[tauri::command]
async fn compartment_timeline(csv: String, max_points: Option<i64>) -> Result<Value, String> {
    if csv.trim().is_empty() {
        return Err("csv path is required".to_string());
    }
    let bounded_points = max_points.unwrap_or(400).clamp(2, 2000);
    run_python_bridge_async(vec![
        "compartments".to_string(),
        "--csv".to_string(),
        csv,
        "--max-points".to_string(),
        bounded_points.to_string(),
    ])
    .await
}

#[tauri::command]
async fn run_history(output_dir: String, limit: Option<i64>) -> Result<Value, String> {
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    let bounded_limit = limit.unwrap_or(25).clamp(1, 200);
    run_python_bridge_async(vec![
        "history".to_string(),
        "--output-dir".to_string(),
        output_dir,
        "--limit".to_string(),
        bounded_limit.to_string(),
    ])
    .await
}

#[tauri::command]
async fn certify_mdmp(
    csv: String,
    quick_rows: Option<i64>,
    full: Option<bool>,
) -> Result<Value, String> {
    if csv.trim().is_empty() {
        return Err("csv path is required".to_string());
    }
    let bounded_rows = quick_rows.unwrap_or(5000).clamp(10, 250_000);
    let mut args = vec![
        "mdmp-certify".to_string(),
        "--csv".to_string(),
        csv,
        "--quick-rows".to_string(),
        bounded_rows.to_string(),
    ];
    if full.unwrap_or(false) {
        args.push("--full".to_string());
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn export_academic_bundle(
    run_dir: String,
    title: Option<String>,
    description: Option<String>,
    creator: Option<String>,
    orcid: Option<String>,
    license_id: Option<String>,
    source_ids: Option<Vec<String>>,
) -> Result<Value, String> {
    if run_dir.trim().is_empty() {
        return Err("run_dir is required".to_string());
    }
    let mut args = vec![
        "academic-bundle".to_string(),
        "--run-dir".to_string(),
        run_dir,
    ];
    for (flag, value) in [
        ("--title", title),
        ("--description", description),
        ("--creator", creator),
        ("--orcid", orcid),
        ("--license", license_id),
    ] {
        if let Some(text) = value {
            if !text.trim().is_empty() {
                args.push(flag.to_string());
                args.push(text);
            }
        }
    }
    for source_id in source_ids.unwrap_or_default() {
        if !source_id.trim().is_empty() {
            args.push("--source-id".to_string());
            args.push(source_id);
        }
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn check_local_ai(model: String, host: Option<String>) -> Result<Value, String> {
    let mut args = vec![
        "ai-check".to_string(),
        "--model".to_string(),
        if model.trim().is_empty() {
            "ministral-3:8b".to_string()
        } else {
            model
        },
    ];
    if let Some(host_value) = host {
        if !host_value.trim().is_empty() {
            args.push("--host".to_string());
            args.push(host_value);
        }
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn list_local_ai_models(host: Option<String>) -> Result<Value, String> {
    let mut args = vec!["ai-models".to_string()];
    if let Some(host_value) = host {
        if !host_value.trim().is_empty() {
            args.push("--host".to_string());
            args.push(host_value);
        }
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn start_local_ai(
    model: String,
    host: Option<String>,
    no_pull: Option<bool>,
) -> Result<Value, String> {
    let mut args = vec![
        "ai-start".to_string(),
        "--model".to_string(),
        if model.trim().is_empty() {
            "ministral-3:8b".to_string()
        } else {
            model
        },
    ];
    if let Some(host_value) = host {
        if !host_value.trim().is_empty() {
            args.push("--host".to_string());
            args.push(host_value);
        }
    }
    if no_pull.unwrap_or(false) {
        args.push("--no-pull".to_string());
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn ask_local_ai(
    question: String,
    model: String,
    host: Option<String>,
    csv: Option<String>,
) -> Result<Value, String> {
    if question.trim().is_empty() {
        return Err("question is required".to_string());
    }
    let mut args = vec![
        "ai-ask".to_string(),
        "--question".to_string(),
        question,
        "--model".to_string(),
        if model.trim().is_empty() {
            "ministral-3:8b".to_string()
        } else {
            model
        },
    ];
    if let Some(host_value) = host {
        if !host_value.trim().is_empty() {
            args.push("--host".to_string());
            args.push(host_value);
        }
    }
    if let Some(csv_value) = csv {
        if !csv_value.trim().is_empty() {
            args.push("--csv".to_string());
            args.push(csv_value);
        }
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn run_foundation_arena(
    output_dir: String,
    result_files: Vec<String>,
) -> Result<Value, String> {
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    let mut args = vec![
        "foundation-arena".to_string(),
        "--output-dir".to_string(),
        output_dir,
    ];
    if result_files.is_empty() {
        return Err("Select at least one measured foundation evaluation JSON".to_string());
    }
    for result_file in result_files {
        if result_file.trim().is_empty() {
            continue;
        }
        args.push("--result".to_string());
        args.push(result_file);
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn extract_glucofm_embedding(
    csv: String,
    checkpoint: String,
    glucose_column: Option<String>,
    timestamp_column: Option<String>,
) -> Result<Value, String> {
    if csv.trim().is_empty() || checkpoint.trim().is_empty() {
        return Err("A CGM CSV and trained GlucoFM checkpoint are required".to_string());
    }
    let mut args = vec![
        "glucofm-embed".to_string(),
        "--csv".to_string(),
        csv,
        "--checkpoint".to_string(),
        checkpoint,
    ];
    if let Some(column) = glucose_column.filter(|value| !value.trim().is_empty()) {
        args.push("--glucose-column".to_string());
        args.push(column);
    }
    if let Some(column) = timestamp_column.filter(|value| !value.trim().is_empty()) {
        args.push("--timestamp-column".to_string());
        args.push(column);
    }
    run_python_bridge_async(args).await
}

// Each parameter is a distinct named argument the frontend passes via
// invoke(), matching every other Tauri command in this file; bundling them
// into a struct here would be inconsistent with the rest of the file for no
// benefit, so the arg count lint is silenced rather than restructured.
#[allow(clippy::too_many_arguments)]
#[tauri::command]
async fn pretrain_glucofm(
    source: String,
    output_dir: String,
    glucose_column: Option<String>,
    timestamp_column: Option<String>,
    subject_column: Option<String>,
    epochs: Option<i64>,
    batch_size: Option<i64>,
    device: Option<String>,
    seed: Option<i64>,
) -> Result<Value, String> {
    if source.trim().is_empty() || output_dir.trim().is_empty() {
        return Err("A CGM dataset and output directory are required".to_string());
    }
    let mut args = vec![
        "glucofm-pretrain".to_string(),
        "--source".to_string(),
        source,
        "--output-dir".to_string(),
        output_dir,
        "--epochs".to_string(),
        epochs.unwrap_or(120).max(1).to_string(),
        "--batch-size".to_string(),
        batch_size.unwrap_or(128).max(1).to_string(),
        "--device".to_string(),
        device.unwrap_or_else(|| "auto".to_string()),
        "--seed".to_string(),
        seed.unwrap_or(42).to_string(),
    ];
    for (flag, value) in [
        ("--glucose-column", glucose_column),
        ("--timestamp-column", timestamp_column),
        ("--subject-column", subject_column),
    ] {
        if let Some(column) = value.filter(|candidate| !candidate.trim().is_empty()) {
            args.push(flag.to_string());
            args.push(column);
        }
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn load_cgmacros_cohort(
    output_dir: String,
    participants: Option<i64>,
) -> Result<Value, String> {
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    let mut args = vec![
        "cgmacros-cohort".to_string(),
        "--output-dir".to_string(),
        output_dir,
    ];
    if let Some(count) = participants {
        args.push("--participants".to_string());
        args.push(count.to_string());
    }
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn run_fda_safety_benchmark(output_dir: String) -> Result<Value, String> {
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    let args = vec![
        "fda-safety-benchmark".to_string(),
        "--output-dir".to_string(),
        output_dir,
    ];
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn generate_scientific_visualizations(output_dir: String) -> Result<Value, String> {
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    let args = vec![
        "foundation-visualize".to_string(),
        "--output-dir".to_string(),
        output_dir,
    ];
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn generate_eucys_playbook(output_dir: String) -> Result<Value, String> {
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    let args = vec![
        "eucys-playbook".to_string(),
        "--output-dir".to_string(),
        output_dir,
    ];
    run_python_bridge_async(args).await
}

#[tauri::command]
async fn open_path(path: String) -> Result<(), String> {
    open_path_allowlisted(path).await
}

#[tauri::command]
async fn reveal_path(path: String) -> Result<(), String> {
    reveal_path_allowlisted(path).await
}

#[tauri::command]
async fn open_external_url(url: String) -> Result<(), String> {
    open_external_url_allowlisted(url).await
}

#[tauri::command]
async fn open_sdk_update_terminal() -> Result<(), String> {
    open_sdk_update_terminal_allowlisted().await
}

async fn open_path_allowlisted(path: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let resolved = resolve_user_path(&path)?;
        validate_open_target(&resolved)?;
        open_with_platform(&resolved)
    })
    .await
    .map_err(|error| format!("Open-path task failed: {error}"))?
}

async fn reveal_path_allowlisted(path: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let resolved = resolve_user_path(&path)?;
        validate_open_target(&resolved)?;
        reveal_with_platform(&resolved)
    })
    .await
    .map_err(|error| format!("Reveal-path task failed: {error}"))?
}

async fn open_external_url_allowlisted(url: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        validate_external_url(&url)?;
        open_url_with_platform(&url)
    })
    .await
    .map_err(|error| format!("Open-url task failed: {error}"))?
}

async fn open_sdk_update_terminal_allowlisted() -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let command = build_sdk_maintenance_command_text()?;
        open_terminal_with_command(&command)
    })
    .await
    .map_err(|error| format!("Open-update-terminal task failed: {error}"))?
}

fn resolve_user_path(raw: &str) -> Result<PathBuf, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("Path is required.".to_string());
    }
    let expanded = if trimmed == "~" || trimmed.starts_with("~/") || trimmed.starts_with("~\\") {
        let home = home_dir().ok_or_else(|| "Could not resolve the home directory.".to_string())?;
        if trimmed.len() == 1 {
            home
        } else {
            home.join(&trimmed[2..])
        }
    } else {
        PathBuf::from(trimmed)
    };
    expanded.canonicalize().map_err(|error| {
        format!(
            "Cannot open path because it does not exist: {} ({error})",
            expanded.display()
        )
    })
}

fn home_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        env::var_os("USERPROFILE").map(PathBuf::from)
    } else {
        env::var_os("HOME").map(PathBuf::from)
    }
}

fn managed_python_engine_root() -> Option<PathBuf> {
    home_dir().map(|home| managed_python_engine_root_for(&home))
}

fn managed_python_engine_path() -> Option<PathBuf> {
    managed_python_engine_root().map(|root| managed_python_engine_path_for(&root))
}

fn managed_python_engine_root_for(home: &Path) -> PathBuf {
    home.join(".iints-af").join("python-engine")
}

fn managed_python_engine_path_for(root: &Path) -> PathBuf {
    if cfg!(windows) {
        root.join("Scripts").join("python.exe")
    } else {
        root.join("bin").join("python")
    }
}

fn validate_external_url(raw: &str) -> Result<(), String> {
    let trimmed = raw.trim();
    if !trimmed.starts_with("https://") {
        return Err("Only HTTPS evidence links are allowed.".to_string());
    }
    let host = extract_https_host(trimmed)?;
    const ALLOWED_EXTERNAL_HOSTS: &[&str] = &[
        "alphafold.ebi.ac.uk",
        "rest.ensembl.org",
        "platform.opentargets.org",
        "platform-docs.opentargets.org",
        "reactome.org",
        "www.rcsb.org",
        "data.rcsb.org",
        "search.rcsb.org",
        "www.uniprot.org",
        "rest.uniprot.org",
        "www.proteinatlas.org",
        "gtexportal.org",
        "www.ebi.ac.uk",
        "www.biomodels.org",
        "chembl.gitbook.io",
        "api.pharmgkb.org",
        "string-db.org",
        "clinicaltables.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "www.researchobject.org",
        "www.nature.com",
        "sed-ml.org",
        "sbml.org",
        "libroadrunner.readthedocs.io",
        "copasi.org",
        "opencor.ws",
        "models.physiomeproject.org",
        "fmi-standard.org",
        "fmpy.readthedocs.io",
        "www.bindingdb.org",
        "clinicaltrials.gov",
        "zenodo.org",
        "developers.zenodo.org",
        "github.com",
        "python35.github.io",
        "iints.org",
        "www.iints.org",
    ];
    if ALLOWED_EXTERNAL_HOSTS.contains(&host.as_str()) {
        Ok(())
    } else {
        Err(format!(
            "Refusing to open non-allowlisted external evidence host: {host}"
        ))
    }
}

fn extract_https_host(raw: &str) -> Result<String, String> {
    let without_scheme = raw
        .strip_prefix("https://")
        .ok_or_else(|| "Only HTTPS evidence links are allowed.".to_string())?;
    let authority = without_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .trim();
    if authority.is_empty() {
        return Err("Evidence URL is missing a host.".to_string());
    }
    if authority.contains('@') {
        return Err("Evidence URLs with user information are not allowed.".to_string());
    }
    let host = authority
        .split(':')
        .next()
        .unwrap_or("")
        .trim()
        .trim_end_matches('.')
        .to_ascii_lowercase();
    if host.is_empty() {
        Err("Evidence URL is missing a host.".to_string())
    } else {
        Ok(host)
    }
}

fn validate_open_target(path: &Path) -> Result<(), String> {
    if path.is_dir() {
        return Ok(());
    }
    if !path.is_file() {
        return Err(format!(
            "Open target is not a regular file or folder: {}",
            path.display()
        ));
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    const SAFE_EXTENSIONS: &[&str] = &[
        "csv", "json", "md", "pdf", "png", "jpg", "jpeg", "svg", "html", "htm", "txt", "log",
        "cif", "mmcif",
    ];
    if SAFE_EXTENSIONS.contains(&extension.as_str()) {
        Ok(())
    } else {
        Err(format!(
            "Refusing to open unsupported file type '.{}'. Open the containing folder instead.",
            extension
        ))
    }
}

fn build_sdk_update_command_parts() -> Result<Vec<String>, String> {
    let candidate = python_candidates()
        .into_iter()
        .find(python_candidate_has_iints_sdk)
        .ok_or_else(|| {
            "Could not find a Python interpreter with iints-sdk-python35 installed. Set IINTS_PYTHON to the correct executable before updating.".to_string()
        })?;
    let mut parts = vec![candidate.program];
    parts.extend(candidate.prefix_args);
    parts.extend([
        "-m".to_string(),
        "pip".to_string(),
        "install".to_string(),
        "-U".to_string(),
        "iints-sdk-python35[desktop-all]".to_string(),
    ]);
    Ok(parts)
}

fn python_candidate_has_iints_sdk(candidate: &PythonCandidate) -> bool {
    let mut command = Command::new(&candidate.program);
    command.args(&candidate.prefix_args);
    command.args([
        "-c",
        "import importlib.metadata; importlib.metadata.version('iints-sdk-python35'); import iints_desktop.tauri_bridge",
    ]);
    command.env("PYTHONUTF8", "1");
    command.stdout(Stdio::null()).stderr(Stdio::null());
    command
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn python_candidate_is_supported(candidate: &PythonCandidate) -> bool {
    let mut command = Command::new(&candidate.program);
    command.args(&candidate.prefix_args);
    command.args([
        "-c",
        "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)",
    ]);
    command.stdout(Stdio::null()).stderr(Stdio::null());
    command
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn quote_posix_arg(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=+,@%".contains(ch))
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn quote_cmd_arg(value: &str) -> String {
    if value.is_empty() {
        return "\"\"".to_string();
    }
    if !value
        .chars()
        .any(|ch| matches!(ch, ' ' | '\t' | '"' | '&' | '|' | '<' | '>' | '^'))
    {
        return value.to_string();
    }
    format!("\"{}\"", value.replace('"', "\\\""))
}

fn program_on_path(program: &str) -> bool {
    let Some(paths) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&paths).any(|dir| {
        let candidate = dir.join(program);
        if candidate.is_file() {
            return true;
        }
        if cfg!(target_os = "windows") {
            return dir.join(format!("{program}.exe")).is_file();
        }
        false
    })
}

fn build_sdk_update_command_text() -> Result<String, String> {
    let parts = build_sdk_update_command_parts()?;
    if cfg!(target_os = "windows") {
        Ok(parts
            .iter()
            .map(|part| quote_cmd_arg(part))
            .collect::<Vec<_>>()
            .join(" "))
    } else {
        Ok(parts
            .iter()
            .map(|part| quote_posix_arg(part))
            .collect::<Vec<_>>()
            .join(" "))
    }
}

fn build_sdk_install_command_text() -> Result<String, String> {
    let engine_root = managed_python_engine_root().ok_or_else(|| {
        "Could not resolve the home directory for the private Python engine.".to_string()
    })?;
    let engine_python = managed_python_engine_path()
        .ok_or_else(|| "Could not resolve the private Python engine executable.".to_string())?;
    let engine_candidate = PythonCandidate {
        program: engine_python.to_string_lossy().into_owned(),
        prefix_args: Vec::new(),
    };
    let engine_is_usable =
        engine_python.is_file() && python_candidate_is_supported(&engine_candidate);
    let install_parts = [
        engine_python.to_string_lossy().into_owned(),
        "-m".to_string(),
        "pip".to_string(),
        "install".to_string(),
        "--upgrade".to_string(),
        "pip".to_string(),
        "iints-sdk-python35[desktop-all]".to_string(),
    ];
    let quote = |value: &str| {
        if cfg!(windows) {
            quote_cmd_arg(value)
        } else {
            quote_posix_arg(value)
        }
    };

    let install_command = install_parts
        .iter()
        .map(|part| quote(part))
        .collect::<Vec<_>>()
        .join(" ");
    if engine_is_usable {
        return Ok(install_command);
    }

    let candidate = python_candidates()
        .into_iter()
        .filter(|item| item.program != engine_candidate.program)
        .find(python_candidate_is_supported)
        .ok_or_else(|| {
            "Python 3.10-3.14 was not found. Install a current Python from https://www.python.org/downloads/ and then use this button again.".to_string()
        })?;
    let mut create_parts = vec![candidate.program];
    create_parts.extend(candidate.prefix_args);
    create_parts.extend([
        "-m".to_string(),
        "venv".to_string(),
        "--clear".to_string(),
        engine_root.to_string_lossy().into_owned(),
    ]);
    let create_command = create_parts
        .iter()
        .map(|part| quote(part))
        .collect::<Vec<_>>()
        .join(" ");
    Ok(format!("{create_command} && {install_command}"))
}

fn build_sdk_maintenance_command_text() -> Result<String, String> {
    match build_sdk_update_command_text() {
        Ok(command) => Ok(command),
        Err(_) => build_sdk_install_command_text(),
    }
}

fn escape_applescript_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn open_terminal_with_command(command_text: &str) -> Result<(), String> {
    if cfg!(target_os = "macos") {
        let held_command = format!(
            "{}; echo; echo 'IINTS Python engine maintenance finished. Return to the app and choose Refresh versions.'; exec zsh",
            command_text
        );
        let script = format!(
            "tell application \"Terminal\" to do script \"{}\"",
            escape_applescript_string(&held_command)
        );
        Command::new("osascript")
            .args(["-e", &script])
            .spawn()
            .map(|_| ())
            .map_err(|error| format!("Could not open Terminal.app: {error}"))
    } else if cfg!(target_os = "windows") {
        Command::new("cmd.exe")
            .args([
                "/c",
                "start",
                "IINTS SDK Update",
                "cmd.exe",
                "/k",
                command_text,
            ])
            .spawn()
            .map(|_| ())
            .map_err(|error| format!("Could not open Windows terminal: {error}"))
    } else {
        let held_command = format!(
            "{}; echo; echo 'IINTS update finished. You may close this terminal.'; exec bash",
            command_text
        );
        let terminals: [(&str, Vec<&str>); 6] = [
            (
                "x-terminal-emulator",
                vec!["-e", "bash", "-lc", held_command.as_str()],
            ),
            (
                "gnome-terminal",
                vec!["--", "bash", "-lc", held_command.as_str()],
            ),
            ("konsole", vec!["-e", "bash", "-lc", held_command.as_str()]),
            (
                "xfce4-terminal",
                vec!["-x", "bash", "-lc", held_command.as_str()],
            ),
            ("xterm", vec!["-e", "bash", "-lc", held_command.as_str()]),
            (
                "alacritty",
                vec!["-e", "bash", "-lc", held_command.as_str()],
            ),
        ];
        for (program, args) in terminals {
            if program_on_path(program) {
                return Command::new(program)
                    .args(args)
                    .spawn()
                    .map(|_| ())
                    .map_err(|error| format!("Could not open {program}: {error}"));
            }
        }
        Err("No supported Linux terminal emulator was found.".to_string())
    }
}

fn open_url_with_platform(url: &str) -> Result<(), String> {
    let mut command = if cfg!(target_os = "macos") {
        let mut command = Command::new("open");
        command.arg(url);
        command
    } else if cfg!(target_os = "windows") {
        let mut command = Command::new("explorer");
        command.arg(url);
        command
    } else {
        let mut command = Command::new("xdg-open");
        command.arg(url);
        command
    };
    command
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("Could not open evidence URL {url}: {error}"))
}

fn open_with_platform(path: &Path) -> Result<(), String> {
    let mut command = if cfg!(target_os = "macos") {
        let mut command = Command::new("open");
        command.arg(path);
        command
    } else if cfg!(target_os = "windows") {
        let mut command = Command::new("explorer");
        command.arg(path);
        command
    } else {
        let mut command = Command::new("xdg-open");
        command.arg(path);
        command
    };
    command
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("Could not open {}: {error}", path.display()))
}

fn reveal_with_platform(path: &Path) -> Result<(), String> {
    let mut command = if cfg!(target_os = "macos") {
        let mut command = Command::new("open");
        command.args(["-R"]);
        command.arg(path);
        command
    } else if cfg!(target_os = "windows") {
        let mut command = Command::new("explorer");
        command.arg(format!("/select,{}", path.display()));
        command
    } else {
        let mut command = Command::new("xdg-open");
        command.arg(path.parent().unwrap_or(path));
        command
    };
    command
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("Could not reveal {}: {error}", path.display()))
}

fn main() {
    if env::args().any(|argument| argument == "--smoke") {
        println!("IINTS-AF Research Workbench smoke check passed");
        return;
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            desktop_status,
            desktop_app_info,
            list_workflows,
            desktop_diagnostics,
            desktop_update_info,
            list_molecule_assets,
            generate_molecule_pae,
            list_evidence_connectors,
            run_genomics_simulation,
            run_tissue_stress,
            mechanistic_engine_status,
            inspect_mechanistic_model,
            run_mechanistic_model,
            cross_scale_engine_status,
            inspect_copasi_model,
            run_copasi_analysis,
            inspect_cellml_reference,
            validate_cellml_reference,
            inspect_fmu_model,
            run_fmi_model,
            query_bindingdb_evidence,
            run_workflow,
            start_workflow_job,
            workflow_job_status,
            cancel_workflow_job,
            preview_results,
            compartment_timeline,
            run_history,
            certify_mdmp,
            export_academic_bundle,
            check_local_ai,
            list_local_ai_models,
            start_local_ai,
            ask_local_ai,
            open_path,
            reveal_path,
            open_external_url,
            open_sdk_update_terminal,
            run_foundation_arena,
            extract_glucofm_embedding,
            pretrain_glucofm,
            load_cgmacros_cohort,
            run_fda_safety_benchmark,
            generate_scientific_visualizations,
            generate_eucys_playbook
        ])
        .run(tauri::generate_context!())
        .expect("error while running IINTS-AF Tauri desktop");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_urls_require_an_exact_allowlisted_https_host() {
        assert!(validate_external_url("https://alphafold.ebi.ac.uk/entry/P01308").is_ok());
        assert!(validate_external_url("https://libroadrunner.readthedocs.io/en/latest/").is_ok());
        assert!(validate_external_url("https://models.physiomeproject.org/cellml").is_ok());
        assert!(validate_external_url("https://www.bindingdb.org/rwd/bind/index.jsp").is_ok());
        assert!(validate_external_url("https://www.researchobject.org/ro-crate/").is_ok());
        assert!(validate_external_url("https://sed-ml.org/").is_ok());
        assert!(validate_external_url("https://iints.org/").is_ok());
        assert!(validate_external_url("http://alphafold.ebi.ac.uk/entry/P01308").is_err());
        assert!(validate_external_url("https://alphafold.ebi.ac.uk.example.com/").is_err());
        assert!(validate_external_url("https://user@alphafold.ebi.ac.uk/").is_err());
    }

    #[test]
    fn molecule_pae_targets_are_explicitly_allowlisted() {
        assert!(is_allowed_molecule_target("insulin-mutation"));
        assert!(is_allowed_molecule_target("glucagon-receptor"));
        assert!(!is_allowed_molecule_target("all"));
        assert!(!is_allowed_molecule_target("../private"));
    }

    #[test]
    fn desktop_app_info_uses_the_stable_tauri_release_channel() {
        let info = desktop_app_info();
        assert_eq!(info["app_version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(
            info["release_url"],
            "https://github.com/python35/IINTS-SDK/releases/tag/tauri-beta-latest"
        );
    }

    #[test]
    fn managed_python_engine_stays_inside_the_user_home() {
        let root = managed_python_engine_root_for(Path::new("/home/researcher"));
        assert_eq!(root, Path::new("/home/researcher/.iints-af/python-engine"));
        let executable = managed_python_engine_path_for(&root);
        if cfg!(windows) {
            assert!(executable.ends_with(Path::new("Scripts/python.exe")));
        } else {
            assert!(executable.ends_with(Path::new("bin/python")));
        }
    }

    #[test]
    fn posix_update_arguments_are_shell_quoted() {
        assert_eq!(quote_posix_arg("python3"), "python3");
        assert_eq!(quote_posix_arg("/tmp/IINTS Python"), "'/tmp/IINTS Python'");
        assert_eq!(
            quote_posix_arg("value'; echo unsafe"),
            "'value'\"'\"'; echo unsafe'"
        );
    }

    #[test]
    fn windows_update_arguments_quote_shell_metacharacters() {
        assert_eq!(quote_cmd_arg("python.exe"), "python.exe");
        assert_eq!(
            quote_cmd_arg(r"C:\Program Files\Python\python.exe"),
            r#""C:\Program Files\Python\python.exe""#
        );
        assert_eq!(quote_cmd_arg("package&command"), r#""package&command""#);
    }

    #[test]
    fn bridge_timeouts_keep_health_checks_short_and_research_runs_long() {
        assert_eq!(
            python_bridge_timeout(&["status".to_string()]),
            Duration::from_secs(20)
        );
        assert_eq!(
            python_bridge_timeout(&["ai-ask".to_string()]),
            Duration::from_secs(1_200)
        );
        assert_eq!(
            python_bridge_timeout(&["run".to_string()]),
            Duration::from_secs(1_800)
        );
    }
}
