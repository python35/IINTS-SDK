use serde_json::{json, Value};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
struct PythonCandidate {
    program: String,
    prefix_args: Vec<String>,
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

fn run_python_bridge(args: &[String]) -> Result<Value, String> {
    let mut attempts = Vec::new();
    for candidate in python_candidates() {
        let mut command = Command::new(&candidate.program);
        command.args(&candidate.prefix_args);
        command.args(["-m", "iints_desktop.tauri_bridge"]);
        command.args(args);
        command.env("PYTHONUTF8", "1");

        match command.output() {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                if !output.status.success() && stdout.is_empty() {
                    attempts.push(format!(
                        "{} {:?}: exit {}; {}",
                        candidate.program,
                        candidate.prefix_args,
                        output.status,
                        stderr
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
                let details = envelope.get("details").and_then(Value::as_str).unwrap_or("");
                return Err(format!("{message}\n{details}").trim().to_string());
            }
            Err(error) => attempts.push(format!(
                "{} {:?}: {error}",
                candidate.program, candidate.prefix_args
            )),
        }
    }
    Err(format!(
        "Could not start Python for the IINTS SDK bridge. Set IINTS_PYTHON to the Python executable that has iints-sdk-python35 installed.\nAttempts:\n{}",
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
async fn list_workflows() -> Result<Value, String> {
    run_python_bridge_async(vec!["workflows".to_string()]).await
}

#[tauri::command]
async fn desktop_diagnostics() -> Result<Value, String> {
    run_python_bridge_async(vec!["diagnostics".to_string()]).await
}

#[tauri::command]
async fn list_molecule_assets() -> Result<Value, String> {
    run_python_bridge_async(vec!["molecules".to_string()]).await
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
async fn run_workflow(workflow_key: String, output_dir: String, seed: i64) -> Result<Value, String> {
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
async fn open_path(path: String) -> Result<(), String> {
    open_path_allowlisted(path).await
}

#[tauri::command]
async fn open_external_url(url: String) -> Result<(), String> {
    open_external_url_allowlisted(url).await
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

async fn open_external_url_allowlisted(url: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        validate_external_url(&url)?;
        open_url_with_platform(&url)
    })
    .await
    .map_err(|error| format!("Open-url task failed: {error}"))?
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
    expanded
        .canonicalize()
        .map_err(|error| format!("Cannot open path because it does not exist: {} ({error})", expanded.display()))
}

fn home_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        env::var_os("USERPROFILE").map(PathBuf::from)
    } else {
        env::var_os("HOME").map(PathBuf::from)
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
        "chembl.gitbook.io",
        "api.pharmgkb.org",
        "string-db.org",
        "clinicaltables.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov",
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
        return Err(format!("Open target is not a regular file or folder: {}", path.display()));
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

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            desktop_status,
            list_workflows,
            desktop_diagnostics,
            list_molecule_assets,
            list_evidence_connectors,
            run_genomics_simulation,
            run_tissue_stress,
            run_workflow,
            preview_results,
            run_history,
            certify_mdmp,
            check_local_ai,
            list_local_ai_models,
            start_local_ai,
            ask_local_ai,
            open_path,
            open_external_url
        ])
        .run(tauri::generate_context!())
        .expect("error while running IINTS-AF Tauri desktop");
}
