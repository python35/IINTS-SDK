use serde_json::{json, Value};
use std::env;
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

#[tauri::command]
fn desktop_status() -> Result<Value, String> {
    run_python_bridge(&["status".to_string()])
}

#[tauri::command]
fn list_workflows() -> Result<Value, String> {
    run_python_bridge(&["workflows".to_string()])
}

#[tauri::command]
fn run_workflow(workflow_key: String, output_dir: String, seed: i64) -> Result<Value, String> {
    if workflow_key.trim().is_empty() {
        return Err("workflow_key is required".to_string());
    }
    if output_dir.trim().is_empty() {
        return Err("output_dir is required".to_string());
    }
    run_python_bridge(&[
        "run".to_string(),
        "--workflow-key".to_string(),
        workflow_key,
        "--output-dir".to_string(),
        output_dir,
        "--seed".to_string(),
        seed.to_string(),
    ])
}

#[tauri::command]
fn preview_results(csv: String, max_rows: Option<i64>) -> Result<Value, String> {
    if csv.trim().is_empty() {
        return Err("csv path is required".to_string());
    }
    let bounded_rows = max_rows.unwrap_or(80).clamp(1, 200);
    run_python_bridge(&[
        "preview".to_string(),
        "--csv".to_string(),
        csv,
        "--max-rows".to_string(),
        bounded_rows.to_string(),
    ])
}

#[tauri::command]
fn check_local_ai(model: String, host: Option<String>) -> Result<Value, String> {
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
    run_python_bridge(&args)
}

#[tauri::command]
fn list_local_ai_models(host: Option<String>) -> Result<Value, String> {
    let mut args = vec!["ai-models".to_string()];
    if let Some(host_value) = host {
        if !host_value.trim().is_empty() {
            args.push("--host".to_string());
            args.push(host_value);
        }
    }
    run_python_bridge(&args)
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            desktop_status,
            list_workflows,
            run_workflow,
            preview_results,
            check_local_ai,
            list_local_ai_models
        ])
        .run(tauri::generate_context!())
        .expect("error while running IINTS-AF Tauri desktop");
}
