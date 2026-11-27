"""IINTS Research Studio — FastAPI Backend & Interactive Web Interface.

Strictly separates the scientific simulation/statistics kernel (``iints.sdk2`` &
``iints.research``) from the Research Studio HTTP API and interactive HTML UI.
Enforces the pre-registration approval gate before any experiment can execute.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from iints.research.designer import (
    PlanNotApprovedError,
    approve_research_plan,
    list_research_templates,
    load_research_plan,
    propose_research_plan,
    save_research_plan,
)
from iints.research.reports import generate_verifiable_report_bundle
from iints.research.runner import execute_research_plan
from iints.research.statistics import analyze_research_results
from iints.sdk2.contracts import (
    RESEARCH_ONLY_DISCLAIMER,
    ExperimentSpec,
    PatientSpec,
    ScenarioSpec,
    SensorFaultSpec,
    dump_experiment_yaml,
)
from iints.sdk2.reproducibility import ExperimentStore

__all__ = ["create_studio_app", "app"]


class ProposePlanRequest(BaseModel):
    template_id: str = "study_001_cgm_dropout"
    plan_id: Optional[str] = None
    n_profiles: int = Field(default=12, ge=2, le=500)
    seed: int = 42
    duration_hours: float = Field(default=24.0, gt=2.0, le=168.0)
    include_ood_scenarios: bool = True


class ApprovePlanRequest(BaseModel):
    reviewer: str = Field(default="researcher", min_length=1)


class RunPlanRequest(BaseModel):
    max_workers: int = Field(default=2, ge=1, le=16)
    resume: bool = True


class ScenarioPreviewRequest(BaseModel):
    dropout_minutes: int = Field(default=15, ge=0, le=180)
    noise_level: str = "moderate"  # low | moderate | high
    display_mode: str = "nan_blank"  # nan_blank | stale_hold
    cohort_variation: str = "large"


_STUDIO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IINTS Research Studio 2.0</title>
  <style>
    :root { --bg: #f8fafc; --card: #ffffff; --ink: #0f172a; --muted: #475569; --accent: #1b4965; --ok: #15803d; --warn: #b45309; --border: #e2e8f0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--ink); margin: 0; padding: 0; }
    header { background: var(--accent); color: #fff; padding: 18px 28px; display: flex; justify-content: space-between; align-items: center; }
    header h1 { margin: 0; font-size: 1.25rem; }
    .banner { background: #fffbeb; border-bottom: 1px solid #fde68a; color: #92400e; padding: 10px 28px; font-size: 0.85rem; }
    main { max-width: 1220px; margin: 24px auto; padding: 0 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .full { grid-column: 1 / -1; }
    h2 { margin-top: 0; font-size: 1.05rem; color: var(--accent); }
    label { display: block; font-size: 0.82rem; font-weight: 600; color: var(--muted); margin-top: 10px; }
    select, input { width: 100%; box-sizing: border-box; padding: 8px 10px; margin-top: 4px; border: 1px solid var(--border); border-radius: 6px; font-size: 0.9rem; }
    button { margin-top: 14px; background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 9px 15px; font-weight: 600; cursor: pointer; }
    button:disabled { background: #94a3b8; cursor: not-allowed; }
    button.approve { background: var(--ok); }
    .badge { display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
    .proposed { background: #fef3c7; color: var(--warn); }
    .approved { background: #dcfce7; color: var(--ok); }
    pre { background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 0.8rem; max-height: 280px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.84rem; margin-top: 10px; }
    th, td { border-bottom: 1px solid var(--border); padding: 7px 9px; text-align: left; }
    th { background: #f1f5f9; }
  </style>
</head>
<body>
  <header>
    <h1>IINTS Research Studio 2.0 — Reproducible Diabetes Algorithm Lab</h1>
    <span style="font-size:0.82rem;opacity:0.9">SDK 2.0 Engine + Pre-Registered Research Lab</span>
  </header>
  <div class="banner">
    <strong>Pre-Clinical Research &amp; Education Only:</strong> Not a medical device. All numbers come strictly from stored simulation artifacts, never from generative text.
  </div>
  <main>
    <section class="card">
      <h2>1. Experiment Designer &amp; Approval Gate</h2>
      <label>Scientific Template</label>
      <select id="tpl">
        <option value="study_001_cgm_dropout">Study 001 — CGM Dropout Robustness (0/5/15/30 min)</option>
        <option value="study_002_inclusive_stale_ui">Study 002 — Blank (NaN) vs Stale Display Hazard</option>
        <option value="study_003_cross_simulator">Study 003 — Cross-Simulator Validation (Hovorka/Bergman/ODE)</option>
      </select>
      <label>Virtual Patient Profiles (N)</label>
      <input id="n_profiles" type="number" value="10" min="4" max="200" />
      <button onclick="proposePlan()">1. Propose Experimental Design</button>
      <div id="planBox" style="margin-top:14px;display:none;">
        <div>Status: <span id="planStatus" class="badge proposed">PROPOSED</span> | Cells: <strong id="planCells"></strong></div>
        <p id="planQuestion" style="font-size:0.86rem;color:#334155;"></p>
        <label>Approving Researcher</label>
        <input id="reviewer" type="text" value="Rune Bobbaers" />
        <button id="btnApprove" class="approve" onclick="approvePlan()">2. Approve &amp; Lock Pre-Registration (SHA-256)</button>
        <button id="btnRun" disabled onclick="runPlan()">3. Execute Reproducible Experiment</button>
      </div>
    </section>

    <section class="card">
      <h2>2. Interactive Sensor Fault Scenario Configurator</h2>
      <label>CGM Signal Interruption (minutes)</label>
      <input id="cfgDrop" type="number" value="15" min="0" max="120" step="5" onchange="previewYaml()" />
      <label>Sensor Measurement Noise</label>
      <select id="cfgNoise" onchange="previewYaml()">
        <option value="low">Low (3.5 mg/dL)</option>
        <option value="moderate" selected>Moderate (6.0 mg/dL)</option>
        <option value="high">High (10.5 mg/dL)</option>
      </select>
      <label>Missing-Data Display Behavior</label>
      <select id="cfgDisplay" onchange="previewYaml()">
        <option value="nan_blank" selected>Explicit Blank / Missing (NaN)</option>
        <option value="stale_hold">Unflagged Stale Reading Held (UI Hazard Test)</option>
      </select>
      <pre id="yamlBox">Loading YAML...</pre>
    </section>

    <section class="card full">
      <h2>3. Verifiable Paired Within-Patient Results (Zero Pseudoreplication)</h2>
      <div id="resultsStatus" style="font-size:0.88rem;color:#475569;">Approve and run a plan above to compute deterministic paired metrics, bootstrap 95% CIs, and Holm-Bonferroni p-values.</div>
      <div id="resultsTable"></div>
    </section>
  </main>
  <script>
    let activePlanId = null;
    async function previewYaml() {
      const res = await fetch('/api/scenarios/preview-yaml', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          dropout_minutes: parseInt(document.getElementById('cfgDrop').value || '15'),
          noise_level: document.getElementById('cfgNoise').value,
          display_mode: document.getElementById('cfgDisplay').value
        })
      });
      const data = await res.json();
      document.getElementById('yamlBox').textContent = data.yaml;
    }
    async function proposePlan() {
      const res = await fetch('/api/plans/propose', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          template_id: document.getElementById('tpl').value,
          n_profiles: parseInt(document.getElementById('n_profiles').value || '10')
        })
      });
      const data = await res.json();
      activePlanId = data.plan_id;
      document.getElementById('planBox').style.display = 'block';
      document.getElementById('planCells').textContent = data.total_cells;
      document.getElementById('planQuestion').textContent = data.research_question;
      const st = document.getElementById('planStatus');
      st.textContent = 'PROPOSED (NOT EXECUTED)'; st.className = 'badge proposed';
      document.getElementById('btnRun').disabled = true;
    }
    async function approvePlan() {
      if (!activePlanId) return;
      const res = await fetch(`/api/plans/${activePlanId}/approve`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({reviewer: document.getElementById('reviewer').value})
      });
      const data = await res.json();
      const st = document.getElementById('planStatus');
      st.textContent = `APPROVED (${data.plan_lock_sha256.slice(0, 12)}...)`;
      st.className = 'badge approved';
      document.getElementById('btnRun').disabled = false;
    }
    async function runPlan() {
      if (!activePlanId) return;
      document.getElementById('resultsStatus').textContent = 'Running simulations across virtual cohort...';
      const res = await fetch(`/api/plans/${activePlanId}/execute`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({max_workers: 2, resume: true})
      });
      const data = await res.json();
      document.getElementById('resultsStatus').textContent =
        `Completed ${data.completed_cells} cells | Report: ${data.report_path}`;
      const rows = data.paired_vs_control || [];
      if (!rows.length) return;
      const cols = ['condition_id','algorithm_id','n_paired_patients','mean_delta_mae','ci95_delta_low','ci95_delta_high','p_holm_bonferroni'];
      let html = '<table><thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
      for (const r of rows) {
        html += '<tr>' + cols.map(c => `<td>${typeof r[c]==='number' ? r[c].toFixed(3) : r[c]}</td>`).join('') + '</tr>';
      }
      html += '</tbody></table>';
      document.getElementById('resultsTable').innerHTML = html;
    }
    previewYaml();
  </script>
</body>
</html>
"""


def create_studio_app(workspace_dir: Path | str = "results/research_studio") -> FastAPI:
    ws = Path(workspace_dir)
    plans_dir = ws / "plans"
    runs_dir = ws / "runs"
    plans_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    studio = FastAPI(
        title="IINTS Research Studio 2.0",
        description=RESEARCH_ONLY_DISCLAIMER,
        version="2.0.0",
    )

    @studio.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _STUDIO_HTML

    @studio.get("/api/templates")
    def get_templates() -> Dict[str, Any]:
        return {"disclaimer": RESEARCH_ONLY_DISCLAIMER, "templates": list_research_templates()}

    @studio.post("/api/scenarios/preview-yaml")
    def preview_scenario_yaml(req: ScenarioPreviewRequest) -> Dict[str, str]:
        noise_map = {"low": 3.5, "moderate": 6.0, "high": 10.5}
        sd = noise_map.get(req.noise_level, 6.0)
        display = "stale_hold" if req.display_mode == "stale_hold" else "nan_blank"
        spec = ExperimentSpec(
            id=f"cgm_dropout_{req.dropout_minutes:03d}m",
            seed=42,
            patient=PatientSpec(model="hovorka_equilibrium", profile="virtual_adult_01"),
            scenario=ScenarioSpec(id="standard_day", duration_hours=24.0, meal_schedule="standard_day"),
            sensor=SensorFaultSpec(
                model="cgm_v1",
                condition_id=f"dropout_{req.dropout_minutes}m_{display}",
                noise_sd_mgdl=sd,
                dropout_minutes=req.dropout_minutes,
                dropout_at_hour=8.5,
                display_mode=display,  # type: ignore[arg-type]
            ),
        )
        return {"yaml": dump_experiment_yaml(spec)}

    @studio.post("/api/plans/propose")
    def api_propose_plan(req: ProposePlanRequest) -> Dict[str, Any]:
        plan = propose_research_plan(
            req.template_id,
            plan_id=req.plan_id,
            n_profiles=req.n_profiles,
            seed=req.seed,
            duration_hours=req.duration_hours,
            include_ood_scenarios=req.include_ood_scenarios,
        )
        save_research_plan(plan, plans_dir / f"{plan.plan_id}.yaml")
        return {
            "plan_id": plan.plan_id,
            "status": plan.status,
            "research_question": plan.research_question,
            "total_cells": plan.total_simulation_cells(),
            "hypotheses": [h.model_dump() for h in plan.hypotheses],
        }

    @studio.post("/api/plans/{plan_id}/approve")
    def api_approve_plan(plan_id: str, req: ApprovePlanRequest) -> Dict[str, Any]:
        plan_file = plans_dir / f"{plan_id}.yaml"
        if not plan_file.exists():
            raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found.")
        plan = load_research_plan(plan_file)
        approved = approve_research_plan(plan, reviewer=req.reviewer)
        save_research_plan(approved, plan_file)
        return {
            "plan_id": approved.plan_id,
            "status": approved.status,
            "approved_by": approved.approved_by,
            "approved_utc": approved.approved_utc,
            "plan_lock_sha256": approved.plan_lock_sha256,
        }

    @studio.post("/api/plans/{plan_id}/execute")
    def api_execute_plan(plan_id: str, req: RunPlanRequest) -> Dict[str, Any]:
        plan_file = plans_dir / f"{plan_id}.yaml"
        if not plan_file.exists():
            raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found.")
        plan = load_research_plan(plan_file)
        try:
            run_out = execute_research_plan(
                plan,
                runs_dir / plan_id,
                max_workers=req.max_workers,
                resume=req.resume,
            )
        except PlanNotApprovedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        analysis = analyze_research_results(run_out["cell_metrics"], plan)
        bundle = generate_verifiable_report_bundle(run_out, analysis)

        def _json_clean(df) -> list[Dict[str, Any]]:
            if df.empty:
                return []
            cleaned = df.astype(object).where(df.notna(), None)
            return cleaned.to_dict(orient="records")

        return {
            "plan_id": plan_id,
            "completed_cells": int(run_out["resumed_cells"] + run_out["executed_cells"]),
            "report_path": str(bundle["report_md"]),
            "condition_summary": _json_clean(analysis["condition_summary"]),
            "paired_vs_control": _json_clean(analysis["paired_vs_control"]),
        }

    @studio.get("/api/experiments")
    def api_list_experiments() -> Dict[str, Any]:
        store = ExperimentStore(runs_dir / "experiments.sqlite")
        return {"experiments": store.list_experiments()}

    return studio


app = create_studio_app()
