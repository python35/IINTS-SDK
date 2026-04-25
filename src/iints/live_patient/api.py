from __future__ import annotations

import json
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .edge_ops import summarize_edge_workspace
from .runtime import PatientRuntimeStore


class MealRequest(BaseModel):
    carbs: float = Field(..., gt=0.0, description="Meal carbohydrate amount in grams")


class ScenarioResetRequest(BaseModel):
    scenario_profile: str = Field(..., description="Scenario profile to load during expo reset")


CONTROL_HEADER_NAME = "X-IINTS-Control"
CONTROL_HEADER_VALUE = "1"
SECURITY_RESPONSE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
}


def _build_dashboard_csp(nonce: str) -> str:
    return (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        f"style-src 'self' 'nonce-{nonce}'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "object-src 'none'"
    )


def _render_dashboard_html(*, kiosk: bool = False, api_token: str | None = None, csp_nonce: str) -> str:
    body_class = "kiosk" if kiosk else ""
    subtitle = (
        "Fullscreen expo view for the persistent IINTS digital patient."
        if kiosk
        else "A live virtual diabetes patient running on-device for expo demos, teaching, and long-horizon algorithm validation."
    )
    token_literal = json.dumps(api_token or "")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IINTS Digital Patient</title>
  <style nonce="{csp_nonce}">
    :root {{
      --bg: #f5f7f2;
      --panel: #ffffff;
      --ink: #162116;
      --muted: #5d6b5d;
      --line: #2a79c9;
      --warn: #c94a2a;
      --ok: #166f47;
      --border: #d8e0d5;
      --target: rgba(89, 160, 96, 0.14);
    }}
    body {{ font-family: Georgia, 'Times New Roman', serif; margin: 0; background: radial-gradient(circle at top, #fefefe, var(--bg)); color: var(--ink); }}
    body.kiosk {{ background: linear-gradient(180deg, #fbfcf8, #eef4eb); }}
    .wrap {{ max-width: 1240px; margin: 0 auto; padding: 32px 20px 48px; }}
    body.kiosk .wrap {{ max-width: 1400px; padding-top: 24px; }}
    h1 {{ font-size: 2.1rem; margin: 0 0 8px; }}
    body.kiosk h1 {{ font-size: 2.8rem; }}
    p.sub {{ margin: 0 0 24px; color: var(--muted); font-size: 1.05rem; }}
    body.kiosk p.sub {{ font-size: 1.2rem; }}
    .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 16px; }}
    body.kiosk .grid {{ grid-template-columns: repeat(4, minmax(220px, 1fr)); }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 16px 18px; box-shadow: 0 8px 24px rgba(20,40,20,0.06); }}
    .label {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 6px; }}
    body.kiosk .label {{ font-size: 1rem; }}
    .value {{ font-size: 1.65rem; font-weight: 700; }}
    body.kiosk .value {{ font-size: 2rem; }}
    .value.small {{ font-size: 1.2rem; }}
    .value.compact {{ font-size: 1.15rem; }}
    body.kiosk .value.small {{ font-size: 1.35rem; }}
    body.kiosk .value.compact {{ font-size: 1.35rem; }}
    .pill {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.85rem; font-weight: 700; }}
    .pill.ok {{ background: rgba(22, 111, 71, 0.12); color: var(--ok); }}
    .pill.warn {{ background: rgba(201, 74, 42, 0.12); color: var(--warn); }}
    .chart-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 22px; padding: 16px; box-shadow: 0 8px 24px rgba(20,40,20,0.06); }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0 4px; }}
    .controls button {{ border: 1px solid var(--border); background: #f0f5ee; color: var(--ink); padding: 10px 14px; border-radius: 999px; cursor: pointer; font-size: 0.95rem; }}
    .controls button.primary {{ background: #1f6f45; color: white; border-color: #1f6f45; }}
    .controls button.warn {{ background: #fff3ec; color: #8a3e1f; border-color: #e6c2b0; }}
    .section-title {{ font-size: 1rem; font-weight: 700; margin-top: 10px; margin-bottom: 8px; color: var(--muted); }}
    .status-note {{ margin-top: 10px; color: var(--muted); font-size: 0.95rem; min-height: 1.2em; }}
    .mini-meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 14px; }}
    svg {{ width: 100%; height: 360px; display: block; }}
    body.kiosk svg {{ height: 460px; }}
    .axis text {{ fill: var(--muted); font-size: 12px; }}
    .target-band {{ fill: var(--target); }}
    .glucose-line {{ fill: none; stroke: var(--line); stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
    .threshold {{ stroke: #d8a027; stroke-dasharray: 6 6; stroke-width: 1.5; }}
    .threshold.low {{ stroke: var(--warn); }}
    .footer {{ margin-top: 18px; color: var(--muted); font-size: 0.95rem; }}
    code {{ background: #eef3eb; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body class="{body_class}">
  <div class="wrap">
    <h1>IINTS Digital Patient</h1>
    <p class="sub">{subtitle}</p>
    <div class="grid">
      <div class="card"><div class="label">Runtime</div><div class="value" id="runtime-status">-</div></div>
      <div class="card"><div class="label">Simulated Clock</div><div class="value" id="runtime-clock">-</div></div>
      <div class="card"><div class="label">Current Glucose</div><div class="value" id="runtime-glucose">-</div></div>
      <div class="card"><div class="label">Last Event</div><div class="value compact" id="runtime-event">-</div></div>
    </div>
    <div class="mini-meta">
      <div class="card"><div class="label">Scenario</div><div class="value small" id="runtime-scenario">-</div></div>
      <div class="card"><div class="label">Algorithm</div><div class="value small" id="runtime-algo">-</div></div>
      <div class="card"><div class="label">Certification</div><div class="value small" id="runtime-cert">-</div></div>
      <div class="card"><div class="label">Realism Review</div><div class="value small" id="runtime-review">-</div></div>
    </div>
    <div class="chart-card">
      <svg viewBox="0 0 960 360" preserveAspectRatio="none" id="glucose-chart"></svg>
      <div class="section-title">Live controls</div>
      <div class="controls">
        <button data-command="/control/pause">Pause</button>
        <button data-command="/control/resume">Resume</button>
        <button class="primary" data-meal="60">Inject 60 g meal</button>
        <button class="warn" data-command="/control/expo-reset">Expo reset</button>
      </div>
      <div class="section-title">Scenario shortcuts</div>
      <div class="controls">
        <button data-scenario="normal_day">Normal day</button>
        <button data-scenario="sport_day">Sport day</button>
        <button data-scenario="bad_carb_count">Bad carb count</button>
        <button data-scenario="night_hypo_risk">Night hypo risk</button>
        <button class="warn" data-scenario="expo_hot_start">Expo hot start</button>
      </div>
      <div class="status-note" id="status-note"></div>
    </div>
    <p class="footer">Tip: present this page full-screen on the Raspberry Pi and use <code>Raspberry Pi Connect</code> screen sharing to control it from your laptop.</p>
  </div>
  <script nonce="{csp_nonce}">
    const CONTROL_HEADER_NAME = {json.dumps(CONTROL_HEADER_NAME)};
    const CONTROL_HEADER_VALUE = {json.dumps(CONTROL_HEADER_VALUE)};
    const AUTH_TOKEN = {token_literal};

    async function fetchJson(path, options) {{
      const response = await fetch(path, options);
      if (!response.ok) {{
        throw new Error(`Request failed: ${{response.status}}`);
      }}
      return response.json();
    }}

    function buildControlHeaders() {{
      const headers = {{
        'Content-Type': 'application/json',
        [CONTROL_HEADER_NAME]: CONTROL_HEADER_VALUE,
      }};
      if (AUTH_TOKEN) {{
        headers['Authorization'] = `Bearer ${{AUTH_TOKEN}}`;
      }}
      return headers;
    }}

    function buildReadHeaders() {{
      if (!AUTH_TOKEN) {{
        return {{}};
      }}
      return {{ 'Authorization': `Bearer ${{AUTH_TOKEN}}` }};
    }}

    function setStatusNote(message) {{
      document.getElementById('status-note').textContent = message;
    }}

    async function sendCommand(path) {{
      try {{
        await fetchJson(path, {{ method: 'POST', headers: buildControlHeaders() }});
        setStatusNote(`Queued: ${{path.split('/').pop()}}`);
        await refresh();
      }} catch (error) {{
        console.error(error);
        setStatusNote(`Command failed: ${{error.message}}`);
      }}
    }}

    async function sendScenario(profile) {{
      try {{
        await fetchJson('/control/scenario-reset', {{
          method: 'POST',
          headers: buildControlHeaders(),
          body: JSON.stringify({{ scenario_profile: profile }})
        }});
        setStatusNote(`Queued scenario reset: ${{profile}}`);
        await refresh();
      }} catch (error) {{
        console.error(error);
        setStatusNote(`Scenario reset failed: ${{error.message}}`);
      }}
    }}

    async function sendMeal(carbs) {{
      try {{
        await fetchJson('/events/meal', {{
          method: 'POST',
          headers: buildControlHeaders(),
          body: JSON.stringify({{ carbs }})
        }});
        setStatusNote(`Queued: manual meal (${{carbs}} g)`);
        await refresh();
      }} catch (error) {{
        console.error(error);
        setStatusNote(`Meal injection failed: ${{error.message}}`);
      }}
    }}

    function renderChart(records) {{
      const svg = document.getElementById('glucose-chart');
      const width = 960;
      const height = 360;
      const margin = {{ top: 16, right: 18, bottom: 30, left: 48 }};
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const minY = 40;
      const maxY = 300;
      const values = records.length ? records : [{{ simulated_minutes: 0, glucose_mgdl: 110 }}];
      const minX = values[0].simulated_minutes;
      const maxX = values[values.length - 1].simulated_minutes || minX + 5;
      const xScale = (value) => margin.left + ((value - minX) / Math.max(maxX - minX, 1)) * innerWidth;
      const yScale = (value) => margin.top + ((maxY - value) / (maxY - minY)) * innerHeight;
      const line = values.map((item, index) => `${{index === 0 ? 'M' : 'L'}} ${{xScale(item.simulated_minutes)}} ${{yScale(item.glucose_mgdl)}}`).join(' ');

      const bandTop = yScale(180);
      const bandBottom = yScale(70);
      const lowThreshold = yScale(70);
      const highThreshold = yScale(180);

      svg.innerHTML = `
        <rect class="target-band" x="${{margin.left}}" y="${{bandTop}}" width="${{innerWidth}}" height="${{bandBottom - bandTop}}"></rect>
        <line class="threshold low" x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{lowThreshold}}" y2="${{lowThreshold}}"></line>
        <line class="threshold" x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{highThreshold}}" y2="${{highThreshold}}"></line>
        <path class="glucose-line" d="${{line}}"></path>
        <g class="axis">
          <text x="${{margin.left}}" y="${{height - 8}}">Start</text>
          <text x="${{width - margin.right - 60}}" y="${{height - 8}}">Now</text>
          <text x="6" y="${{yScale(180) + 4}}">180</text>
          <text x="6" y="${{yScale(70) + 4}}">70</text>
        </g>
      `;
    }}

    function renderBadge(elementId, label, variant) {{
      const target = document.getElementById(elementId);
      target.replaceChildren();
      const pill = document.createElement('span');
      pill.className = `pill ${{variant}}`;
      pill.textContent = label;
      target.appendChild(pill);
    }}

    function bindControls() {{
      document.querySelectorAll('[data-command]').forEach((button) => {{
        button.addEventListener('click', async () => {{
          await sendCommand(button.dataset.command);
        }});
      }});
      document.querySelectorAll('[data-scenario]').forEach((button) => {{
        button.addEventListener('click', async () => {{
          await sendScenario(button.dataset.scenario);
        }});
      }});
      document.querySelectorAll('[data-meal]').forEach((button) => {{
        button.addEventListener('click', async () => {{
          const carbs = Number(button.dataset.meal || '0');
          await sendMeal(carbs);
        }});
      }});
    }}

    async function refresh() {{
      try {{
        const [status, history] = await Promise.all([
          fetchJson('/status', {{ headers: buildReadHeaders() }}),
          fetchJson('/glucose/history?limit=96', {{ headers: buildReadHeaders() }})
        ]);
        document.getElementById('runtime-status').textContent = status.daemon_status || '-';
        document.getElementById('runtime-clock').textContent = status.simulated_clock || '-';
        const glucose = typeof status.last_glucose_mgdl === 'number' ? `${{status.last_glucose_mgdl.toFixed(0)}} mg/dL` : '-';
        document.getElementById('runtime-glucose').textContent = glucose;
        document.getElementById('runtime-event').textContent = status.last_event_summary || 'No recent manual events';
        document.getElementById('runtime-scenario').textContent = status.scenario_profile || '-';
        document.getElementById('runtime-algo').textContent = status.algorithm_name || '-';
        if (!status.certification || !status.certification.exists) {{
          renderBadge('runtime-cert', 'Not certified yet', 'warn');
        }} else {{
          renderBadge('runtime-cert', status.certification.grade || 'certified', 'ok');
        }}
        if (!status.review || !status.review.exists) {{
          renderBadge('runtime-review', 'No review yet', 'warn');
        }} else {{
          renderBadge('runtime-review', 'Review ready', 'ok');
        }}
        if (status.message) {{
          setStatusNote(status.message);
        }}
        renderChart(history.records || []);
      }} catch (error) {{
        console.error(error);
        setStatusNote(`Refresh failed: ${{error.message}}`);
      }}
    }}

    bindControls();
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


def create_patient_app(workspace: str | Path, api_token: str | None = None) -> FastAPI:
    workspace_path = Path(workspace).expanduser().resolve()
    store = PatientRuntimeStore(workspace_path / "patient_state.db")
    app = FastAPI(title="IINTS Digital Patient")

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        for header_name, header_value in SECURITY_RESPONSE_HEADERS.items():
            response.headers.setdefault(header_name, header_value)
        return response

    def _request_token(request: Request) -> str | None:
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            return authorization[7:].strip()
        query_token = request.query_params.get("token")
        return query_token.strip() if query_token else None

    def _require_read_access(request: Request) -> None:
        if api_token is None:
            return
        if _request_token(request) != api_token:
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")

    def _require_control_access(request: Request) -> None:
        _require_read_access(request)
        if request.headers.get(CONTROL_HEADER_NAME) != CONTROL_HEADER_VALUE:
            raise HTTPException(
                status_code=403,
                detail="Control requests must include the dedicated IINTS control header.",
            )

    @app.get("/", response_class=HTMLResponse)
    def root(request: Request) -> HTMLResponse:
        _require_read_access(request)
        nonce = secrets.token_urlsafe(16)
        return HTMLResponse(
            _render_dashboard_html(api_token=api_token, csp_nonce=nonce),
            headers={"Content-Security-Policy": _build_dashboard_csp(nonce)},
        )

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        _require_read_access(request)
        nonce = secrets.token_urlsafe(16)
        return HTMLResponse(
            _render_dashboard_html(api_token=api_token, csp_nonce=nonce),
            headers={"Content-Security-Policy": _build_dashboard_csp(nonce)},
        )

    @app.get("/kiosk", response_class=HTMLResponse)
    def kiosk(request: Request) -> HTMLResponse:
        _require_read_access(request)
        nonce = secrets.token_urlsafe(16)
        return HTMLResponse(
            _render_dashboard_html(kiosk=True, api_token=api_token, csp_nonce=nonce),
            headers={"Content-Security-Policy": _build_dashboard_csp(nonce)},
        )

    @app.get("/status")
    def status(request: Request) -> JSONResponse:
        _require_read_access(request)
        return JSONResponse(summarize_edge_workspace(workspace_path))

    @app.get("/glucose/latest")
    def glucose_latest(request: Request) -> JSONResponse:
        _require_read_access(request)
        latest = store.get_latest_reading()
        if latest is None:
            raise HTTPException(status_code=404, detail="No glucose samples recorded yet.")
        return JSONResponse(latest)

    @app.get("/glucose/history")
    def glucose_history(request: Request, limit: int = 288) -> JSONResponse:
        _require_read_access(request)
        rows = store.get_recent_readings(limit=limit)
        payload = {
            "records": [
                {
                    "simulated_minutes": int(row["simulated_minutes"]),
                    "simulated_clock": row["simulated_clock"],
                    "glucose_mgdl": float(row["glucose_mgdl"] or 0.0),
                    "safety_reason": row["safety_reason"],
                    "event_summary": row["event_summary"],
                }
                for row in rows
            ]
        }
        return JSONResponse(payload)

    @app.post("/events/meal")
    def inject_meal(request: MealRequest, http_request: Request) -> JSONResponse:
        _require_control_access(http_request)
        command_id = store.enqueue_command("inject_meal", {"carbs": request.carbs})
        return JSONResponse({"command_id": command_id, "queued": True})

    @app.post("/control/pause")
    def pause(request: Request) -> JSONResponse:
        _require_control_access(request)
        command_id = store.enqueue_command("pause")
        return JSONResponse({"command_id": command_id, "queued": True})

    @app.post("/control/resume")
    def resume(request: Request) -> JSONResponse:
        _require_control_access(request)
        command_id = store.enqueue_command("resume")
        return JSONResponse({"command_id": command_id, "queued": True})

    @app.post("/control/expo-reset")
    def expo_reset(request: Request) -> JSONResponse:
        _require_control_access(request)
        command_id = store.enqueue_command("expo_reset")
        return JSONResponse({"command_id": command_id, "queued": True})

    @app.post("/control/scenario-reset")
    def scenario_reset(request: ScenarioResetRequest, http_request: Request) -> JSONResponse:
        _require_control_access(http_request)
        command_id = store.enqueue_command("expo_reset", {"scenario_profile": request.scenario_profile})
        return JSONResponse({"command_id": command_id, "queued": True})

    @app.post("/control/stop")
    def stop(request: Request) -> JSONResponse:
        _require_control_access(request)
        command_id = store.enqueue_command("stop")
        return JSONResponse({"command_id": command_id, "queued": True})

    return app
