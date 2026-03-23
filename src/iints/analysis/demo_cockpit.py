from __future__ import annotations

import base64
import html
import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd

from iints.analysis.booth_demo import build_booth_demo


@dataclass(frozen=True)
class DemoCockpitScenario:
    label: str
    slug: str
    headline: str
    jury_takeaway: str
    results_csv: str
    report_pdf: str
    run_manifest_path: str
    tir_70_180: float
    tir_below_70: float
    tir_above_180: float
    mean_glucose: float
    min_glucose: float
    max_glucose: float
    supervisor_events: int
    meal_events: int
    duration_hours: float
    total_steps: int
    chart_svg: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DemoCockpitBundle:
    output_dir: str
    html_path: str
    summary_json: str
    poster_png: str
    poster_summary_json: str
    demo_summary_json: str
    jury_talk_track: str
    live_demo_script: str
    run_commands: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


SVG_WIDTH = 720
SVG_HEIGHT = 240
SVG_PAD_X = 24
SVG_PAD_Y = 20
SVG_MIN_GLUCOSE = 40.0
SVG_MAX_GLUCOSE = 300.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _embed_image(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
    }.get(suffix, "application/octet-stream")
    return f"data:{mime};base64,{encoded}"


def _escape(text: str) -> str:
    return html.escape(text, quote=False)


def _glucose_column(df: pd.DataFrame) -> str:
    for candidate in ("glucose_actual_mgdl", "glucose_to_algo_mgdl", "glucose"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find a glucose column in results.csv")


def _time_column(df: pd.DataFrame) -> str:
    if "time_minutes" in df.columns:
        return "time_minutes"
    raise ValueError("results.csv must include a 'time_minutes' column")


def _override_mask(df: pd.DataFrame) -> pd.Series:
    if "safety_triggered" in df.columns:
        return df["safety_triggered"].fillna(False).astype(bool)
    if {"algo_recommended_insulin_units", "delivered_insulin_units"}.issubset(df.columns):
        return (df["algo_recommended_insulin_units"] - df["delivered_insulin_units"]) > 1e-9
    return pd.Series([False] * len(df), index=df.index)


def _meal_mask(df: pd.DataFrame) -> pd.Series:
    if "carb_intake_grams" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df["carb_intake_grams"].fillna(0).astype(float) > 0


def _scale_x(time_value: float, min_time: float, max_time: float) -> float:
    if max_time <= min_time:
        return SVG_PAD_X
    usable = SVG_WIDTH - (SVG_PAD_X * 2)
    return SVG_PAD_X + ((time_value - min_time) / (max_time - min_time)) * usable


def _scale_y(glucose_value: float) -> float:
    usable = SVG_HEIGHT - (SVG_PAD_Y * 2)
    clamped = min(max(glucose_value, SVG_MIN_GLUCOSE), SVG_MAX_GLUCOSE)
    normalized = (clamped - SVG_MIN_GLUCOSE) / (SVG_MAX_GLUCOSE - SVG_MIN_GLUCOSE)
    return SVG_HEIGHT - SVG_PAD_Y - (normalized * usable)


def _build_chart_svg(df: pd.DataFrame) -> str:
    glucose_col = _glucose_column(df)
    time_col = _time_column(df)

    times = (df[time_col].astype(float) / 60.0).tolist()
    glucose = df[glucose_col].astype(float).tolist()
    override_mask = _override_mask(df)
    meal_mask = _meal_mask(df)

    min_time = min(times) if times else 0.0
    max_time = max(times) if times else 24.0

    points = " ".join(
        f"{_scale_x(t, min_time, max_time):.2f},{_scale_y(g):.2f}"
        for t, g in zip(times, glucose)
    )

    def _circle_markup(mask: pd.Series, fill: str, radius: int) -> str:
        nodes: list[str] = []
        for idx, enabled in enumerate(mask.astype(bool).tolist()):
            if not enabled:
                continue
            nodes.append(
                f'<circle cx="{_scale_x(times[idx], min_time, max_time):.2f}" '
                f'cy="{_scale_y(glucose[idx]):.2f}" r="{radius}" fill="{fill}" />'
            )
        return "".join(nodes)

    target_y_top = _scale_y(180.0)
    target_y_bottom = _scale_y(70.0)
    target_height = max(target_y_bottom - target_y_top, 1.0)

    tick_lines = []
    for tick in range(0, int(max_time) + 1, 2):
        x = _scale_x(float(tick), min_time, max_time)
        tick_lines.append(
            f'<line x1="{x:.2f}" y1="{SVG_PAD_Y}" x2="{x:.2f}" y2="{SVG_HEIGHT - SVG_PAD_Y}" class="grid" />'
            f'<text x="{x:.2f}" y="{SVG_HEIGHT - 4}" class="axis-label" text-anchor="middle">{tick}h</text>'
        )

    y_lines = []
    for y_tick in (70, 120, 180, 250):
        y = _scale_y(float(y_tick))
        y_lines.append(
            f'<line x1="{SVG_PAD_X}" y1="{y:.2f}" x2="{SVG_WIDTH - SVG_PAD_X}" y2="{y:.2f}" class="grid" />'
            f'<text x="6" y="{y + 4:.2f}" class="axis-label">{y_tick}</text>'
        )

    return (
        f'<svg viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" class="scenario-chart" role="img" aria-label="Glucose chart">'
        f'<rect x="{SVG_PAD_X}" y="{target_y_top:.2f}" width="{SVG_WIDTH - (SVG_PAD_X * 2)}" height="{target_height:.2f}" class="target-zone" />'
        + "".join(tick_lines)
        + "".join(y_lines)
        + f'<polyline points="{points}" class="glucose-line" />'
        + _circle_markup(meal_mask, "#f59e0b", 5)
        + _circle_markup(override_mask, "#e63946", 5)
        + "</svg>"
    )


def _load_stage_script_excerpt(script_path: Path) -> str:
    lines = script_path.read_text(encoding="utf-8").splitlines()
    keep: list[str] = []
    for line in lines:
        keep.append(line)
        if line.strip() == "PREPARE_AI = True":
            break
    return "\n".join(keep).strip() + "\n"


def _available_patient_configs() -> list[str]:
    names: list[str] = []
    for item in sorted(files("iints.data.virtual_patients").iterdir(), key=lambda entry: entry.name):
        if item.name.endswith(".yaml"):
            names.append(item.name.replace(".yaml", ""))
    return names


def _build_html(
    *,
    output_dir: Path,
    patient_config: str,
    poster_path: Path,
    stage_script_excerpt: str,
    patient_profiles: list[str],
    scenarios: list[DemoCockpitScenario],
    run_commands: str,
    live_demo_script: str,
    ai_ready: bool,
) -> str:
    poster_data_uri = _embed_image(poster_path)
    ai_pill = "AI-ready safety case" if ai_ready else "AI preparation optional"

    story_cards = [
        {
            "step": "01",
            "title": "Show The Code",
            "body": "Start on a compact file that makes the knobs obvious: patient, duration, seed, output folder.",
        },
        {
            "step": "02",
            "title": "Run One Command",
            "body": "Kick off three reproducible scenarios in one go: normal control, stress handling, and supervisor override.",
        },
        {
            "step": "03",
            "title": "Explain The Result",
            "body": "Walk the jury through the poster, artifact bundle, and optional local AI explanation for the safety case.",
        },
    ]
    story_cards_html = "".join(
        f'<article class="story-card"><div class="story-step">{card["step"]}</div>'
        f'<div class="story-title">{_escape(card["title"])}</div>'
        f'<p class="story-body">{_escape(card["body"])}</p></article>'
        for card in story_cards
    )
    scenario_cards_html = "".join(
        f'''<article class="panel scenario-card">
            <div class="scenario-kicker">{_escape(scenario.slug)}</div>
            <div class="scenario-title">{_escape(scenario.label)}</div>
            <p class="scenario-headline">{_escape(scenario.headline)}</p>
            {scenario.chart_svg}
            <div class="metric-row">
              <div class="metric"><span class="metric-label">TIR 70-180</span><span class="metric-value">{scenario.tir_70_180:.1f}%</span></div>
              <div class="metric"><span class="metric-label">Time &lt;70</span><span class="metric-value">{scenario.tir_below_70:.1f}%</span></div>
              <div class="metric"><span class="metric-label">Overrides</span><span class="metric-value">{scenario.supervisor_events}</span></div>
              <div class="metric"><span class="metric-label">Meal events</span><span class="metric-value">{scenario.meal_events}</span></div>
            </div>
            <p class="takeaway">{_escape(scenario.jury_takeaway)}</p>
            <div class="artifact-list">
              <a class="artifact-link" href="{_escape(scenario.slug + '/results.csv')}">Open results CSV</a>
              <a class="artifact-link" href="{_escape(scenario.slug + '/clinical_report.pdf')}">Open clinical report PDF</a>
              <a class="artifact-link" href="{_escape(scenario.slug + '/run_manifest.json')}">Open run manifest</a>
            </div>
          </article>'''
        for scenario in scenarios
    )

    html_text = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>IINTS Demo Cockpit</title>
  <style>
    :root {{
      --ink: #132535;
      --muted: #557286;
      --panel: rgba(255,255,255,0.92);
      --line: rgba(19,37,53,0.10);
      --navy: #0f2b46;
      --blue: #1877f2;
      --teal: #1b9aaa;
      --gold: #f6aa1c;
      --red: #e63946;
      --green: #2a9d8f;
      --bg-a: #eff7ff;
      --bg-b: #f7f9fc;
      --shadow: 0 20px 50px rgba(15, 43, 70, 0.12);
      --radius: 24px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Avenir Next', 'Segoe UI', 'Helvetica Neue', sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 20%, rgba(24,119,242,0.16), transparent 24%),
        radial-gradient(circle at 80% 0%, rgba(27,154,170,0.18), transparent 26%),
        linear-gradient(180deg, var(--bg-a), var(--bg-b));
      min-height: 100vh;
    }}
    .shell {{ max-width: 1520px; margin: 0 auto; padding: 28px 24px 60px; }}
    .hero {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 22px; align-items: stretch; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); backdrop-filter: blur(12px); }}
    .hero-copy {{ padding: 30px; }}
    .eyebrow {{ display: inline-flex; align-items: center; gap: 8px; font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--teal); font-weight: 700; }}
    h1 {{ margin: 14px 0 14px; font-size: clamp(2.5rem, 4.4vw, 4.3rem); line-height: 0.95; color: var(--navy); }}
    .subtitle {{ font-size: 1.08rem; line-height: 1.6; color: var(--muted); max-width: 62ch; }}
    .pills {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .pill {{ border-radius: 999px; padding: 9px 14px; background: rgba(24,119,242,0.08); color: var(--navy); font-weight: 600; font-size: 0.92rem; }}
    .story-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin-top: 22px; }}
    .story-card {{ padding: 16px; border-radius: 18px; background: rgba(15,43,70,0.04); border: 1px solid rgba(15,43,70,0.07); }}
    .story-step {{ font-size: 0.82rem; font-weight: 800; color: var(--teal); letter-spacing: 0.08em; text-transform: uppercase; }}
    .story-title {{ margin: 10px 0 6px; font-size: 1.08rem; font-weight: 800; color: var(--navy); }}
    .story-body {{ margin: 0; color: var(--muted); line-height: 1.55; font-size: 0.95rem; }}
    .poster-wrap {{ padding: 20px; display: flex; flex-direction: column; gap: 14px; }}
    .poster-wrap img {{ width: 100%; border-radius: 20px; border: 1px solid rgba(15,43,70,0.08); }}
    .callout {{ padding: 16px 18px; border-radius: 18px; background: linear-gradient(135deg, rgba(42,157,143,0.10), rgba(24,119,242,0.10)); color: var(--navy); font-weight: 600; line-height: 1.55; }}
    .grid-two {{ display: grid; grid-template-columns: 0.92fr 1.08fr; gap: 22px; margin-top: 22px; }}
    .code-panel, .terminal-panel, .artifact-panel {{ padding: 24px; }}
    h2 {{ margin: 0 0 14px; font-size: 1.35rem; color: var(--navy); }}
    .section-note {{ color: var(--muted); line-height: 1.55; margin-bottom: 16px; }}
    pre {{ margin: 0; overflow-x: auto; border-radius: 18px; padding: 18px; background: #0d1b2a; color: #f7fbff; font-family: 'SFMono-Regular', Menlo, Consolas, monospace; font-size: 0.88rem; line-height: 1.5; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05); }}
    code.inline {{ font-family: 'SFMono-Regular', Menlo, Consolas, monospace; background: rgba(24,119,242,0.08); padding: 0.18rem 0.45rem; border-radius: 999px; }}
    .profile-list {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .profile-chip {{ border-radius: 999px; padding: 8px 12px; background: #fff; border: 1px solid rgba(15,43,70,0.10); color: var(--navy); font-size: 0.88rem; font-weight: 600; }}
    .scenario-section {{ margin-top: 24px; }}
    .scenario-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }}
    .scenario-card {{ padding: 22px; }}
    .scenario-kicker {{ font-size: 0.82rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--teal); font-weight: 800; }}
    .scenario-title {{ margin: 8px 0 6px; font-size: 1.3rem; font-weight: 800; color: var(--navy); }}
    .scenario-headline {{ margin: 0 0 16px; color: var(--muted); line-height: 1.55; min-height: 3.2em; }}
    .scenario-chart {{ width: 100%; background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(235,245,248,0.84)); border-radius: 18px; border: 1px solid rgba(15,43,70,0.08); margin-bottom: 16px; }}
    .scenario-chart .grid {{ stroke: rgba(19,37,53,0.08); stroke-width: 1; }}
    .scenario-chart .axis-label {{ fill: #6b8798; font-size: 11px; font-family: 'Avenir Next', 'Segoe UI', sans-serif; }}
    .scenario-chart .target-zone {{ fill: rgba(42,157,143,0.14); }}
    .scenario-chart .glucose-line {{ fill: none; stroke: #1877f2; stroke-width: 3.2; stroke-linecap: round; stroke-linejoin: round; }}
    .metric-row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-bottom: 16px; }}
    .metric {{ padding: 12px; border-radius: 16px; background: rgba(15,43,70,0.04); border: 1px solid rgba(15,43,70,0.07); }}
    .metric-label {{ display: block; color: var(--muted); font-size: 0.82rem; margin-bottom: 5px; }}
    .metric-value {{ font-size: 1.15rem; font-weight: 800; color: var(--navy); }}
    .takeaway {{ margin: 0 0 16px; color: var(--ink); line-height: 1.6; font-size: 0.96rem; }}
    .artifact-list {{ display: grid; gap: 8px; }}
    .artifact-link {{ display: block; text-decoration: none; color: var(--navy); background: rgba(255,255,255,0.86); border: 1px solid rgba(15,43,70,0.09); border-radius: 14px; padding: 11px 12px; font-size: 0.9rem; font-weight: 600; }}
    .artifact-link:hover {{ border-color: rgba(24,119,242,0.35); transform: translateY(-1px); }}
    .footer-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin-top: 24px; }}
    .smallprint {{ color: var(--muted); font-size: 0.92rem; line-height: 1.6; }}
    .terminal-command {{ color: #7dd3fc; }}
    .terminal-comment {{ color: #93c5fd; }}
    @media (max-width: 1180px) {{
      .hero, .grid-two, .footer-grid, .scenario-grid, .story-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class=\"shell\">
    <section class=\"hero\">
      <div class=\"panel hero-copy\">
        <div class=\"eyebrow\">IINTS-AF Demo Cockpit</div>
        <h1>Show Code. Run It. Make The Safety Story Visible.</h1>
        <p class=\"subtitle\">This mini-app is built from the real SDK demo pipeline. It turns the booth bundle into a full-screen walkthrough so you can show exactly how the SDK works: swap a patient profile, run the scenarios, and explain the resulting safety evidence.</p>
        <div class=\"pills\">
          <span class=\"pill\">Current patient profile: <code class=\"inline\">{_escape(patient_config)}</code></span>
          <span class=\"pill\">Three scenarios</span>
          <span class=\"pill\">Poster + artifacts</span>
          <span class=\"pill\">{_escape(ai_pill)}</span>
        </div>
        <div class=\"story-grid\">
          {story_cards_html}
        </div>
      </div>
      <div class=\"panel poster-wrap\">
        <img src=\"{poster_data_uri}\" alt=\"IINTS booth poster\">
        <div class=\"callout\">The poster is not a mockup. It is generated from the same run bundles, CSV files, PDF reports, manifests, and optional AI-ready artifacts that the SDK writes during the demo.</div>
      </div>
    </section>

    <section class=\"grid-two\">
      <article class=\"panel code-panel\">
        <h2>1. The Script You Show</h2>
        <p class=\"section-note\">Start from the smallest readable script. On stage, point to the patient profile first, then explain that the same SDK pipeline can be re-run for another packaged patient or a custom YAML config.</p>
        <pre>{_escape(stage_script_excerpt)}</pre>
        <div class=\"profile-list\">
          {''.join(f'<span class="profile-chip">{_escape(name)}</span>' for name in patient_profiles)}
        </div>
      </article>
      <article class=\"panel terminal-panel\">
        <h2>2. The Command You Run</h2>
        <p class=\"section-note\">Use the shell wrapper for the cleanest live flow. It makes the command memorable and keeps the explanation focused on the results, not on typing.</p>
        <pre><span class=\"terminal-comment\"># fastest fair command</span>\n<span class=\"terminal-command\">./scripts/run_live_stage_demo.sh --patient-config {html.escape(patient_config)}</span>\n\n<span class=\"terminal-comment\"># installed CLI alternative</span>\n<span class=\"terminal-command\">iints demo-booth --output-dir results/booth_demo</span></pre>
        <p class=\"section-note\" style=\"margin-top:16px; margin-bottom:0;\">What you tell the jury while it runs: the SDK creates a normal control case, a stress case, and a supervisor override case, and every one of them produces real reproducible artifacts.</p>
      </article>
    </section>

    <section class=\"scenario-section\">
      <div class=\"scenario-grid\">
        {scenario_cards_html}
      </div>
    </section>

    <section class=\"footer-grid\">
      <article class=\"panel artifact-panel\">
        <h2>3. The Commands You Keep Nearby</h2>
        <p class=\"section-note\">These are the exact helper notes you can keep open next to the demo.</p>
        <pre>{_escape(run_commands.strip())}</pre>
      </article>
      <article class=\"panel artifact-panel\">
        <h2>4. The Jury Notes And AI Step</h2>
        <p class=\"section-note\">If the conversation goes deeper, these files keep the explanation tight and consistent.</p>
        <pre>{_escape(live_demo_script.strip())}</pre>
        <p class=\"smallprint\" style=\"margin-top:16px;\">The optional AI follow-up uses the same Supervisor Override run and gives a local explanation only after the SDK has already generated the run bundle and safety artifacts.</p>
      </article>
    </section>
  </main>
</body>
</html>
"""
    return html_text


def build_demo_cockpit(
    output_dir: str | Path = "./results/demo_cockpit",
    *,
    patient_config: str = "default_patient",
    duration_minutes: int = 360,
    time_step: int = 5,
    seed: int = 42,
    prepare_ai: bool = True,
    create_dev_mdmp_cert: bool = True,
) -> dict[str, str]:
    """Build a visual demo cockpit HTML app from the fair-ready booth bundle."""
    resolved_output = Path(output_dir).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    booth_outputs = build_booth_demo(
        output_dir=resolved_output,
        patient_config=patient_config,
        duration_minutes=duration_minutes,
        time_step=time_step,
        seed=seed,
        prepare_ai=prepare_ai,
        create_dev_mdmp_cert=create_dev_mdmp_cert,
    )

    poster_summary = _read_json(Path(booth_outputs["poster_summary_json"]))
    demo_summary = _read_json(Path(booth_outputs["demo_summary_json"]))
    scenario_notes = {
        item["slug"]: item for item in demo_summary.get("scenarios", []) if isinstance(item, dict) and "slug" in item
    }

    scenarios: list[DemoCockpitScenario] = []
    for item in poster_summary.get("scenarios", []):
        run_dir = Path(item["run_dir"])
        df = pd.read_csv(item["results_csv"])
        slug = run_dir.name
        note = scenario_notes.get(slug, {})
        scenarios.append(
            DemoCockpitScenario(
                label=str(item["label"]),
                slug=slug,
                headline=str(note.get("headline", "")),
                jury_takeaway=str(note.get("jury_takeaway", "")),
                results_csv=str(item["results_csv"]),
                report_pdf=str(note.get("report_pdf", run_dir / "clinical_report.pdf")),
                run_manifest_path=str(note.get("run_manifest_path", run_dir / "run_manifest.json")),
                tir_70_180=float(item["tir_70_180"]),
                tir_below_70=float(item["tir_below_70"]),
                tir_above_180=float(item["tir_above_180"]),
                mean_glucose=float(item["mean_glucose"]),
                min_glucose=float(item["min_glucose"]),
                max_glucose=float(item["max_glucose"]),
                supervisor_events=int(item["supervisor_events"]),
                meal_events=int(item["meal_events"]),
                duration_hours=float(item["duration_hours"]),
                total_steps=int(item["total_steps"]),
                chart_svg=_build_chart_svg(df),
            )
        )

    stage_script_path = Path(__file__).resolve().parents[3] / "examples" / "demos" / "07_live_stage_demo.py"
    html_output = resolved_output / "demo_cockpit.html"
    summary_output = resolved_output / "demo_cockpit.json"
    live_demo_script_path = Path(booth_outputs["live_demo_script"])
    run_commands_path = Path(booth_outputs["run_commands"])
    jury_talk_track_path = Path(booth_outputs["jury_talk_track"])

    html_text = _build_html(
        output_dir=resolved_output,
        patient_config=patient_config,
        poster_path=Path(booth_outputs["poster_png"]),
        stage_script_excerpt=_load_stage_script_excerpt(stage_script_path),
        patient_profiles=_available_patient_configs(),
        scenarios=scenarios,
        run_commands=run_commands_path.read_text(encoding="utf-8"),
        live_demo_script=live_demo_script_path.read_text(encoding="utf-8"),
        ai_ready="mdmp_cert" in booth_outputs,
    )
    html_output.write_text(html_text, encoding="utf-8")

    bundle = DemoCockpitBundle(
        output_dir=str(resolved_output),
        html_path=str(html_output),
        summary_json=str(summary_output),
        poster_png=str(booth_outputs["poster_png"]),
        poster_summary_json=str(booth_outputs["poster_summary_json"]),
        demo_summary_json=str(booth_outputs["demo_summary_json"]),
        jury_talk_track=str(jury_talk_track_path),
        live_demo_script=str(live_demo_script_path),
        run_commands=str(run_commands_path),
    )
    summary_payload = {
        **bundle.to_dict(),
        "patient_config": patient_config,
        "duration_minutes": duration_minutes,
        "time_step_minutes": time_step,
        "seed": seed,
        "scenarios": [scenario.to_dict() for scenario in scenarios],
        "mdmp_cert": booth_outputs.get("mdmp_cert"),
    }
    summary_output.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    result = bundle.to_dict()
    if "mdmp_cert" in booth_outputs:
        result["mdmp_cert"] = booth_outputs["mdmp_cert"]
    return result
