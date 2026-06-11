from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from iints.ai.backends.ollama import DEFAULT_MINISTRAL_MODEL, OllamaBackend


@dataclass(frozen=True)
class Anomaly:
    index: int
    time_min: float
    kind: str
    detail: str

    @property
    def time_h(self) -> float:
        return self.time_min / 60.0


class AIRealismAuditor:
    """
    Red-team auditor for long IINTS-AF physiological simulation outputs.

    The auditor deliberately has two stages:

    1. A deterministic, fast heuristic filter scans the whole CSV for impossible
       or physiologically suspicious values.
    2. Optionally, a local Ollama model receives only the small data windows
       around flagged events and explains whether the event is a plausible
       physiological extreme or likely a mathematical/data bug.

    It supports both the scratch AdvancedMetabolicModel CSV format
    (``time_min``, ``glucose``, ``insulin_delivered``, ``ketones``) and the
    official Jetson endurance output columns (``time_minutes``,
    ``glucose_actual_mgdl``, ``delivered_insulin_units``).
    """

    COLUMN_ALIASES = {
        "time_min": ("time_min", "time_minutes", "minute", "minutes", "time"),
        "glucose": (
            "glucose",
            "glucose_actual_mgdl",
            "glucose_mechanistic_mgdl",
            "current_glucose",
            "glucose_mgdl",
            "glucose_mg_dL",
        ),
        "insulin_delivered": (
            "insulin_delivered",
            "delivered_insulin_units",
            "observed_delivered_insulin_units",
            "delivered_insulin",
            "insulin_units",
        ),
        "ketones": ("ketones", "plasma_ketones_mmol_L", "ketones_mmol_l", "ketones_mmol_L"),
        "ffa": ("ffa", "plasma_ffa_mmol_L", "ffa_mmol_l", "ffa_mmol_L"),
        "event": ("event", "scenario_event", "fault_event"),
    }

    def __init__(
        self,
        csv_path: str | Path,
        *,
        enable_ai: bool = True,
        model_name: str = DEFAULT_MINISTRAL_MODEL,
        ollama_host: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.raw_df = pd.read_csv(self.csv_path)
        self.df = self._normalize_columns(self.raw_df)
        self.ai_ready = False
        self.ollama: OllamaBackend | None = None

        if enable_ai:
            self.ollama = OllamaBackend(
                model_name=model_name,
                base_url=ollama_host,
                timeout_seconds=timeout_seconds,
            )
            try:
                self.ollama.ensure_model_ready()
                self.ai_ready = True
            except Exception as exc:
                print(f"Warning: Local AI not ready: {exc}")

    @classmethod
    def _find_column(cls, columns: Iterable[str], canonical: str) -> str | None:
        lookup = {column.lower(): column for column in columns}
        for candidate in cls.COLUMN_ALIASES[canonical]:
            resolved = lookup.get(candidate.lower())
            if resolved is not None:
                return resolved
        return None

    @classmethod
    def _normalize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        normalized = pd.DataFrame(index=df.index)
        missing_required: list[str] = []

        for canonical in ("time_min", "glucose"):
            column = cls._find_column(df.columns, canonical)
            if column is None:
                missing_required.append(canonical)
            else:
                normalized[canonical] = pd.to_numeric(df[column], errors="coerce")

        if missing_required:
            raise ValueError(
                "CSV is missing required physiological columns: "
                + ", ".join(missing_required)
                + ". Supported aliases include: "
                + str({name: cls.COLUMN_ALIASES[name] for name in missing_required})
            )

        for canonical in ("insulin_delivered", "ketones", "ffa"):
            column = cls._find_column(df.columns, canonical)
            if column is None:
                normalized[canonical] = np.nan
            else:
                normalized[canonical] = pd.to_numeric(df[column], errors="coerce")

        event_column = cls._find_column(df.columns, "event")
        normalized["event"] = df[event_column].astype(str) if event_column is not None else ""
        return normalized

    def _rate_of_change(self) -> pd.Series:
        glucose_delta = self.df["glucose"].diff()
        time_delta = self.df["time_min"].diff()
        time_delta = time_delta.where(time_delta > 0)
        return glucose_delta / time_delta

    def find_anomalies(self) -> list[Anomaly]:
        df = self.df.copy()
        df["roc"] = self._rate_of_change()
        anomalies: list[Anomaly] = []

        for idx, row in df.iterrows():
            g = float(row["glucose"]) if pd.notna(row["glucose"]) else np.nan
            roc = float(row["roc"]) if pd.notna(row["roc"]) else np.nan
            ketones = float(row["ketones"]) if pd.notna(row["ketones"]) else np.nan
            insulin = float(row["insulin_delivered"]) if pd.notna(row["insulin_delivered"]) else np.nan
            event = str(row.get("event", ""))
            flag: tuple[str, str] | None = None

            if not np.isfinite(g):
                flag = ("Non-finite glucose", "Glucose is NaN or infinite.")
            elif g < 0:
                flag = ("Negative glucose", "Negative blood glucose is mathematically impossible.")
            elif g < 20:
                flag = ("Extreme hypoglycemia", "Glucose below 20 mg/dL is a lethal/extreme edge case.")
            elif g > 800:
                flag = ("Extreme hyperglycemia", "Glucose above 800 mg/dL is outside the intended simulator envelope.")
            elif np.isfinite(roc) and abs(roc) > 15.0:
                flag = (
                    "Impossible glucose velocity",
                    f"Rate of change is {roc:.2f} mg/dL/min, above the 15 mg/dL/min red-team threshold.",
                )
            elif np.isfinite(ketones) and ketones > 15.0:
                flag = (
                    "Mathematically explosive ketones",
                    f"Ketones reached {ketones:.2f} mmol/L, above the 15 mmol/L red-team threshold.",
                )
            elif np.isfinite(insulin) and insulin < 0.0:
                flag = ("Negative insulin delivery", "Pump delivery cannot be negative.")

            if flag is not None:
                detail = flag[1]
                if event and event.lower() not in {"", "nan", "none"}:
                    detail = f"{detail} Event marker: {event}."
                anomalies.append(
                    Anomaly(
                        index=int(idx),
                        time_min=float(row["time_min"]),
                        kind=flag[0],
                        detail=detail,
                    )
                )

        return self._deduplicate(anomalies)

    @staticmethod
    def _deduplicate(anomalies: list[Anomaly], *, min_gap_minutes: float = 60.0) -> list[Anomaly]:
        deduped: list[Anomaly] = []
        last_time = -1e18
        for anomaly in anomalies:
            if anomaly.time_min - last_time >= min_gap_minutes:
                deduped.append(anomaly)
                last_time = anomaly.time_min
        return deduped

    # Backward-compatible name used by earlier scratch scripts.
    def _find_anomalies(self) -> list[dict[str, Any]]:
        return [
            {
                "index": anomaly.index,
                "time_h": anomaly.time_h,
                "type": anomaly.kind,
                "detail": anomaly.detail,
            }
            for anomaly in self.find_anomalies()
        ]

    def _window_for(self, anomaly: Anomaly, *, minutes: float = 60.0) -> pd.DataFrame:
        half = minutes / 2.0
        start = anomaly.time_min - half
        end = anomaly.time_min + half
        return self.df[(self.df["time_min"] >= start) & (self.df["time_min"] <= end)]

    def _ai_verdict(self, anomaly: Anomaly, window: pd.DataFrame) -> str:
        if not self.ai_ready or self.ollama is None:
            return ""

        system_prompt = (
            "You are an expert physiological red-team auditor reviewing a Type 1 Diabetes Digital Twin. "
            "Classify the flagged event as PHYSIOLOGICAL EXTREME or MATHEMATICAL/DATA BUG. "
            "Use the glucose, insulin, FFA, ketone, and event-marker context. "
            "Be strict: negative glucose, negative insulin, NaN values, and instantaneous impossible jumps are bugs. "
            "Keep the answer under 4 sentences."
        )
        data_str = window[["time_min", "glucose", "insulin_delivered", "ffa", "ketones", "event"]].to_string(index=False)
        user_prompt = f"Anomaly Type: {anomaly.kind}\nDetail: {anomaly.detail}\nData Window:\n{data_str}\n\nDiagnosis:"
        return self.ollama.complete(system_prompt=system_prompt, user_prompt=user_prompt)

    def run_audit(self, report_path: str | Path) -> dict[str, Any]:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        anomalies = self.find_anomalies()

        print(f"Starting AI Realism Audit on {len(self.df)} rows from {self.csv_path}...")
        print(f"Heuristic filter found {len(anomalies)} suspicious events.")

        lines: list[str] = [
            "# IINTS-AF: AI Realism Auditor Report",
            "",
            f"**Input CSV:** `{self.csv_path}`",
            f"**Rows Scanned:** {len(self.df)}",
            f"**Anomalies Detected by Filter:** {len(anomalies)}",
            f"**Local AI Verdicts:** **{'enabled' if self.ai_ready else 'offline / disabled'}**",
            "",
        ]

        if not self.ai_ready:
            lines.extend([
                "> Local Ollama is offline or disabled. The report contains deterministic heuristic results only.",
                "",
            ])

        if not anomalies:
            lines.extend([
                "## Result",
                "",
                "No red-team anomalies crossed the configured thresholds.",
                "",
            ])

        for number, anomaly in enumerate(anomalies, start=1):
            window = self._window_for(anomaly)
            lines.extend([
                f"## Anomaly {number}: {anomaly.kind} (hour {anomaly.time_h:.2f})",
                "",
                anomaly.detail,
                "",
            ])

            if self.ai_ready:
                try:
                    print(f"Querying local AI for anomaly {number}...")
                    verdict = self._ai_verdict(anomaly, window)
                except Exception as exc:
                    verdict = f"AI Error: {exc}"
                lines.extend(["### AI Verdict", "", f"> {verdict}", ""])

            lines.extend([
                "### Raw Data Window",
                "",
                "```text",
                window[["time_min", "glucose", "insulin_delivered", "ffa", "ketones", "event"]].to_string(index=False),
                "```",
                "",
                "---",
                "",
            ])

        report.write_text("\n".join(lines), encoding="utf-8")
        print(f"Audit complete. Report saved to {report}")
        return {
            "rows": len(self.df),
            "anomalies": len(anomalies),
            "ai_ready": self.ai_ready,
            "report_path": str(report),
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the IINTS-AF AI Realism Red-Team Auditor on a simulation CSV.")
    parser.add_argument("csv", type=Path, help="Input CSV from an endurance run or AdvancedMetabolicModel stress test.")
    parser.add_argument("--report", type=Path, default=Path("results/red_team/AI_REALISM_AUDIT.md"), help="Markdown report path.")
    parser.add_argument("--no-ai", action="store_true", help="Disable Ollama and run the deterministic heuristic filter only.")
    parser.add_argument("--model", default=DEFAULT_MINISTRAL_MODEL, help="Local Ollama model name.")
    parser.add_argument("--ollama-host", default=None, help="Local Ollama host, normally http://127.0.0.1:11434.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    auditor = AIRealismAuditor(
        args.csv,
        enable_ai=not args.no_ai,
        model_name=args.model,
        ollama_host=args.ollama_host,
    )
    auditor.run_audit(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
