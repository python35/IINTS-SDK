"""Figures computed from the seed-fixed runs in jury_build/fresh/runs.

Run with exec() inside the python kernel so the figure-style helpers are in
scope. Every value drawn here is read from the run CSVs; nothing is typed in.
"""
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

RUNS = pathlib.Path("jury_build/fresh/runs")
FIGS = pathlib.Path("jury_build/fresh/figures")
FIGS.mkdir(parents=True, exist_ok=True)

apply_figure_style()

TARGET_LO, TARGET_HI, HYPO = 70.0, 180.0, 70.0
BAND = "#cfe3d4"
ALARM = "#c0392b"

base = pd.read_csv(RUNS / "baseline_t1d__clinical_baseline.csv")
hypo = pd.read_csv(RUNS / "hypo_prone_night__clinical_baseline.csv")
metrics = pd.read_csv(RUNS / "run_metrics.csv")


def hours(frame):
    return frame["time_minutes"] / 60.0


def target_band(ax, label=False):
    ax.axhspan(TARGET_LO, TARGET_HI, color=BAND, zorder=0,
               label="target 70-180 mg/dL" if label else None)


# ---------------------------------------------------------------- 03 physiology
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True,
                               gridspec_kw={"height_ratios": [1.25, 1]})
t = hours(base)
target_band(ax1)
ax1.plot(t, base["glucose_actual_mgdl"], color="#1f4e79", lw=1.4, label="plasma glucose (model)")
ax1.plot(t, base["glucose_to_algo_mgdl"], color="#7f8c8d", lw=0.9, ls="--",
         label="sensor value seen by controller")
meals = base.loc[base["carb_intake_grams"] > 0, ["time_minutes", "carb_intake_grams"]]
grouped = meals.groupby((meals["time_minutes"].diff().fillna(99) > 10).cumsum())
for _, chunk in grouped:
    grams = chunk["carb_intake_grams"].sum()
    x = chunk["time_minutes"].iloc[0] / 60.0
    ax1.annotate(f"{grams:.0f} g", xy=(x, 262), ha="center", va="bottom", fontsize=6, color="#8a5a00")
    ax1.axvline(x, color="#e0a800", lw=0.8, alpha=0.7, zorder=1)
ax1.set_ylabel("glucose (mg/dL)")
ax1.set_title("A full simulated day: four meals, exercise and stress drive the glucose the controller sees")
ax1.legend(loc="upper left", fontsize=6)
ax1.margins(x=0.01)

flux_map = {
    "patient_flux_glucose_appearance": "glucose appearance from gut",
    "patient_flux_basal_production": "hepatic glucose production",
    "patient_flux_glucose_uptake": "insulin-driven uptake",
    "patient_flux_exercise_uptake": "exercise uptake",
    "patient_flux_renal_clearance": "renal clearance",
}
names = list(flux_map.values())
colors = dict(zip(names, focal_palette(names, "glucose appearance from gut", "#1f4e79", other="muted")))
for column, label in flux_map.items():
    ax2.plot(t, base[column], lw=1.1, color=colors[label], label=label)
ax2.set_ylabel("flux (mg/dL/min)")  # unit declared by the SDK compartment schema
ax2.set_xlabel("time (hours from midnight)")
ax2.set_title("The same day inside the model: the ODE fluxes that produce that curve")
ax2.legend(loc="upper left", fontsize=6, ncol=2)
ax2.set_xticks(range(0, 25, 3))
ax2.margins(x=0.01)
for ax, letter in ((ax1, "a"), (ax2, "b")):
    panel_letter(ax, letter, dx=-0.08)
fig.tight_layout()
fig.savefig(FIGS / "physiology_full_day.png", dpi=300, bbox_inches="tight")

# ---------------------------------------------------------------- 02 pump
# Column semantics verified in src/iints/core/simulator.py:
#   bolus_insulin_units / basal_insulin_units  = what the algorithm COMMANDS (insulin_output)
#   delivered_insulin_units                    = what remains after the safety supervisor
#                                                (approved_insulin) and the pump model
commanded = base["bolus_insulin_units"] + base["basal_insulin_units"]
delivered = base["delivered_insulin_units"]
withheld = float((commanded - delivered).clip(lower=0).sum())
n_clipped = int((commanded - delivered > 1e-9).sum())
total_bolus = float(base["bolus_insulin_units"].sum())
total_basal = float(base["basal_insulin_units"].sum())

fig2, (bx1, bx2, bx3, bx4) = plt.subplots(
    4, 1, figsize=(7.2, 8.0), sharex=True,
    gridspec_kw={"height_ratios": [1, 0.7, 1, 1.2]})

bolus = base[base["bolus_insulin_units"] > 0]
bx1.bar(hours(bolus), bolus["bolus_insulin_units"], width=0.10, color="#e0a800")
bx1.set_ylabel("bolus (U)")
bx1.set_title(f"Commanded boluses: {len(bolus)} doses, {total_bolus:.1f} U requested")
bx1.margins(x=0.01)

bx2.plot(t, base["basal_insulin_units"], color="#1f4e79", lw=1.0, drawstyle="steps-post")
bx2.set_ylabel("basal per\n5 min (U)")
bx2.set_ylim(0, float(base["basal_insulin_units"].max()) * 1.6)  # from zero: the rate is constant, not noisy
bx2.set_title(f"Commanded basal, own scale: {total_basal:.1f} U requested, "
              f"constant {base['basal_insulin_units'].max():.3f} U per 5 min")
bx2.margins(x=0.01)

gap = (commanded - delivered).clip(lower=0)
bx3.bar(t, gap, width=0.06, color=ALARM, label="withheld this step (U)")
bx3.set_ylabel("withheld per\n5 min (U)")
bx3b = bx3.twinx()
bx3b.plot(t, gap.cumsum(), color="#1f4e79", lw=1.3, label="cumulative withheld (U)")
bx3b.set_ylabel("cumulative\nwithheld (U)")
bx3b.set_ylim(0, float(gap.sum()) * 1.15)
bx3.set_title(f"The supervisor is not decoration: {withheld:.1f} U of {float(commanded.sum()):.1f} U "
              f"withheld on {n_clipped} of {len(base)} steps")
bx3.legend(bx3.get_legend_handles_labels()[0] + bx3b.get_legend_handles_labels()[0],
           bx3.get_legend_handles_labels()[1] + bx3b.get_legend_handles_labels()[1],
           loc="upper left", fontsize=6)
bx3.margins(x=0.01)

bx4.plot(t, base["patient_iob_units"], color="#1f4e79", lw=1.4, label="insulin on board (U)")
bx4.set_ylabel("insulin on\nboard (U)")
bx4b = bx4.twinx()
bx4b.plot(t, base["patient_flux_insulin_depot_transfer"], color="#9a9a9a", lw=1.0, ls="--",
          label="transfer out of subcutaneous depot (mU/min)")
bx4b.plot(t, base["patient_flux_insulin_appearance"], color="#b24ca8", lw=1.0,
          label="appearance in plasma (mU/min)")
bx4b.set_ylabel("insulin flux (mU/min)")  # unit declared by the SDK compartment schema
handles = bx4.get_legend_handles_labels()[0] + bx4b.get_legend_handles_labels()[0]
labels = bx4.get_legend_handles_labels()[1] + bx4b.get_legend_handles_labels()[1]
bx4.legend(handles, labels, loc="upper left", fontsize=6)
bx4.set_title("Delivery is not action: the depot delays every delivered unit before it reaches plasma")
bx4.set_xlabel("time (hours from midnight)")
for ax, letter in ((bx1, "a"), (bx2, "b"), (bx3, "c"), (bx4, "d")):
    panel_letter(ax, letter, dx=-0.08)
fig2.tight_layout()
fig2.savefig(FIGS / "pump_delivery_and_iob.png", dpi=300, bbox_inches="tight")

# ---------------------------------------------------------------- 04 controllers
controllers = {
    "clinical_baseline": "Clinical Baseline",
    "mpc": "MPC",
    "pid": "PID",
    "fixed_basal_bolus": "Fixed basal-bolus",
}
fig3, (cx1, cx2) = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.45, 1]})
cnames = list(controllers.values())
palette = dict(zip(cnames, focal_palette(cnames, "PID", ALARM, other="muted")))
target_band(cx1)
ends = []
for key, label in controllers.items():
    frame = pd.read_csv(RUNS / f"baseline_t1d__{key}.csv")
    tt = hours(frame)
    cx1.plot(tt, frame["glucose_actual_mgdl"], lw=1.1, color=palette[label], label=label)
    ends.append((tt.iloc[-1], frame["glucose_actual_mgdl"].iloc[-1], label))
cx1.axhline(HYPO, color=ALARM, lw=0.8, ls=":")
cx1.annotate("70 mg/dL", xy=(19.5, HYPO), xytext=(19.5, HYPO + 6), fontsize=6, color=ALARM)
end_of_line_labels(cx1, [[e[0]] for e in ends], [[e[1]] for e in ends], [e[2] for e in ends],
                   colors=[palette[e[2]] for e in ends], fontsize=6)
cx1.set_xlabel("time (hours from midnight)")
cx1.set_ylabel("glucose (mg/dL)")
cx1.set_title("Four shipped controllers, identical day and seed")
cx1.set_xticks(range(0, 25, 6))
cx1.margins(x=0.06)

sub = metrics[metrics["preset"] == "baseline_t1d"].copy()
sub["label"] = sub["controller"].map(controllers)
cx2.scatter(sub["tir_70_180"], sub["tir_below_70"], s=42,
            color=[palette[l] for l in sub["label"]], zorder=3)
offsets = {"Fixed basal-bolus": (0, 12), "MPC": (0, 24), "Clinical Baseline": (0, 36),
           "PID": (-8, 10)}
for _, row in sub.iterrows():
    cx2.annotate(row["label"], xy=(row["tir_70_180"], row["tir_below_70"]),
                 xytext=offsets[row["label"]], textcoords="offset points", fontsize=6,
                 ha="center", color=palette[row["label"]],
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=palette[row["label"]]))
cx2.set_xlabel("time in range 70-180 mg/dL (%)")
cx2.set_ylabel("time below 70 mg/dL (%)")
cx2.set_title("The best time in range is also the only\ncontroller that goes hypoglycaemic")
cx2.margins(0.18)
goodness_arrow(cx2, "higher = better", loc="lower right", axis="x")
goodness_arrow(cx2, "lower = better", loc="upper left", axis="y")
for ax, letter in ((cx1, "a"), (cx2, "b")):
    panel_letter(ax, letter, dx=-0.14)
fig3.tight_layout()
fig3.savefig(FIGS / "controller_comparison.png", dpi=300, bbox_inches="tight")

# ---------------------------------------------------------------- 08 failure case
fig4, (dx1, dx2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True,
                                gridspec_kw={"height_ratios": [1.3, 0.7]})
th = hours(hypo)
target_band(dx1)
dx1.plot(th, hypo["glucose_actual_mgdl"], color="#1f4e79", lw=1.4)
below = hypo["glucose_actual_mgdl"] < HYPO
dx1.fill_between(th, 0, 400, where=below, color=ALARM, alpha=0.13, zorder=0)
dx1.axhline(HYPO, color=ALARM, lw=0.8, ls=":")
low = hypo["glucose_actual_mgdl"].min()
tlow = th[hypo["glucose_actual_mgdl"].idxmin()]
dx1.annotate(f"nadir {low:.0f} mg/dL", xy=(tlow, low), xytext=(tlow + 0.6, low - 18),
             fontsize=6.5, color=ALARM,
             arrowprops=dict(arrowstyle="-", lw=0.7, color=ALARM))
dx1.set_ylim(40, 210)
dx1.set_ylabel("glucose (mg/dL)")
dx1.set_title("A hypoglycaemia-prone night: the run the SDK is meant to expose, not hide")
dx1.margins(x=0.01)

hypo_cmd = hypo["bolus_insulin_units"] + hypo["basal_insulin_units"]
hypo_withheld = float((hypo_cmd - hypo["delivered_insulin_units"]).clip(lower=0).sum())
dx2.bar(th, hypo["delivered_insulin_units"], width=0.06, color="#1f4e79", label="insulin delivered")
dx2.set_title(f"The supervisor withheld {hypo_withheld:.1f} U of {float(hypo_cmd.sum()):.1f} U commanded "
              f"and the run still reached {float(hypo["glucose_actual_mgdl"].min()):.0f} mg/dL")
triggered = hypo[hypo["safety_triggered"].astype(bool)]
dx2.plot(hours(triggered), [0] * len(triggered), marker="|", ls="none", ms=7,
         color=ALARM, label=f"safety supervisor acted ({len(triggered)} steps)")
dx2.set_ylabel("insulin per\n5 min (U)")
dx2.set_xlabel("time (hours from simulation start)")
dx2.legend(loc="upper right", fontsize=6)
dx2.margins(x=0.01)
for ax, letter in ((dx1, "a"), (dx2, "b")):
    panel_letter(ax, letter, dx=-0.08)
fig4.tight_layout()
fig4.savefig(FIGS / "hypo_failure_case.png", dpi=300, bbox_inches="tight")

print("figures written:", sorted(p.name for p in FIGS.glob("*.png")))
print("meals detected:", len(grouped), "| hypo nadir:", round(float(low), 1),
      "| supervisor steps in hypo run:", int(len(triggered)))
