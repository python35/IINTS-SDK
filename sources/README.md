# Closed-Loop Insulin Pump (AID) Research — Literature & Source Library

This directory collects and orders the foundational and state-of-the-art scientific literature for university research on **open-source models of closed-loop Automated Insulin Delivery (AID) algorithms**.

> **Research-only boundary:** These references support academic modeling, *in silico* benchmarking, and software verification. Listing clinical trials or regulatory clearances does not imply that IINTS-AF is a medical device or suitable for insulin dosing or treatment decisions.

---

## Directory Contents (Ordered by Topic)

| File | Topic | Focus |
| :--- | :--- | :--- |
| [`01_commercial_aid_algorithms.md`](01_commercial_aid_algorithms.md) | **Commercial AID Systems & Control Laws** | Published mathematical foundations of Medtronic SmartGuard (PID-IFB), Tandem Control-IQ (Zone-MPC), CamAPS FX (Hovorka NMPC), Omnipod 5, and iLet Bionic Pancreas. |
| [`02_open_source_aid_and_clinical_trials.md`](02_open_source_aid_and_clinical_trials.md) | **Open-Source AID & Clinical Evidence** | OpenAPS (`oref0`/`oref1`), AndroidAPS, Trio, Loop (`LoopKit`), Tidepool Loop, the CREATE RCT, international consensus statements, and real-world systematic reviews. |
| [`03_physiology_simulators_and_digital_twins.md`](03_physiology_simulators_and_digital_twins.md) | **Physiology, Simulators & Digital Twins** | Bergman & Hovorka ODEs, Dalla Man meal model, UVA/Padova T1D Simulator, `simglucose`, `ReplayBG`, exercise/circadian/counterregulation dynamics, and consensus glycemic metrics. |
| [`04_control_theory_and_ai_rl.md`](04_control_theory_and_ai_rl.md) | **Control Engineering, AI & Reinforcement Learning** | Algorithm reviews (PID, MPC, Fuzzy Logic, RL), Deep RL & Offline RL benchmarks, Dual-Hormone AP, CGM foundation models (`GlucoFM`), and Fully Closed-Loop challenges. |
| [`05_safety_fault_tolerance_and_regulation.md`](05_safety_fault_tolerance_and_regulation.md) | **Safety, Fault Tolerance & Regulation** | Artificial pancreas hazard taxonomies, compression low (PISA) detection, pump occlusion detection, safety supervisors, FDA iAGC (Tidepool Loop), and EU MDR pathways. |
| [`06_datasets_and_benchmarks.md`](06_datasets_and_benchmarks.md) | **Datasets & Evaluation Benchmarks** | Public and DUA-governed T1D datasets (OhioT1DM, T1DEXI, OpenAPS Data Commons, DiaTrend, AZT1D, HUPA-UCM, DCLP3/iDCL, MetaboNet, Glucose-ML) and split hygiene. |
| [`references.bib`](references.bib) | **Verified BibTeX Bibliography** | Ready-to-use BibTeX entries for LaTeX, Overleaf, or Zotero, ordered by the same six categories. |

---

## Recommended Reading Order ("Start Here")

If you are onboarding a student, co-author, or supervisor to this research track, read in this order:

1. **Landscape & engineering overview:** Doyle III (2014) — *Closed-Loop Artificial Pancreas Systems: Engineering the Algorithms* (`04_control_theory_and_ai_rl.md`).
2. **Open-source AID consensus & proof of concept:** Braune et al. (2022, *Lancet Diabetes Endocrinol.*) and Burnside et al. (2022, *NEJM*, CREATE trial) (`02_open_source_aid_and_clinical_trials.md`).
3. **Commercial algorithm baselines:** Steil et al. (2011, PID-IFB), Grosman et al. (2010, Zone-MPC), Hovorka et al. (2004, NMPC), Brown et al. (2019, Control-IQ), Russell et al. (2022, iLet) (`01_commercial_aid_algorithms.md`).
4. **In-silico validation & Digital Twins:** Dalla Man et al. (2014, UVA/Padova), Cappon et al. (2023, `ReplayBG`), Battelino et al. (2019, ATTD TIR consensus), Kovatchev et al. (1997/1998) (`03_physiology_simulators_and_digital_twins.md`).
5. **Real-world faults, safety & regulatory translation:** Ramkissoon et al. (2017), Baysal et al. (2014, PISA), Meneghetti et al. (2021, pump faults), Braune et al. (2023, Tidepool Loop FDA iAGC), Downey et al. (2023, EU pathway) (`05_safety_fault_tolerance_and_regulation.md`).
6. **Modern AI/RL & open frontiers:** Tejedor et al. (2020), Fox et al. (2020), Emerson et al. (2023, Offline RL), Li et al. (2026, `GlucoFM`) (`04_control_theory_and_ai_rl.md`).

---

## Relation to Broader SDK Documentation

- This `sources/` folder is curated specifically for **closed-loop insulin pump algorithm research**.
- For the full software-wide bibliography (including structural biology, proteomics, SBML/CellML/FMI standards, and desktop tooling), see [`docs/SDK_BIBLIOGRAPHY.md`](../docs/SDK_BIBLIOGRAPHY.md) and [`docs/SOURCE_LIBRARY.md`](../docs/SOURCE_LIBRARY.md).
- For notes on the repository's best-effort commercial algorithm emulators, see [`docs/emulation_references.md`](../docs/emulation_references.md) and [`src/iints/emulation/`](../src/iints/emulation/).
