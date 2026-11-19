# 05 — Safety, Fault Tolerance & Regulatory Pathways

Real-world closed-loop systems do not operate on ideal signals: sensors suffer from lag, dropouts, and pressure-induced attenuations (*compression lows*); pumps suffer from partial or complete occlusions; and users miss or misestimate meals. This section covers fault-tolerant control, supervisory safety architectures, and regulatory pathways for open-source AID software.

---

## 1. Hazards, Sensor Faults & Pump Occlusion Detection

1. **Ramkissoon, C. M., Aufderheide, B., Bequette, B. W., & Vehí, J. (2017).** *A Review of Safety and Hazards Associated With the Artificial Pancreas.* **IEEE Reviews in Biomedical Engineering**, 10, 44–62.
   - **DOI:** [10.1109/RBME.2017.2749038](https://doi.org/10.1109/RBME.2017.2749038) · **BibTeX:** `ramkissoon2017_safety`
   - **Why it matters:** Comprehensive engineering taxonomy of artificial pancreas hazards (CGM faults, infusion-set failures, software/communication errors, human factors) and mitigation strategies.

2. **Baysal, N., Cameron, F., Buckingham, B. A., Wilson, D. M., Chase, H. P., Maahs, D. M., & Bequette, B. W. (2014).** *A Novel Method to Detect Pressure-Induced Sensor Attenuations (PISA) in an Artificial Pancreas.* **Journal of Diabetes Science and Technology**, 8(6), 1091–1096.
   - **DOI:** [10.1177/1932296814553267](https://doi.org/10.1177/1932296814553267) · **BibTeX:** `baysal2014_pisa`
   - **Why it matters:** Foundational paper on real-time detection of **compression lows (PISA)** — sudden, physiologically improbable nocturnal CGM drops caused by sleeping on the sensor that can trick a controller into inappropriately suspending insulin.

3. **Meneghetti, L., Facchinetti, A., & Del Favero, S. (2021).** *Model-Based Detection and Classification of Insulin Pump Faults and Missed Meal Announcements in Artificial Pancreas Systems for Type 1 Diabetes Therapy.* **IEEE Transactions on Biomedical Engineering**, 68(1), 170–180.
   - **DOI:** [10.1109/TBME.2020.3004270](https://doi.org/10.1109/TBME.2020.3004270) · **BibTeX:** `meneghetti2021_pump_faults`
   - **Why it matters:** Distinguishes true **insulin pump faults (occlusions/disconnections)** from **unannounced meals** using physiological residual modeling — both cause unexplained hyperglycemia, yet require opposite interventions.

4. **Fault-Injection Testbeds (`APS_TestBed` / UVA-DSA):**
   - **Repository:** [github.com/UVA-DSA/APS_TestBed](https://github.com/UVA-DSA/APS_TestBed)
   - **Why it matters:** Combines closed-loop simulation with fault injection derived from FDA adverse-event reports (MAUDE/recalls) to stress-test safety monitors — closely aligned with IINTS-AF's degraded-data benchmark design ([`paper.md`](../paper.md)) and OpenFDA safety module ([`src/iints/safety/openfda_safety.py`](../src/iints/safety/openfda_safety.py)).

---

## 2. Regulatory Pathways for Open-Source AID Algorithms

5. **Braune, K., Hussain, S., & Lal, R. A. (2023).** *The First Regulatory Clearance of an Open-Source Automated Insulin Delivery Algorithm.* **Journal of Diabetes Science and Technology**, 17(5), 1139–1141.
   - **DOI:** [10.1177/19322968231164166](https://doi.org/10.1177/19322968231164166) · **BibTeX:** `braune2023_tidepool_fda`
   - **Why it matters:** Analyzes the historic January 2023 FDA 510(k) clearance of **Tidepool Loop** as an **Interoperable Automated Glycemic Controller (iAGC)** (`21 CFR 862.1356`, product code `QJI`) operating modularly with iCGMs and ACE pumps.

6. **Downey, L., O'Donnell, S., Melvin, T., & Quigley, M. (2024).** *A European regulatory pathway for Tidepool loop following clearance in the United States?* **Diabetic Medicine**, 41(2), e15246.
   - **DOI:** [10.1111/dme.15246](https://doi.org/10.1111/dme.15246) · **BibTeX:** `downey2024_eu_pathway`
   - **Why it matters:** Crucial reading for European university research: contrasts the US FDA modular iAGC/iCGM/ACE interoperability framework against the **EU Medical Device Regulation (EU MDR 2017/745)** and UK regulatory hurdles for open-source AID software.

7. **U.S. Food & Drug Administration (FDA).** *Infusion Pump Software Safety Research.*
   - **URL:** [fda.gov/medical-devices/infusion-pumps/infusion-pump-software-safety-research-fda](https://www.fda.gov/medical-devices/infusion-pumps/infusion-pump-software-safety-research-fda)
   - **Why it matters:** Reference safety-assurance methodology emphasizing formal specification, hazard analysis, and *in silico* verification prior to any hardware deployment.

---

## Mapping to IINTS-AF Governance & Safety Architecture

- **Deterministic Safety Supervisor:** [`src/iints/core/supervisor.py`](../src/iints/core/supervisor.py) and [`src/iints/core/safety/`](../src/iints/core/safety/)
- **EU Research Software Boundary & Risk Dossier:** [`docs/governance/EU_RESEARCH_SOFTWARE_COMPLIANCE.md`](../docs/governance/EU_RESEARCH_SOFTWARE_COMPLIANCE.md) and [`docs/governance/RISK_REGISTER.md`](../docs/governance/RISK_REGISTER.md)
- **OpenFDA Adverse-Event Benchmark Notes:** [`docs/OPENFDA_SAFETY_BENCHMARK.md`](../docs/OPENFDA_SAFETY_BENCHMARK.md)
