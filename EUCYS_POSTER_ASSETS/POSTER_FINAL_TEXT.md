# IINTS-AF SDK: Final Poster Text Blocks

Dit document bevat de **exacte, definitieve Engelstalige teksten** voor je poster, geformatteerd volgens de perfecte academische 5-stappen structuur. Je kunt deze blokken direct kopiëren en plakken in je ontwerpprogramma.

---

## 📌 HEADER

**Title:**  
IINTS-AF SDK: An Open-Source Digital Twin Platform for Diabetes Algorithm Research

**Subtitle:**  
Studying, comparing and explaining diabetes algorithms in a safe simulated environment

**Disclaimer (kleine lettertjes bovenaan of onderaan):**  
*Research and education software only — not a medical device and not intended for treatment decisions.*

---

## 1. ABSTRACT

IINTS-AF SDK is an open-source research and education platform for virtual Type 1 diabetes simulation, algorithm benchmarking, deterministic safety supervision and physiology-aware glucose forecasting. It was developed to provide a transparent environment in which diabetes algorithms can be studied before real-world use.

**Research question:**  
Can an open-source simulation SDK help researchers and students study diabetes algorithms transparently before real-world use?

---

## 2. MATERIALS

The SDK combines several research components:

*   **Virtual patient models:** custom transparent model, Bergman-style ODE model, and Hovorka-style ODE model
*   **Device models:** CGM lag, noise, drift, dropout, compression-low effects, and pump-delivery abstractions
*   **Candidate algorithms:** PID, MPC, AI-based or custom research controllers
*   **Safety layer:** deterministic supervision of proposed actions
*   **Reporting layer:** AGP-style reports, safety reports, manifests and explainable event logs
*   **AI workflow:** glucose forecasting with IOB/COB feature extraction and physiology-aware loss functions

---

## 3. METHODOLOGY

*(Verdeel dit visueel in 3 duidelijke blokken op je poster)*

**A. Simulation Architecture**  
*(Plaats hier de `step_sequence` of `architecture_flowchart` grafiek)*  
Each simulation run begins with a configurable scenario, including patient profile, meals, exercise, stress, sensor events, controller settings and safety parameters. The virtual patient produces latent glucose, the sensor model converts it into a CGM-like observation, the algorithm proposes an action, and the deterministic supervisor validates the action before it is applied.

**B. Separation of Authority**  
**Core design principle:** The algorithm *proposes* an action, but only the *supervised* action is applied in simulation. This separation makes algorithm behaviour easier to study, compare and audit.

**C. AI & Physiology-Aware Evaluation**  
*(Plaats hier de PINN loss / convergence grafiek)*  
**$L = MSE(Y, \hat{Y}) + \lambda L_{phys}$**  
The forecasting model is trained not only for numerical accuracy, but also for physiological plausibility. This means impossible biological glucose behaviour is heavily penalized during training.

**D. Safety Momentum Guard (Example)**  
*(Plaats hier de `poster_supervisor_intervention` grafiek)*  
**$G_{momentum,30} = G_t + 30v_G$**  
The supervisor calculates a future trajectory. If the momentum crosses a severe-low threshold, it overrides the AI/MPC and blocks the insulin request.

---

## 4. RESULTS

The result is a reproducible SDK for studying diabetes algorithms under controlled virtual conditions. *(Plaats hier de `poster_error_distribution` violin plot)*

*   **Simulation result:** candidate controllers can be evaluated across meals, exercise, stress, sensor errors and other realistic scenario variations.
*   **Transparency result:** each run produces auditable artifacts, including time-series outputs, safety reports, manifests and AGP-style summaries.
*   **Forecasting result:** physiology-aware models achieved an RMSE of 20.31, restricting physiological constraint violations to just 5%. Models can be evaluated not only by error metrics, but also by their biological plausibility under clinically relevant conditions.

---

## 5. CONCLUSION

IINTS-AF SDK shows that diabetes algorithms should not be studied only through average prediction error. They should also be examined through physiology, safety supervision, reproducibility, explainability and realistic scenario testing.

By combining virtual patients, deterministic supervision, auditable reporting and physiology-aware forecasting, the SDK provides a transparent environment for diabetes algorithm research and education.

Its value is not treatment recommendation, but understanding: helping researchers and students study algorithm behaviour before real-world use.
