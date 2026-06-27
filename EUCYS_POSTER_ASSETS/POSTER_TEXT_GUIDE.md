# EUCYS Poster Content Guide: IINTS-AF SDK

Deze gids is speciaal ontworpen voor je A0-poster. Een goede academische poster heeft weinig tekst (bullet points) en laat de grafieken het verhaal vertellen. Hier is exact hoe je de tekst uit je paper koppelt aan de grafieken in je `EUCYS_POSTER_ASSETS` map.

---

## 📍 Sectie 1: De Titel & Het Probleem (Links Boven)
**Titel:** IINTS-AF SDK: A Safety-First Research Platform for Transparent Diabetes Algorithm Simulation
**Ondertitel:** Exposing Unsafe AI Behavior Before Real-World Deployment

**Tekst (Gebaseerd op je Abstract & Intro):**
*   **The Black-Box Problem:** Closed-loop diabetes technologies operate in a domain where errors can be dangerous and delayed. Current algorithms often act as "black boxes".
*   **The Risk:** A controller that appears acceptable on average can still propose unsafe insulin doses during falling glucose or sensor dropouts.
*   **The Mission:** We need tools that make unsafe algorithm behavior visible *before* they are tested near real patients.

---

## 📍 Sectie 2: De Oplossing & Architectuur (Midden Boven)
**Visual:** Gebruik hier de `architecture_flowchart.pdf` (of .svg).

**Tekst (Gebaseerd op Section 4 & 6):**
*   **Separation of Authority:** The core principle of the SDK is: 
    *`Controller Proposes ➔ Safety Supervisor Verifies ➔ Simulator Applies`*
*   **Deterministic Guard:** Candidate algorithms (like AI or MPC) are completely stripped of direct dosing authority. Every proposed action is mathematically checked against transparent, physical safety limits.

---

## 📍 Sectie 3: Veiligheidsinterventie in Actie (Links Onder)
**Visual:** Gebruik hier de `poster_supervisor_intervention.pdf`.

**Tekst (Gebaseerd op Section 6: Safety Supervisor):**
*   **Dual-Guard Intervention:** The deterministic supervisor enforces hard constraints to prevent hypoglycemia.
*   **Bifurcation Risk & PD Clearance:** The supervisor calculates the "momentum trajectory" and tracks active insulin (IOB). If the AI makes a faulty prediction that causes the MPC to request too much insulin, the supervisor instantly caps or blocks the dose to prevent a crash.
*   *(Onderschrift bij grafiek)*: "Simulated scenario demonstrating the supervisor successfully overriding a dangerous 4.0U bolus request during dropping glucose."

---

## 📍 Sectie 4: Fysica-Geïnformeerde AI (Rechts Boven)
**Visual:** Gebruik hier de `poster_loss_convergence.pdf`.

**Tekst (Gebaseerd op Section 8: PINN-Style Loss):**
*   **Physics-Informed Neural Networks (PINN):** Standard AI models overfit mathematically but fail biologically. 
*   **Physiological Penalties:** The SDK trains forecasting models using custom loss functions that severely penalize impossible biological behaviors (e.g., impossible glucose rises without carbohydrates, or impossible drops without active insulin).
*   **Result:** The model is forced to respect human biology, leading to deeper, safer convergence.

---

## 📍 Sectie 5: Resultaten & Validatie (Rechts Onder)
**Visual:** Gebruik hier de `poster_error_distribution.pdf` (De Violin Plot).

**Tekst (Gebaseerd op je Training Report & Section 10):**
*   **Empirical Validation:** Models were trained and evaluated on the OhioT1DM dataset using a low-power edge workflow (Jetson Nano AutoML Factory).
*   **Metrics:** The PINN-trained glucose predictor achieved a verified RMSE of **20.31**, while restricting severe physiological constraint violations to just **5%**.
*   **Conclusion:** IINTS-AF successfully provides a reproducible, auditable ecosystem to make diabetes algorithm research safer, more transparent, and ready for rigorous discussion.

---

### 💡 Ontwerptips voor je Poster:
1.  **Gebruik altijd de .pdf of .svg versies** van de grafieken in Canva, Illustrator of PowerPoint. Ze blijven haarscherp, hoe groot je ze ook print.
2.  Zet de tekst in de blokken hierboven om in strakke bullet-points. Niemand leest lange paragrafen op een poster.
3.  Zorg voor veel 'witruimte' (ademruimte) rondom je grafieken zodat ze er echt uitspringen.
