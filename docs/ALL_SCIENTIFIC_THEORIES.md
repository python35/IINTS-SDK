# Complete Scientific Theories in IINTS-AF

This document provides a comprehensive list of every scientific, physiological, and mathematical theory encoded within the core simulation engine of the IINTS-AF Digital Twin SDK. This serves as the ultimate scientific foundation for the EUCYS presentation.

The simulation engine is primarily built on the `AdvancedMetabolicModel`, an extended 16-state differential equation solver that upgrades the classical Bergman Minimal Model into a full-scale clinical Type 1 Diabetes Digital Twin.

---

## 1. The Bergman Minimal Model
**Foundation:** The core system relies on the classical Bergman minimal model (1989) which uses a 3-compartment design to track Plasma Glucose ($G$), Plasma Insulin ($I$), and Insulin Action / Remote Compartment ($X$).
**Key Feature:** It elegantly defines Glucose Effectiveness ($p_1$) and Insulin Sensitivity ($p_3$).

## 2. Multi-Compartment Gastric Emptying (Dalla Man)
**Foundation:** Rather than assuming carbohydrates magically enter the blood, the SDK implements the Dalla Man (2006) gut absorption models.
**Key Feature:** Uses three compartments—Solid Stomach ($Q_{sto1}$), Liquid Stomach ($Q_{sto2}$), and Intestine ($Q_{gut}$)—to simulate the complex physiological delays in the Glucose Rate of Appearance ($R_a$).

## 3. Subcutaneous Insulin Absorption Kinetics
**Foundation:** Models from Hovorka and Dalla Man.
**Key Feature:** Tracks the delayed absorption of insulin from a physical pump infusion site into the plasma. It uses two subcutaneous compartments ($S_1$, $S_2$) to mathematically represent the physiological lag (tau) between insulin delivery and its metabolic availability.

## 4. Lipotoxicity & Free Fatty Acid (FFA) Dynamics
**Foundation:** Advanced pathophysiology of insulin resistance.
**Key Feature:** Insulin is a powerful inhibitor of lipolysis. In a T1D patient with zero insulin, adipose tissue massively releases Free Fatty Acids ($F$). The SDK mathematically tracks FFA concentration, which in turn causes severe, compounding insulin resistance by down-regulating the $p_3$ parameter (Lipotoxicity).

## 5. Ketogenesis & Diabetic Ketoacidosis (DKA)
**Foundation:** The lethal cascade of insulin deficiency.
**Key Feature:** High FFA combined with near-zero insulin triggers hepatic ketogenesis. The SDK models the rate of Ketone body ($K$) production, allowing the AI to detect the exact onset of Diabetic Ketoacidosis—a deadly state for T1D patients.

## 6. Hypoglycemia-Associated Autonomic Failure (HAAF)
**Foundation:** Cryer's theory of defective counter-regulation.
**Key Feature:** Repeated episodes of hypoglycemia desensitize the brain's autonomic response. The SDK includes a "HAAF Memory" differential equation ($HAAF$), tracking past hypos and mathematically suppressing the body's natural adrenaline rescue mechanisms in future hypos.

## 7. Circadian Rhythms & Dawn Phenomenon
**Foundation:** Chronobiology and counter-regulatory morning hormones (Cortisol, Growth Hormone).
**Key Feature:** Insulin sensitivity is not static. The SDK models the *Dawn Phenomenon* by applying a continuous sinusoidal wave to Endogenous Glucose Production (EGP), peaking around 05:00 AM, causing natural, unexplained morning hyperglycemia.

## 8. Physiological Renal Glucose Clearance (RGC)
**Foundation:** Kidney filtration physics.
**Key Feature:** When plasma glucose exceeds the Renal Threshold for Glucose (RTG, typically ~180 mg/dL), the kidneys begin excreting glucose into the urine (glycosuria). The SDK mathematically models this as a dynamic glucose sink, acting as a natural brake on infinite extreme hyperglycemia.

## 9. Exercise Physiology & Stress
**Foundation:** Metabolic shifts during physical exertion.
**Key Feature:** Introduces an Exercise Intensity parameter ($E$) that dynamically increases insulin-independent glucose uptake by skeletal muscles, while simultaneously massively increasing insulin sensitivity, successfully simulating exercise-induced hypoglycemia.

## 10. Residual Beta-Cell Autoimmune Decay
**Foundation:** The T1D "Honeymoon Phase".
**Key Feature:** newly diagnosed patients still produce trace amounts of endogenous insulin. The SDK tracks the residual Beta-cell mass fraction ($\beta$), which undergoes exponential autoimmune decay over time, gradually shifting the patient from easy control to brittle diabetes.

## 11. Exogenous Glucagon Kinetics
**Foundation:** Emergency hormonal rescue.
**Key Feature:** Simulates the pharmacokinetics of a glucagon injection (via compartments $Y_1, Y_2, \Gamma$) and its immediate physiological trigger on hepatic glycogen release to rescue the digital twin from severe hypoglycemia.

## 12. Multi-Macronutrient Gastric Emptying
**Foundation:** Advanced meal composition modeling.
**Key Feature:** Moving beyond simple carbohydrate counting, the SDK models fat and protein intake ($Q_{fat}, Q_{prot}$). Fat intake applies a massive exponential delay to the gastric emptying rate ($k_{emp}$), trapping food in the stomach longer. Protein slowly undergoes gluconeogenesis, feeding a delayed glucose rate of appearance ($R_{a,prot}$) up to 6 hours later.

## 13. Cannula Degradation & Lipohypertrophy
**Foundation:** Mechanical tissue resistance.
**Key Feature:** In the real world, insulin pump sites degrade over time. The SDK tracks the age of the inserted cannula. After 48 hours, the subcutaneous insulin absorption constants ($k_{a1}, k_{a2}$) begin to degrade linearly by up to 30%, mathematically simulating tissue scarring (lipohypertrophy) and causing late-stage insulin resistance in endurance simulations.

## 14. Menstrual Cycle Hormonal Drifts
**Foundation:** Female biology and long-term hormonal cyclical resistance.
**Key Feature:** The SDK overlays a 28-day low-frequency sinusoidal wave onto the base insulin sensitivity parameter ($p_3$). This perfectly models the massive fluctuations in insulin requirements across the Follicular and Luteal phases, enabling true month-long algorithmic stress testing for female patients.

## 15. Acute Illness & Cytokine Resistance
**Foundation:** Immune system stress response.
**Key Feature:** When the digital twin enters a state of acute illness (e.g., flu or bacterial infection), the release of cytokines triggers a massive spike in counter-regulatory stress hormones. The SDK simulates this by violently suppressing insulin sensitivity ($p_3$) while simultaneously spiking Basal Glucose Production ($Gb$), acting as the ultimate catalyst for spontaneous Diabetic Ketoacidosis.

---

### Conclusion
By blending these **15 isolated physiological theories** into a single cohesive, interacting 18-state mathematical framework, IINTS-AF successfully creates a true-to-life Digital Twin capable of predicting real-world biological chaos.
