# Complete Scientific Theories in IINTS-AF

This document provides a comprehensive list of every scientific, physiological, and mathematical theory encoded within the core simulation engine of the IINTS-AF Digital Twin SDK. This serves as the ultimate scientific foundation for the EUCYS presentation.

The simulation engine is primarily built on the `AdvancedMetabolicModel`, an extended 18-state differential equation solver that upgrades the classical Bergman Minimal Model into a full-scale clinical Type 1 Diabetes Digital Twin.

---

## 1. The Bergman Minimal Model
**Foundation:** The core system relies on the classical Bergman minimal model (1989) which uses a 3-compartment design to track Plasma Glucose ($G$), Plasma Insulin ($I$), and Insulin Action ($X$).
**Mathematics:**
$$ \frac{dG}{dt} = -X \cdot G + p_1 \cdot G_b + R_a $$
$$ \frac{dX}{dt} = -p_2 \cdot X + p_3 \cdot \max(I - I_b, 0) $$
*Here, $p_1$ represents glucose effectiveness (GEZI) and $p_3$ represents insulin sensitivity.*

## 2. Multi-Compartment Gastric Emptying (Dalla Man)
**Foundation:** The Dalla Man (2006) absorption model.
**Mathematics:** Rather than assuming immediate absorption, carbohydrates flow through three delayed compartments:
$$ \frac{dQ_{sto1}}{dt} = -k_{gri} \cdot Q_{sto1} + D_{carbs} $$
$$ \frac{dQ_{sto2}}{dt} = k_{gri} \cdot Q_{sto1} - k_{emp} \cdot Q_{sto2} $$
$$ \frac{dQ_{gut}}{dt} = k_{emp} \cdot Q_{sto2} - k_{abs} \cdot Q_{gut} $$

## 3. Subcutaneous Insulin Absorption Kinetics
**Foundation:** Hovorka / Dalla Man pharmacokinetics.
**Mathematics:** Tracks the physiological lag ($\tau$) between pump delivery ($u_{ins}$) and plasma appearance via two subcutaneous compartments:
$$ \frac{dS_1}{dt} = u_{ins} - k_a \cdot S_1 $$
$$ \frac{dS_2}{dt} = k_a \cdot S_1 - k_a \cdot S_2 $$
$$ R_{a, I} = k_a \cdot S_2 $$

## 4. Lipotoxicity & Free Fatty Acid (FFA) Dynamics
**Foundation:** Advanced pathophysiology of insulin resistance.
**Mathematics:** Insulin inhibits lipolysis. Without insulin, FFA ($F$) rises and down-regulates insulin sensitivity ($p_3$):
$$ \frac{dF}{dt} = l_0 \cdot e^{-l_1 I} - k_f F $$
$$ p_{3, eff} = p_3 \times \frac{0.4}{\max(0.4, F)} $$

## 5. Ketogenesis & Diabetic Ketoacidosis (DKA)
**Foundation:** The lethal cascade of insulin deficiency.
**Mathematics:** Ketone ($K$) production is driven by extreme FFA levels and near-zero insulin:
$$ \frac{dK}{dt} = k_0 \cdot F \cdot e^{-k_1 I} - k_2 K $$

## 6. Hypoglycemia-Associated Autonomic Failure (HAAF)
**Foundation:** Cryer's theory of defective counter-regulation.
**Mathematics:** Tracks "hypo-memory". Past hypos suppress future adrenaline rescue mechanisms:
$$ \Delta_{hypo} = \max(0, 70-G) $$
$$ \frac{dHAAF}{dt} = k_{build} \cdot \Delta_{hypo} \cdot (1-HAAF) - k_{decay} \cdot HAAF $$

## 7. Circadian Rhythms & Dawn Phenomenon
**Foundation:** Chronobiology and counter-regulatory morning hormones.
**Mathematics:** Applies a continuous sinusoidal wave to Endogenous Glucose Production (EGP), peaking around 05:00 AM:
$$ \text{Circadian} = 1.0 + A_{circ} \cdot \cos\left(\frac{2\pi}{24} (t_{hours} - 5)\right) $$

## 8. Physiological Renal Glucose Clearance (RGC)
**Foundation:** Kidney filtration physics.
**Mathematics:** When glucose exceeds the Renal Threshold (~180 mg/dL), kidneys excrete it. Modeled via a softplus function to prevent stiff-ODE crashes:
$$ RGC = c_{renal} \cdot 10 \cdot \ln\left(1 + \exp\left(\frac{G - 180}{10}\right)\right) $$

## 9. Exercise Physiology & Stress
**Foundation:** Metabolic shifts during physical exertion.
**Mathematics:** Exercise intensity ($E$) massively amplifies insulin sensitivity ($p_3$) and drives insulin-independent muscle uptake:
$$ p_{3,eff} = p_3 \cdot (1 + 2E) $$
$$ \text{Uptake}_{muscle} = E \cdot 0.005 \cdot G $$

## 10. Residual Beta-Cell Autoimmune Decay
**Foundation:** The T1D "Honeymoon Phase".
**Mathematics:** The residual healthy Beta-cell mass fraction ($\beta$) undergoes exponential autoimmune decay:
$$ \frac{d\beta}{dt} = -\alpha \cdot \beta $$
$$ \text{Secretion}_{endo} = \beta \cdot \gamma \cdot \max(G - h, 0) $$

## 11. Exogenous Glucagon Kinetics
**Foundation:** Emergency hormonal rescue pharmacokinetics.
**Mathematics:** Simulates glucagon transport ($\Gamma$) causing direct hepatic glycogen release:
$$ \frac{dY_1}{dt} = u_{gluc} - \frac{Y_1}{\tau} \quad \rightarrow \quad \frac{d\Gamma}{dt} = \frac{Y_2}{\tau \cdot V_{gluc}} - k_e \Gamma $$

## 12. Multi-Macronutrient Gastric Emptying
**Foundation:** Advanced meal composition (Fat & Protein).
**Mathematics:** Fat ($Q_{fat}$) exponentially delays gastric emptying ($k_{emp}$). Protein ($Q_{prot}$) triggers slow gluconeogenesis:
$$ k_{emp,eff} = \frac{1}{\tau_{meal}} \cdot e^{-0.02 \cdot Q_{fat}} $$
$$ R_{a,prot} = 0.5 \cdot k_{prot} \cdot Q_{prot} $$

## 13. Cannula Degradation & Lipohypertrophy
**Foundation:** Mechanical tissue resistance & inflammation.
**Mathematics:** Subcutaneous absorption ($k_a$) degrades linearly by up to 30% after wearing the pump for 48 hours (2880 mins):
$$ k_{a,eff} = k_a \cdot \left(1 - 0.3 \cdot \max\left(0, \frac{t - 2880}{2880}\right)\right) $$

## 14. Menstrual Cycle Hormonal Drifts
**Foundation:** Female biology and cyclical resistance.
**Mathematics:** A 28-day low-frequency sinusoidal wave alters insulin sensitivity ($p_3$), peaking in resistance during the Luteal phase:
$$ p_{3,eff} = p_3 \cdot \left(1 - 0.25 \cdot \sin\left(\frac{2\pi \cdot t}{28 \cdot 24 \cdot 60}\right)\right) $$

## 15. Acute Illness & Cytokine Resistance
**Foundation:** Immune system stress response.
**Mathematics:** A sickness severity factor ($\zeta$) violently spikes Basal Glucose Production ($G_b$) while crushing tissue sensitivity:
$$ Gb_{eff} = Gb \cdot (1 + 0.8 \cdot \zeta) $$
$$ p_{3,eff} = p_3 \cdot (1 - 0.5 \cdot \zeta) $$

---

### Conclusion
By blending these **15 isolated physiological theories** into a single cohesive, interacting 18-state mathematical framework, IINTS-AF successfully creates a true-to-life Digital Twin capable of predicting real-world biological chaos.
