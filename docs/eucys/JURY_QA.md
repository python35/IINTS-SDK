# EUCYS Jury Q&A

Use this page as the short-answer defense pack for jury questions.

The pattern is:

- answer briefly first
- expand only if the jury asks for more
- never overclaim to sound impressive

## 1. What is the core contribution of your project?

Short answer:

> I built a transparent and reproducible benchmark platform for testing AI-guided insulin decision systems under controlled, safety-focused conditions.

Longer answer:

> The contribution is not only an algorithm. It is an evaluation framework that combines simulation, baseline comparison, safety analysis, subgroup reporting, and reproducible study bundles.

## 2. Is this a medical device?

Short answer:

> No. It is research software, not a medical product.

Longer answer:

> IINTS-AF is designed for simulation-first evaluation and benchmarking. It is not approved for clinical use, and it does not provide real-world dosing advice to patients.

## 3. Why work on simulation instead of directly testing on real patients?

Short answer:

> Because unsafe systems should be filtered out before any real-world exposure is even considered.

Longer answer:

> Diabetes decision systems affect patient safety directly. Simulation allows controlled, repeatable testing across many conditions without exposing people to experimental failures.

## 4. Why is safety such a central part of the project?

Short answer:

> Because good glucose averages are not enough if the path to those results is unsafe.

Longer answer:

> In insulin decision research, a system can look strong on time-in-range while still taking risky actions. That is why the platform tracks safety interventions, unsafe candidate actions, and the effect of the safety gate.

## 5. Why compare against classical baselines?

Short answer:

> Because a new method only matters if it is compared fairly to reasonable existing strategies.

Longer answer:

> I compare the candidate method against baseline controllers such as `PID Controller`, `Standard Pump`, and `Correction Bolus` using the same profiles, scenarios, and seeds. That makes the comparison much more honest than showing a single custom demo.

## 6. What makes the evaluation fair?

Short answer:

> Shared protocols, shared seeds, fixed profiles, fixed scenarios, and explicit study arms.

Longer answer:

> The platform writes a protocol bundle and study matrix first. Then every algorithm is evaluated on the same benchmark structure, which reduces cherry-picking and makes subgroup comparisons possible.

## 7. What does “transparent” mean in your project?

Short answer:

> The system separates forecast, candidate action, safety review, and final action so that the decision path can be inspected.

Longer answer:

> Instead of treating the whole system like a black box, IINTS-AF makes the reasoning pipeline visible and records decision artifacts for later analysis.

## 8. What does “reproducible” mean here?

Short answer:

> Another researcher should be able to rerun the same protocol and regenerate the same study structure.

Longer answer:

> The SDK records study design, seeds, outputs, and machine-readable summaries. It also exports manifests and protocol bundles so benchmark conclusions can be traced back to exact runs.

## 9. Are your patient models real patients?

Short answer:

> No. They are simulated patient profiles designed for controlled evaluation.

Longer answer:

> The profiles are useful for testing controller behavior and safety logic, but they are still abstractions. That is why I describe the project as simulation-first and do not claim direct clinical validity from the benchmark alone.

## 10. How do real datasets fit into the project?

Short answer:

> As plausibility references, not as proof that the system is clinically ready.

Longer answer:

> Public datasets can help test realism and data handling, but they do not turn a simulation benchmark into a clinical validation study. I keep that boundary explicit.

## 11. What happens when the AI is uncertain?

Short answer:

> The platform can record uncertainty and relate it to error and safety outcomes.

Longer answer:

> The project supports uncertainty-aware evaluation and calibration reporting when the controller or predictor exposes those signals. The main goal is to study whether uncertainty lines up with real risk or forecast error.

## 12. Did your candidate algorithm always win?

Short answer:

> Not necessarily, and that is important.

Longer answer:

> A serious benchmark should reveal trade-offs, not only victories. In some settings a candidate may improve one metric while worsening another, and that is part of the result.

## 13. Why deploy on Raspberry Pi or Arduino UNO Q?

Short answer:

> To show that the platform is not only a desktop notebook experiment.

Longer answer:

> Local edge deployment demonstrates that the evaluation and demo pipeline can run on modest hardware and remain usable for fairs, classroom setups, and local validation without cloud dependence.

## 14. Why does edge deployment matter scientifically?

Short answer:

> It strengthens realism and reproducibility.

Longer answer:

> Edge deployment helps test runtime constraints such as latency, persistence, kiosk behavior, and offline operation. It turns the work into a more complete systems study rather than only a modeling exercise.

## 15. What are the main limitations of the project?

Short answer:

> It is simulation-first, not clinical; it depends on benchmark design; and it cannot prove patient benefit by itself.

Longer answer:

> The conclusions are limited by the quality of the patient models, scenario design, baseline definitions, and available datasets. The project supports better preclinical evaluation, but it does not replace clinical trials, regulatory review, or medical supervision.

## 16. Why is this relevant beyond a school project?

Short answer:

> Because trustworthy AI in healthcare needs better evaluation tools, not only better models.

Longer answer:

> Many projects focus only on prediction accuracy. IINTS-AF focuses on how to test decision systems in a way that is explainable, safety-aware, reproducible, and easier to compare fairly.

## 17. What would be the next research step?

Short answer:

> Stronger benchmark results, wider profile diversity, and better calibration against realistic data.

Longer answer:

> The next steps would be expanding the benchmark, improving the realism of the digital patients, strengthening uncertainty evaluation, and linking the platform more closely to real data sources for plausibility studies.

## 18. What should the jury remember in one sentence?

Short answer:

> IINTS-AF is a transparent, safety-first benchmark platform for evaluating AI-guided insulin decision systems before real-world use is even considered.

## Delivery Tips

- If a question pushes toward medical claims, narrow the scope immediately.
- If a result sounds too perfect, explain the trade-off.
- If the jury asks “why not test on patients?”, return to safety and ethics.
- If they ask “what is original here?”, emphasize the combination of benchmarking, safety, explainability, reproducibility, and edge deployment in one platform.
