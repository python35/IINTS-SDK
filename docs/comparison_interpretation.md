# Interpreting Glucose Forecast Results

This note explains how to read IINTS-AF glucose model comparison outputs.
It is written for research, model-card review, and jury discussions. It is not a clinical validation claim.

## 1. What The Comparison Is Trying To Answer

The comparison should not answer only: which model has the lowest average error?
For diabetes-technology research, it should also answer:

- Which model misses hypoglycemia least often?
- Which model creates the fewest physiologically impossible predictions?
- Which model degrades most gracefully at longer horizons?
- Which model remains consistent with insulin-on-board and carbs-on-board context?
- Which model is easiest to explain and audit?

- Best-by-MAE model: not bundled yet

## 2. Why MSE Can Look Best

A standard MSE model minimizes the squared forecast error:

```text
L_MSE = mean((predicted_glucose - observed_glucose)^2)
```

Under strong assumptions such as symmetric noise, independent errors, and a squared-error objective, this is a sensible estimator of average behavior. In classical linear settings, least-squares reasoning is related to the Gauss-Markov result: among linear unbiased estimators under homoscedastic uncorrelated errors, ordinary least squares has minimum variance.

That does not mean the lowest-MSE model is automatically the safest or most useful model for diabetes research. MSE treats errors symmetrically and averages over all windows. A model can have good average MAE/RMSE while still making rare but important errors around hypoglycemia, meals, sensor artifacts, or fast insulin action.

## 3. Why PINN Is Different

The IINTS physiological loss keeps the normal forecast-error term, but adds penalties when predictions violate basic physiology:

```text
L_total = L_MSE + lambda * L_physiology
```

In the current SDK implementation, the physiological penalty includes:

- impossible glucose bounds below 20 mg/dL or above 600 mg/dL
- excessive first-step glucose rate-of-change from the last observed glucose
- suspicious fast rise when insulin-on-board is high and carbs-on-board is near zero
- suspicious fast drop when carbs-on-board is present but insulin-on-board is low

This can make a PINN model trade a small amount of average error for fewer implausible or safety-relevant failures. That tradeoff is intentional: in medical-device research, a model that is slightly less optimal on average but more physiologically conservative can be more useful for simulation and safety-supervisor experiments.

The recommended `band_pinn` objective keeps the same physiological penalty, but replaces plain MSE with band-weighted MSE. That keeps extra attention on hypo and hyperglycemia windows while still discouraging impossible glucose trajectories.

## 4. Why Longer Horizons Are Harder

Short horizons, such as 15 or 30 minutes, are often dominated by recent glucose trend and sensor continuity. Longer horizons, such as 60 or 120 minutes, depend much more on delayed meal absorption, insulin pharmacodynamics, activity, stress, circadian effects, and sensor lag.

Forecast uncertainty normally grows with horizon because each future step depends on uncertain previous state evolution. In simple stochastic systems, error variance can accumulate with the number of steps; in nonlinear glucose physiology, meals, insulin, exercise, and counter-regulation can amplify this growth in a non-linear way.

For that reason, a strong result should be reported by horizon, not only as one average number. Use `horizon_metrics.csv` to check whether the model remains plausible at 30, 60, and 120 minutes.

## 5. How To Promote A Model

Do not promote a model only because it has the lowest MAE or RMSE. Use this order:

1. Reject models with private-data leakage or invalid splits.
2. Reject models with high impossible-glucose or rate-of-change violations.
3. Review missed hypoglycemia and false hypo alarms.
4. Compare horizon-specific degradation.
5. Use MAE/RMSE as final tie-breakers, not as the only decision rule.

## 6. Pitch-Friendly Explanation

A normal AI model learns to be close on average. IINTS-AF also asks whether the prediction still behaves like glucose in a human body. The PINN loss adds a mathematical penalty when the model predicts values or rates that are physiologically implausible, and the comparison report checks those errors separately from normal MAE/RMSE.

The current recommended training recipe is `band_pinn`: it asks the model to care more about clinically important low/high glucose ranges while preserving the PINN physiology guardrails.

## 7. Boundary

These metrics support research and education. They are not regulatory validation, clinical validation, or evidence that the model can be used for treatment decisions.
