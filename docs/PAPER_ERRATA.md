# Errata: technical paper vs. shipped code

Corrections to the 14-page technical paper (`paper/paper-SDK.pdf`), established
by reading the paper against the source at SDK v1.5.37. Each entry gives what
the paper states, what the code does, and which is authoritative.

**The code is authoritative in every case below.** These are documentation
defects, not code defects.

---

## E1 — The physiology-informed loss has two terms the code does not compute

**Paper.** §8.2, equations (56)–(58) define IOB/COB consistency penalties:

```
L_rise = 2·E[max(0, ROC − 2)·1{IOB>1}·1{COB<5}]      (56)
L_drop = 2·E[max(0, −ROC − 2)·1{COB>10}·1{IOB<0.5}]  (57)
L_phys = L_bounds + L_roc + L_rise + L_drop          (58)
```

**Code.** `src/iints/research/losses.py`,
`PhysiologicalPINNLoss.physiology_penalty` computes only the absolute-bounds
penalty and the rate-of-change penalty. There is no IOB or COB term. The
omission is deliberate and documented in the source:

> IOB/COB direction rules are intentionally excluded: stress, exercise, illness
> and unannounced meals make them non-universal.

**Correct statement.** Equation (58) should read `L_phys = L_bounds + L_roc`.
Equations (56) and (57) describe a design that was considered and rejected, and
should be removed or explicitly labelled as not implemented.

**Why the code is right.** A penalty that asserts "glucose must fall when IOB is
high" is false precisely in the situations that matter most — during illness,
stress, exercise, or an unannounced meal. Training against it would teach the
model to be confidently wrong in exactly those cases.

---

## E2 — The Hovorka meal chain is two compartments, not three

**Paper.** §5.3, Table 5 lists state indices 8–10 as:

> `D1, D2, D3` — Solid stomach, liquid stomach, and gut carbohydrate
> compartments.

That is a three-stage solid/liquid/gut structure, which is the topology of the
Dalla Man meal model, not of Hovorka.

**Code.** `src/iints/core/patient/hovorka_model.py` implements the published
Hovorka **two**-compartment chain:

```
dD1/dt = -D1 / t_max_G
dD2/dt =  D1 / t_max_G - D2 / t_max_G
U_G    =  A_G · D2 / t_max_G
```

Index 10 (`D3`) is inert. The state docstring reads *"Reserved legacy meal mass
(migrated into D2 on restore)"*, the derivative is hard-coded `dD3_dt = 0.0`,
and `set_state` folds any legacy `D3` mass into `D2` when restoring a pre-v3
snapshot.

**Correct statement.** Indices 8–9 are the two-compartment Hovorka meal
absorption chain; index 10 is a retained legacy slot that is always zero and
carries no physiology.

**Note on the other backend.** The Bergman backend *does* use a three-stage
stomach → gut chain (`Q_sto1 → Q_sto2 → Q_gut`, parameters `tau_meal`, `k_abs`,
`f_bio`). That chain is an IINTS adaptation, is classed `adapted` in the formula
registry, and requires dataset calibration. It must not be attributed to
Hovorka, and numbers from the two backends are not interchangeable.

---

## Items raised in review that could not be reproduced

Recorded so they are not silently dropped.

**§5.3 graft / immune terms.** No graft, immune or stem-cell term appears in
§5.3 or in the Hovorka 19-state table, and none should: `M_graft` and
`immune_rejection_rate` belong to the 14-state Bergman backend
(`bergman_model.py`), where they default to zero. If a specific sentence is
wrong, it needs to be quoted.

**pLDDT coupled to insulin sensitivity.** The string `pLDDT` does not occur
anywhere in the paper text. Documentation and code consistently *refuse* the
coupling:

- `formula_registry.py` (F07): the sensitivity scalars *"are not inferred from
  AlphaFold confidence or ClinVar labels and require independent calibration
  before quantitative interpretation."*
- `docs/DIGITAL_TWIN_BIOLOGY.md`: *"Neither score measures biological activity,
  mutation severity, insulin sensitivity, glucose effect, PK/PD, or clinical
  safety. IINTS-AF never converts these values into a patient parameter or
  treatment action."*
- `docs/REGENERATIVE_ISLET_RESEARCH.md` lists pLDDT→insulin-sensitivity among
  the explicitly forbidden inferences.

If a claim to the contrary exists in another document, that document is the one
to correct.
