# Contributing to IINTS-AF SDK

Thanks for your interest. IINTS-AF is research and educational software for
in-silico diabetes-technology work. It is **not a medical device** and no
contribution may present it as one.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting set up

```bash
git clone https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                 # 176 test modules
mypy src/iints/           # must be clean
flake8                    # must be clean
```

Python 3.10+ is supported.

## Ways to contribute

- **Report a bug.** Open an issue with the smallest command or script that
  reproduces it, the seed, and the SDK version (`python -c "import iints; print(iints.__version__)"`).
- **Report a physiological implausibility.** These are the most valuable
  reports. Include the trace, the configuration, and why the behaviour is
  wrong. A run that produces an impossible trajectory is a bug even if every
  test passes.
- **Add or correct a model term.** See the rules below.
- **Improve documentation**, especially where a claim is stronger than the
  evidence behind it.

## Rules specific to this project

These exist because the software makes scientific claims, and the claims have
to survive scrutiny.

**1. Every equation is registered.** Physiological, sensor, safety and loss
formulas live in `src/iints/core/formula_registry.py` with a canonical
expression, LaTeX, units, implementation paths, literature DOIs and an
`evidence_class` of `canonical`, `adapted` or `heuristic`. A pull request that
adds or changes a model term must update its registry entry in the same commit.

**2. Label the evidence honestly.** If a coefficient is your assumption rather
than an identified value, mark the entry `heuristic` and say so in
`validation_note`. Being explicit about an assumption is never a defect; hiding
one is.

**3. New physiology defaults to inert.** A new term must default to zero, off,
or a multiplier of 1, so that existing runs are bit-identical after the change.
State this in the pull request.

**4. Do not weaken a guard to make a test pass.** The independent supervisor,
the input validator and the domain guards in the patient models exist to make
failures visible. If a guard fires wrongly, fix the cause or change the guard
deliberately with a justification — do not raise a threshold to get green.

**5. Fail loudly, not quietly.** A negative compartment, a non-finite state or
an out-of-bounds fraction must raise. Clipping a physiological error into a
plausible-looking number is worse than crashing.

**6. Keep randomness out of the dose path.** Stochastic components take an
explicit seed, record it, and never mutate global random state. A sampled value
must not become an insulin result.

**7. No clinical claims.** Do not describe output as diagnostic, do not imply
validation that has not been done, and do not add language suggesting the
simulator can determine a real dose.

## Pull requests

- Branch from `main`; keep the change focused.
- Add tests. Behavioural changes need a test that fails before and passes after.
- Run `pytest`, `mypy src/iints/` and `flake8` before pushing.
- Describe what changed, why, and what would have to be true for it to be
  wrong. If the change alters any published number, say which and by how much.
- Reference the issue it closes.

Maintainer review focuses on: does the code do what the registry says, is the
evidence class right, and can a result still be traced to its inputs.

## Reporting a security or safety issue

Do not open a public issue. Email <rune.bobbaers@gmail.com> with details and a
reproduction. This includes anything that could cause the SDK to emit an unsafe
value without an accompanying warning.

## Licence

Contributions are accepted under the [Apache-2.0](LICENSE) licence of this
project.
