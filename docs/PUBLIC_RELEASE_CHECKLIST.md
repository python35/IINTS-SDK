# Public Release Checklist

Use this checklist the day before a public release, booth demo, or external handoff.

## 1. Fresh-Machine Install Smoke Test

Run this on a clean macOS or Linux machine:

```bash
python3 -m venv .venv-iints
source .venv-iints/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[mdmp]"
iints doctor --smoke-run
```

Expected result:

- the SDK installs without needing the repository checkout
- `iints doctor --smoke-run` exits cleanly

## 2. Demo Export Smoke Test

Verify that the installed SDK can export the public demo code:

```bash
iints demo-export --output-dir iints_demo
cd iints_demo
python 07_live_stage_demo.py
```

Expected result:

- the demo runs on a non-repository machine
- `results/booth_demo_live/booth_demo_poster.png` is created
- `results/booth_demo_live/JURY_TALK_TRACK.md` is created

## 3. Local AI Readiness

If you plan to show the local AI layer:

```bash
ollama pull ministral-3:3b
iints ai local-check --model ministral-3:3b
```

Expected result:

- Ollama is reachable
- the local model is installed
- the smoke test completes a real tiny generation

## 4. Realism Review

Generate a realism-oriented critique from the same run bundle:

```bash
iints ai review results/booth_demo_live/03_supervisor_override --model ministral-3:3b
```

Expected result:

- `results/booth_demo_live/03_supervisor_override/ai/realism_review.md` is created automatically
- the review contains concrete feedback points you can improve before release

## 5. Core Public Artifacts

Before release, confirm these public-facing artifacts are present and readable:

- demo poster PNG
- manual PDF
- release notes
- installation guide
- AI assistant guide
- evidence/source list

## 6. Security and Trust

Confirm:

- the packaged SDK still includes bundled MDMP support
- the Apache 2.0 `LICENSE` and `NOTICE` files are present
- CI is green on lint, tests, build, and dependency audit

## 7. What To Show Publicly

Recommended order:

1. show `07_live_stage_demo.py`
2. point at the patient/config knobs
3. run the demo
4. open the poster
5. optionally show `iints ai review ...`

That sequence keeps the story simple: code -> run -> result -> critique.
