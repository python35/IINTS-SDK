---
tags:
  - iints/user
  - iints/troubleshooting
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Troubleshooting From User Perspective

## Fast Diagnosis Table

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `No such command jetson` | old package installed | run `python -m pip install -U iints-sdk-python35` and check `iints --version` |
| extras warning in pip | old metadata or wrong version | verify PyPI version and run `python -m pip show iints-sdk-python35` |
| run stops early | patient/scenario/algorithm caused critical event | check completion ratio, safety summary, and `iints run preview` |
| glucose crashes before first meal | unsafe patient drift/defaults | use `patients/stable_patient.yaml` from quickstart |
| Tidepool/Nightscout import fails | auth/API/config issue | run command with `--help`, verify base URL and token source |
| Pico upload worries | real hardware risk | use `--dry-run`; keep bench-only test fixtures |

## Always Capture

```bash
iints doctor > doctor.txt
iints --help > commands.txt
python -m pip show iints-sdk-python35 > package.txt
```

## Evidence When Asking For Help

- command you ran
- SDK version
- `doctor.txt`
- output folder path
- patient/scenario/algorithm files
- whether the run completed or terminated early
