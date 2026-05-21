---
tags:
  - iints/presentation
  - iints/demo
  - iints/script
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# 5-Minute Live Demo Script

## Before The Call

- [ ] Terminal open in project folder.
- [ ] Virtual environment activated.
- [ ] `iints doctor` already tested.
- [ ] Result folder cleared or timestamped.
- [ ] Obsidian open at [[SDK User Home]].

## Script

### 0:00 - Context

"IINTS is a simulation and evidence SDK for insulin-algorithm research. I will show a small demo, then the evidence trail behind it."

### 0:45 - Run Command

```bash
iints demo-live --output-dir results/live_demo
```

### 1:30 - Show Output

- output folder
- summary/report
- command metadata

### 2:30 - Explain Evidence

Open [[Sources by SDK Feature]] and say which sources support metrics, datasets, and safety wording.

### 3:30 - Safety Boundary

Open [[Safety Boundary Script]]. Make it explicit: simulation only, bench-only hardware, no treatment advice.

### 4:15 - Invite Feedback

Ask one targeted question from [[Questions To Ask Experts]].

## Backup If Command Fails

Use screenshots/previous outputs and explain that reproducibility includes showing failures honestly.
