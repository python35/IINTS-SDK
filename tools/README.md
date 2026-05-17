# Tools Layout

This repository uses the same tool layout rules as the standalone MDMP repository.

- `tools/ci/`: CI and policy checks used by GitHub Actions
- `tools/dev/`: local maintainer workflows, release checks, and multi-repo helpers
- `tools/docs/`: manual and documentation builders
- `tools/data/`: dataset import and conversion utilities
- `tools/analysis/`: plotting, diagnostics, and report helpers
- `tools/assets/`: branding and asset generation helpers

User-facing entrypoints stay in `scripts/`.

Common maintainer commands:

```bash
tools/dev/sdk_check.sh quick
tools/dev/sdk_check.sh edge
tools/dev/sdk_check.sh docs
tools/dev/sdk_check.sh full
tools/dev/release_audit.sh 1.5.5
```
