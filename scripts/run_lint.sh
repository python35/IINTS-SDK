#!/usr/bin/env bash
set -euo pipefail

python3 -m flake8 .
python3 -m mypy src/iints
