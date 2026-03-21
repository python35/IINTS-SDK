#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 examples/demos/06_booth_demo.py "$@"
