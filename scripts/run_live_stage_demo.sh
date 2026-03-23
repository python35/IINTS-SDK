#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 examples/demos/07_live_stage_demo.py "$@"
