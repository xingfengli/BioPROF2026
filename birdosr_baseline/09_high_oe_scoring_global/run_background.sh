#!/usr/bin/env bash
set -euo pipefail

exec bash "$(dirname "$0")/../_run_final_group_background.sh" high_oe_scoring_global 09_high_oe_scoring_global "${1:-all}"
