#!/usr/bin/env bash
set -euo pipefail

exec bash "$(dirname "$0")/../_run_final_group_background.sh" high_oe_threshold 10_high_oe_threshold "${1:-all}"
