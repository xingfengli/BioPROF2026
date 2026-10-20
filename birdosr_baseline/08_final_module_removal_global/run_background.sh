#!/usr/bin/env bash
set -euo pipefail

exec bash "$(dirname "$0")/../_run_final_group_background.sh" final_module_global 08_final_module_removal_global "${1:-all}"
