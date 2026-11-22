#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
explicit=""
args=("$@")
for ((i=0;i<${#args[@]};i++)); do
  case "${args[i]}" in
    --venv-path) explicit="${args[i+1]:?missing --venv-path}";;
    --venv-path=*) explicit="${args[i]#*=}";;
  esac
done
venv="${explicit:-${SEAM_VENV:-}}"
if [[ -z "$venv" ]]; then
  declare -A found=()
  for start in "$ROOT" "$PWD"; do
    here="$(cd "$start" && pwd -P)"
    while [[ "$here" != / ]]; do
      if [[ -f "$here/.venv/pyvenv.cfg" ]]; then found["$(cd "$here/.venv" && pwd -P)"]=1; fi
      here="$(dirname "$here")"
    done
  done
  if ((${#found[@]} != 1)); then echo 'Cannot uniquely identify SEAM .venv. Supply --venv-path /absolute/path/.venv' >&2; exit 2; fi
  for venv in "${!found[@]}"; do :; done
fi
[[ -f "$venv/pyvenv.cfg" && -x "$venv/bin/python" ]] || { echo "Invalid Linux venv: $venv" >&2; exit 2; }
venv="$(cd "$venv" && pwd -P)"
export SEAM_SELECTED_VENV="$venv" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_INDEX=1
unset PYTHONHOME PYTHONPATH
cd "$ROOT"
exec "$venv/bin/python" -m stage_b.cli "$@"
