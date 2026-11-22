#!/usr/bin/env bash
set -euo pipefail
root="${1:?usage: bash scripts/package_results.sh results/RUN [artifact.zip]}"
out="${2:-${root%/}.zip}"
[[ -f "$root/pipeline.json" && -d "$root/stage_a" && -d "$root/stage_b" ]] || { echo 'Expected a joint run directory containing pipeline.json, stage_a and stage_b' >&2; exit 2; }
[[ ! -e "$out" ]] || { echo "Output already exists: $out" >&2; exit 2; }
case "$(realpath -m "$out")" in "$(realpath "$root")"/*) echo 'Archive must be outside its input tree' >&2; exit 2;; esac
absolute="$(realpath -m "$out")"
(cd "$(dirname "$root")" && zip -q -r "$absolute" "$(basename "$root")")
