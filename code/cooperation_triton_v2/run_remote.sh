#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results/v2_$(date +%Y%m%d_%H%M%S)}"
python -m unittest cooperation_triton_v2.tests -v
CUDA_VISIBLE_DEVICES=0 python -m cooperation_triton_v2.gpu_checks
CUDA_VISIBLE_DEVICES=0 python -m cooperation_triton_v2.benchmark run \
  --suite smoke --blocks 1 --repeats 5 --discovery-blocks 1 --validation-repeats 3 \
  --out "$OUT/smoke"
CUDA_VISIBLE_DEVICES=0 python -m cooperation_triton_v2.benchmark calibrate \
  --suite calibration --out "$OUT/calibration"
CUDA_VISIBLE_DEVICES=0 TORCH_LOGS=output_code python -m cooperation_triton_v2.benchmark run \
  --suite targeted --model "$OUT/calibration/model.json" \
  --blocks 5 --repeats 20 --discovery-blocks 3 --validation-repeats 9 \
  --out "$OUT/targeted" 2> "$OUT/inductor_output_code.log"
printf '%s\n' "Results: $OUT/targeted/RESULTS.md" "Charts: $OUT/targeted/figures/"
