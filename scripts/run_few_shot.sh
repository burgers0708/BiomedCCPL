#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 DATA_ROOT DATASET SHOTS [GPU_ID]" >&2
  exit 2
fi

DATA_ROOT=$1
DATASET=$2
SHOTS=$3
GPU_ID=${4:-0}

case "${SHOTS}" in
  1|2|4|8|16) ;;
  *) echo "SHOTS must be one of: 1, 2, 4, 8, 16" >&2; exit 2 ;;
esac

CONFIG="configs/trainers/BiomedCCPL/few_shot/${SHOTS}/${DATASET}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Unknown dataset or missing config: ${CONFIG}" >&2
  exit 2
fi

for SEED in 1 2 3; do
  OUTPUT="output/few_shot/${DATASET}/${SHOTS}shot/seed${SEED}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT}" \
    --config-file "${CONFIG}" \
    DATASET.NAME "${DATASET}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES all
done

python tools/parse_test_res.py "output/few_shot/${DATASET}/${SHOTS}shot"
