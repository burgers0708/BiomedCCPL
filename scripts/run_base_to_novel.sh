#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 DATA_ROOT DATASET [GPU_ID]" >&2
  exit 2
fi

DATA_ROOT=$1
DATASET=$2
GPU_ID=${3:-0}
CONFIG="configs/trainers/BiomedCCPL/base_to_novel/${DATASET}.yaml"

if [[ "${DATASET}" == "BUSI" ]]; then
  echo "BUSI is excluded from base-to-novel evaluation (only three classes)." >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Unknown dataset or missing config: ${CONFIG}" >&2
  exit 2
fi

for SEED in 1 2 3; do
  BASE_OUTPUT="output/base_to_novel/${DATASET}/base/seed${SEED}"
  NOVEL_OUTPUT="output/base_to_novel/${DATASET}/novel/seed${SEED}"

  CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --output-dir "${BASE_OUTPUT}" \
    --config-file "${CONFIG}" \
    DATASET.NAME "${DATASET}" \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES base

  CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --output-dir "${NOVEL_OUTPUT}" \
    --config-file "${CONFIG}" \
    --model-dir "${BASE_OUTPUT}" \
    --load-epoch 50 \
    --eval-only \
    DATASET.NAME "${DATASET}" \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES new
done

python tools/parse_test_res.py "output/base_to_novel/${DATASET}/base"
python tools/parse_test_res.py "output/base_to_novel/${DATASET}/novel"
