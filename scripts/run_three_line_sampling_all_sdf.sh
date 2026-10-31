#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MRSA_DATA_ROOT="${MRSA_DATA_ROOT:?Set MRSA_DATA_ROOT to your processed dataset}"
GEOM_DATA_ROOT="${GEOM_DATA_ROOT:?Set GEOM_DATA_ROOT to your pretrained geometry dataset}"
GEOM_CKPT="${GEOM_CKPT:?Set GEOM_CKPT to the pretrained backbone checkpoint}"

SCRATCH_CKPT=${PROJECT_ROOT}/results/mrsa_strict_scratch_scheme_a/mrsa_strict_scratch_scheme_a/checkpoints/last.ckpt
FINETUNE_CKPT=${PROJECT_ROOT}/results/mrsa_strict_finetune_scheme_a/mrsa_strict_finetune_scheme_a/checkpoints/last.ckpt
PREDICTOR=${MRSA_DATA_ROOT}/processed/mrsa_activity_predictor_schemeA.joblib

N_GRAPHS=2000
TOP_K=2000
BATCH_SIZE=100
TIMESTEPS=500

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

mkdir -p outputs/compare_guided/pretrained_baseline
mkdir -p outputs/compare_guided/scratch_mrsa
mkdir -p outputs/compare_guided/finetune_mrsa
mkdir -p outputs/compare_guided/logs

echo "Checking required files"

test -f "${GEOM_CKPT}" || { echo "Missing GEOM_CKPT ${GEOM_CKPT}"; exit 1; }
test -f "${SCRATCH_CKPT}" || { echo "Missing SCRATCH_CKPT ${SCRATCH_CKPT}"; exit 1; }
test -f "${FINETUNE_CKPT}" || { echo "Missing FINETUNE_CKPT ${FINETUNE_CKPT}"; exit 1; }
test -f "${PREDICTOR}" || { echo "Missing PREDICTOR ${PREDICTOR}"; exit 1; }

test -f "${GEOM_DATA_ROOT}/processed/train_n_h.pickle" || { echo "Missing GEOM statistics"; exit 1; }
test -f "${MRSA_DATA_ROOT}/processed/train_n_h.pickle" || { echo "Missing MRSA statistics"; exit 1; }

echo "Sampling Original-GEOM"

PYTHONUNBUFFERED=1 python scripts/sample_domain_adapted.py \
  --compare_group pretrained_baseline \
  --config_path scripts/conf/drugs/megalodon_diffusion.yaml \
  --ckpt_path "${GEOM_CKPT}" \
  --predictor "${PREDICTOR}" \
  --n_graphs "${N_GRAPHS}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --timesteps "${TIMESTEPS}" \
  --statistics_split train \
  --save_path outputs/compare_guided/pretrained_baseline \
  --save_name pretrained_baseline_guided_all_sdf \
  --activity_threshold 0.5 \
  --mic_ceiling 100.0 \
  --w_mic 0.40 \
  --w_activity_prob 0.35 \
  --w_qed 0.15 \
  --w_lipinski 0.05 \
  --w_alert 0.05 \
  --save_raw_pickle \
  2>&1 | tee outputs/compare_guided/logs/pretrained_baseline_sampling.log

echo "Sampling Scratch-MRSA"

PYTHONUNBUFFERED=1 python scripts/sample_domain_adapted.py \
  --compare_group scratch_mrsa \
  --config_path scripts/conf/drugs/mrsa_diffusion_scratch_scheme_a.yaml \
  --ckpt_path "${SCRATCH_CKPT}" \
  --predictor "${PREDICTOR}" \
  --n_graphs "${N_GRAPHS}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --timesteps "${TIMESTEPS}" \
  --statistics_split train \
  --save_path outputs/compare_guided/scratch_mrsa \
  --save_name scratch_mrsa_guided_all_sdf \
  --activity_threshold 0.5 \
  --mic_ceiling 100.0 \
  --w_mic 0.40 \
  --w_activity_prob 0.35 \
  --w_qed 0.15 \
  --w_lipinski 0.05 \
  --w_alert 0.05 \
  --save_raw_pickle \
  2>&1 | tee outputs/compare_guided/logs/scratch_mrsa_sampling.log

echo "Sampling Finetune-MRSA"

PYTHONUNBUFFERED=1 python scripts/sample_domain_adapted.py \
  --compare_group finetune_mrsa \
  --config_path scripts/conf/drugs/mrsa_diffusion_finetune_scheme_a.yaml \
  --ckpt_path "${FINETUNE_CKPT}" \
  --predictor "${PREDICTOR}" \
  --n_graphs "${N_GRAPHS}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --timesteps "${TIMESTEPS}" \
  --statistics_split train \
  --save_path outputs/compare_guided/finetune_mrsa \
  --save_name finetune_mrsa_guided_all_sdf \
  --activity_threshold 0.5 \
  --mic_ceiling 100.0 \
  --w_mic 0.40 \
  --w_activity_prob 0.35 \
  --w_qed 0.15 \
  --w_lipinski 0.05 \
  --w_alert 0.05 \
  --save_raw_pickle \
  2>&1 | tee outputs/compare_guided/logs/finetune_mrsa_sampling.log

echo "All sampling finished"

echo "Original-GEOM output"
ls -lh outputs/compare_guided/pretrained_baseline

echo "Scratch-MRSA output"
ls -lh outputs/compare_guided/scratch_mrsa

echo "Finetune-MRSA output"
ls -lh outputs/compare_guided/finetune_mrsa
