#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MRSA_DATA_ROOT="${MRSA_DATA_ROOT:?Set MRSA_DATA_ROOT to the processed conformer dataset}"
GEOM_CKPT="${GEOM_CKPT:?Set GEOM_CKPT to the trusted pretrained checkpoint}"
PREDICTOR="${PREDICTOR:-${MRSA_DATA_ROOT}/processed/mrsa_activity_predictor_schemeA.joblib}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
unset MRSA_RESUME_TRAINING_STATE

python scripts/check_strict_mrsa_compare.py

PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python scripts/train_domain_adaptation.py \
  --config-path conf/drugs \
  --config-name mrsa_diffusion_scratch_scheme_a \
  data.dataset_root="${MRSA_DATA_ROOT}" \
  strict_compare.dataset_root="${MRSA_DATA_ROOT}"

PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python scripts/train_domain_adaptation.py \
  --config-path conf/drugs \
  --config-name mrsa_diffusion_finetune_scheme_a \
  resume="${GEOM_CKPT}" \
  data.dataset_root="${MRSA_DATA_ROOT}" \
  strict_compare.dataset_root="${MRSA_DATA_ROOT}"

python scripts/check_ckpt_hparams_strict.py \
  results/mrsa_strict_scratch_scheme_a/mrsa_strict_scratch_scheme_a/checkpoints/last.ckpt
python scripts/check_ckpt_hparams_strict.py \
  results/mrsa_strict_finetune_scheme_a/mrsa_strict_finetune_scheme_a/checkpoints/last.ckpt

python scripts/sample_domain_adapted.py \
  --compare_group scratch_mrsa \
  --config_path scripts/conf/drugs/mrsa_diffusion_scratch_scheme_a.yaml \
  --ckpt_path results/mrsa_strict_scratch_scheme_a/mrsa_strict_scratch_scheme_a/checkpoints/last.ckpt \
  --predictor "${PREDICTOR}" \
  --n_graphs 2000 \
  --top_k 200 \
  --batch_size 100 \
  --timesteps 500 \
  --statistics_split train \
  --save_path outputs/compare_guided/scratch_mrsa \
  --save_name scratch_mrsa_guided \
  --activity_threshold 0.5 \
  --mic_ceiling 100.0 \
  --w_mic 0.40 \
  --w_activity_prob 0.35 \
  --w_qed 0.15 \
  --w_lipinski 0.05 \
  --w_alert 0.05 \
  --save_raw_pickle

python scripts/sample_domain_adapted.py \
  --compare_group finetune_mrsa \
  --config_path scripts/conf/drugs/mrsa_diffusion_finetune_scheme_a.yaml \
  --ckpt_path results/mrsa_strict_finetune_scheme_a/mrsa_strict_finetune_scheme_a/checkpoints/last.ckpt \
  --predictor "${PREDICTOR}" \
  --n_graphs 2000 \
  --top_k 200 \
  --batch_size 100 \
  --timesteps 500 \
  --statistics_split train \
  --save_path outputs/compare_guided/finetune_mrsa \
  --save_name finetune_mrsa_guided \
  --activity_threshold 0.5 \
  --mic_ceiling 100.0 \
  --w_mic 0.40 \
  --w_activity_prob 0.35 \
  --w_qed 0.15 \
  --w_lipinski 0.05 \
  --w_alert 0.05 \
  --save_raw_pickle
