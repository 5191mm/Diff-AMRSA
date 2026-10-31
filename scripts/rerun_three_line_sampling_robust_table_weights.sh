#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MRSA_DATA_ROOT="${MRSA_DATA_ROOT:?Set MRSA_DATA_ROOT}"
GEOM_CKPT="${GEOM_CKPT:?Set GEOM_CKPT}"

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

for f in \
  "${GEOM_CKPT}" \
  "${SCRATCH_CKPT}" \
  "${FINETUNE_CKPT}" \
  "${PREDICTOR}" \
  "scripts/conf/drugs/megalodon_diffusion.yaml" \
  "scripts/conf/drugs/mrsa_diffusion_scratch_scheme_a.yaml" \
  "scripts/conf/drugs/mrsa_diffusion_finetune_scheme_a.yaml"
do
  test -f "$f" || { echo "Missing file: $f"; exit 1; }
  echo "OK $f"
done

run_one() {
  local group="$1"
  local config="$2"
  local ckpt="$3"
  local outdir="$4"
  local save_name="$5"
  local log_path="$6"

  echo "Sampling ${group}"

  PYTHONUNBUFFERED=1 python scripts/sample_domain_adapted_robust.py \
    --compare_group "${group}" \
    --config_path "${config}" \
    --ckpt_path "${ckpt}" \
    --predictor "${PREDICTOR}" \
    --n_graphs "${N_GRAPHS}" \
    --top_k "${TOP_K}" \
    --batch_size "${BATCH_SIZE}" \
    --timesteps "${TIMESTEPS}" \
    --save_path "${outdir}" \
    --save_name "${save_name}" \
    --activity_threshold 0.5 \
    --mic_ceiling 100.0 \
    --w_mic 0.40 \
    --w_activity_prob 0.35 \
    --w_qed 0.15 \
    --w_lipinski 0.05 \
    --w_alert 0.05 \
    --save_raw_pickle \
    2>&1 | tee "${log_path}"

  test -f "${outdir}/${save_name}_all_scored.csv" || { echo "Missing all_scored csv for ${group}"; exit 1; }
  test -f "${outdir}/${save_name}_ranked.csv" || { echo "Missing ranked csv for ${group}"; exit 1; }
  test -f "${outdir}/${save_name}_top${TOP_K}.sdf" || { echo "Missing sdf for ${group}"; exit 1; }
  test -f "${outdir}/${save_name}_summary.json" || { echo "Missing summary json for ${group}"; exit 1; }

  ls -lh \
    "${outdir}/${save_name}_all_scored.csv" \
    "${outdir}/${save_name}_ranked.csv" \
    "${outdir}/${save_name}_top${TOP_K}.sdf" \
    "${outdir}/${save_name}_summary.json"
}

run_one \
  pretrained_baseline \
  scripts/conf/drugs/megalodon_diffusion.yaml \
  "${GEOM_CKPT}" \
  outputs/compare_guided/pretrained_baseline \
  pretrained_baseline_guided_all_sdf \
  outputs/compare_guided/logs/pretrained_baseline_sampling.log

run_one \
  scratch_mrsa \
  scripts/conf/drugs/mrsa_diffusion_scratch_scheme_a.yaml \
  "${SCRATCH_CKPT}" \
  outputs/compare_guided/scratch_mrsa \
  scratch_mrsa_guided_all_sdf \
  outputs/compare_guided/logs/scratch_mrsa_sampling.log

run_one \
  finetune_mrsa \
  scripts/conf/drugs/mrsa_diffusion_finetune_scheme_a.yaml \
  "${FINETUNE_CKPT}" \
  outputs/compare_guided/finetune_mrsa \
  finetune_mrsa_guided_all_sdf \
  outputs/compare_guided/logs/finetune_mrsa_sampling.log

echo "All sampling finished"
find outputs/compare_guided -type f \( \
  -name "*all_scored.csv" -o \
  -name "*ranked.csv" -o \
  -name "*top*.sdf" -o \
  -name "*summary.json" -o \
  -name "*raw_results.pkl" \
\) | sort
