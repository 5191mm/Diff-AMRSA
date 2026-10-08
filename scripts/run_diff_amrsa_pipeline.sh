#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${1:-}" == "--config" ]]; then
  shift
  source "${1}"
  shift
else
  source "${ROOT_DIR}/configs/diff_amrsa_paths.env"
fi

STEP="${1:-help}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

case "${STEP}" in
  raw)
    python data_processing/prepare_raw_tables.py \
      --input_dir "${DIFF_AMRSA_RAW_DIR}" \
      --output_csv "${DIFF_AMRSA_PROCESSED_CSV}" \
      --removed_log_csv "${DIFF_AMRSA_PROCESSED_DIR}/removed_rows_log.csv" \
      --summary_csv "${DIFF_AMRSA_PROCESSED_DIR}/processing_summary.csv"
    ;;
  scheme_a|prepare)
    python data_processing/prepare_dataset.py \
      --input "${DIFF_AMRSA_PROCESSED_CSV}" \
      --outdir "${DIFF_AMRSA_SCHEME_A_DIR}"
    ;;
  conformers)
    python data_processing/build_conformers.py \
      --input "${DIFF_AMRSA_SCHEME_A_CSV}" \
      --output_root "${DIFF_AMRSA_DATASET_ROOT}" \
      --neutralize_core \
      --num_candidates "${DIFF_AMRSA_NUM_CANDIDATES}" \
      --select_conformers "${DIFF_AMRSA_SELECT_CONFORMERS}" \
      --num_workers "${DIFF_AMRSA_NUM_WORKERS}" \
      --rdkit_threads "${DIFF_AMRSA_RDKIT_THREADS}" \
      --molecule_timeout_sec "${DIFF_AMRSA_MOLECULE_TIMEOUT_SEC}"
    ;;
  predictor)
    python scripts/train_activity_predictor.py \
      --input "${DIFF_AMRSA_SCHEME_A_CSV}" \
      --outdir "${DIFF_AMRSA_DATASET_ROOT}/processed" \
      --n_estimators 200 \
      --n_jobs 4
    ;;
  finetune)
    python scripts/finetune.py \
      --config-path conf/drugs \
      --config-name diff_amrsa_finetune \
      resume="${DIFF_AMRSA_PRETRAINED_CKPT}" \
      data.dataset_root="${DIFF_AMRSA_DATASET_ROOT}" \
      train.n_epochs="${DIFF_AMRSA_EPOCHS}" \
      data.batch_size="${DIFF_AMRSA_BATCH_SIZE}"
    ;;
  sample)
    mkdir -p "${DIFF_AMRSA_GENERATED_DIR}"
    python scripts/sample_diff_amrsa.py \
      --config_path scripts/conf/drugs/diff_amrsa_finetune.yaml \
      --ckpt_path "${DIFF_AMRSA_FINETUNED_CKPT}" \
      --predictor "${DIFF_AMRSA_PREDICTOR}" \
      --save_path "${DIFF_AMRSA_GENERATED_DIR}/diff_amrsa" \
      --save_name diff_amrsa_candidates \
      --n_graphs "${DIFF_AMRSA_N_GRAPHS}" \
      --batch_size "${DIFF_AMRSA_SAMPLE_BATCH_SIZE}" \
      --timesteps "${DIFF_AMRSA_TIMESTEPS}" \
      --top_k "${DIFF_AMRSA_TOP_K}"
    ;;
  baseline)
    mkdir -p "${DIFF_AMRSA_GENERATED_DIR}"
    python scripts/sample_baseline.py \
      --config_path scripts/conf/drugs/baseline_diffusion.yaml \
      --ckpt_path "${DIFF_AMRSA_PRETRAINED_CKPT}" \
      --timesteps "${DIFF_AMRSA_TIMESTEPS}" \
      --n_graphs "${DIFF_AMRSA_TOP_K}" \
      --save_path "${DIFF_AMRSA_GENERATED_DIR}/baseline" \
      --save_name baseline_candidates \
      --save_sdf
    ;;
  analyze)
    python table/generate_table_4_2_from_csv.py \
      --generated_csv "${DIFF_AMRSA_GENERATED_DIR}/diff_amrsa/diff_amrsa_candidates_ranked.csv" \
      --training_csv "${DIFF_AMRSA_PROCESSED_CSV}" \
      --out_csv "${DIFF_AMRSA_GENERATED_DIR}/table_4_2_molecular_quality.csv" || true
    python table/generate_table_4_4_domain_consistency.py \
      --generated_csv "${DIFF_AMRSA_GENERATED_DIR}/diff_amrsa/diff_amrsa_candidates_ranked.csv" \
      --reference_csv "${DIFF_AMRSA_PROCESSED_CSV}" \
      --out_csv "${DIFF_AMRSA_GENERATED_DIR}/table_4_4_domain_consistency.csv" || true
    ;;
  all)
    for step_name in raw scheme_a conformers predictor finetune sample analyze; do
      "${BASH_SOURCE[0]}" "${step_name}"
    done
    ;;
  help|*)
    cat <<EOF
Usage: bash scripts/run_diff_amrsa_pipeline.sh [raw|scheme_a|conformers|predictor|finetune|sample|baseline|analyze|all]
Optional: bash scripts/run_diff_amrsa_pipeline.sh --config configs/diff_amrsa_paths.env <step>
EOF
    ;;
esac
