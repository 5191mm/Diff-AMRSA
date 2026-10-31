
#!/usr/bin/env bash
set -euo pipefail

# One-click script: run all figure-generation scripts, then collect bilingual SVG figures.
# Usage:
#   cd /path/to/Diff-AMRSA
#   bash evaluation/figures/run_all_figures_and_collect_svgs.sh

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
cd "$PROJECT_ROOT"

# ========== Fixed input paths ==========
REFERENCE_CSV=${REFERENCE_CSV:-mrsa_scheme_a/processed_drug_mic_dataset_schemeA.csv}
ORIGINAL_RANKED=${ORIGINAL_RANKED:-outputs/compare_guided/pretrained_baseline/pretrained_baseline_guided_all_sdf_ranked.csv}
SCRATCH_RANKED=${SCRATCH_RANKED:-outputs/compare_guided/scratch_mrsa/scratch_mrsa_guided_all_sdf_ranked.csv}
RCFR_RANKED=${RCFR_RANKED:-outputs/compare_guided/finetune_mrsa/finetune_mrsa_guided_all_sdf_ranked.csv}

ORIGINAL_ALL_SCORED=${ORIGINAL_ALL_SCORED:-outputs/compare_guided/pretrained_baseline/pretrained_baseline_guided_all_sdf_all_scored.csv}
SCRATCH_ALL_SCORED=${SCRATCH_ALL_SCORED:-outputs/compare_guided/scratch_mrsa/scratch_mrsa_guided_all_sdf_all_scored.csv}
RCFR_ALL_SCORED=${RCFR_ALL_SCORED:-outputs/compare_guided/finetune_mrsa/finetune_mrsa_guided_all_sdf_all_scored.csv}

ORIGINAL_SUMMARY_JSON=${ORIGINAL_SUMMARY_JSON:-outputs/compare_guided/pretrained_baseline/pretrained_baseline_guided_all_sdf_summary.json}
SCRATCH_SUMMARY_JSON=${SCRATCH_SUMMARY_JSON:-outputs/compare_guided/scratch_mrsa/scratch_mrsa_guided_all_sdf_summary.json}
RCFR_SUMMARY_JSON=${RCFR_SUMMARY_JSON:-outputs/compare_guided/finetune_mrsa/finetune_mrsa_guided_all_sdf_summary.json}

ORIGINAL_SDF=${ORIGINAL_SDF:-outputs/compare_guided/pretrained_baseline/pretrained_baseline_guided_all_sdf_top2000.sdf}
SCRATCH_SDF=${SCRATCH_SDF:-outputs/compare_guided/scratch_mrsa/scratch_mrsa_guided_all_sdf_top2000.sdf}
RCFR_SDF=${RCFR_SDF:-outputs/compare_guided/finetune_mrsa/finetune_mrsa_guided_all_sdf_top2000.sdf}

PY=${PY:-python}

need_file () {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] Missing required file: $1" >&2
    exit 1
  fi
}

for f in   "$REFERENCE_CSV" "$ORIGINAL_RANKED" "$SCRATCH_RANKED" "$RCFR_RANKED"   "$ORIGINAL_ALL_SCORED" "$SCRATCH_ALL_SCORED" "$RCFR_ALL_SCORED"   "$ORIGINAL_SUMMARY_JSON" "$SCRATCH_SUMMARY_JSON" "$RCFR_SUMMARY_JSON"   "$ORIGINAL_SDF" "$SCRATCH_SDF" "$RCFR_SDF"; do
  need_file "$f"
done

mkdir -p outputs/logs
run_and_log () {
  local name="$1"; shift
  echo "[RUN] $name"
  "$@" 2>&1 | tee "outputs/logs/${name}.log"
}

# =========================
# 1) Main integrated paper-style figure sets
# =========================
run_and_log run_paper_style_mrsa_figures   "$PY" evaluation/figures/run_paper_style_mrsa_figures.py   --reference_csv "$REFERENCE_CSV"   --original_ranked_csv "$ORIGINAL_RANKED"   --scratch_ranked_csv "$SCRATCH_RANKED"   --rcfr_ranked_csv "$RCFR_RANKED"   --original_all_scored_csv "$ORIGINAL_ALL_SCORED"   --scratch_all_scored_csv "$SCRATCH_ALL_SCORED"   --rcfr_all_scored_csv "$RCFR_ALL_SCORED"   --original_summary_json "$ORIGINAL_SUMMARY_JSON"   --scratch_summary_json "$SCRATCH_SUMMARY_JSON"   --rcfr_summary_json "$RCFR_SUMMARY_JSON"   --original_sdf "$ORIGINAL_SDF"   --scratch_sdf "$SCRATCH_SDF"   --rcfr_sdf "$RCFR_SDF"   --outdir outputs/paper_style_mrsa_figures   --top_n 200 --sampled_count 2000 --reference_active_only

run_and_log run_full_mrsa_generation_experiments   "$PY" evaluation/figures/run_full_mrsa_generation_experiments.py   --reference_csv "$REFERENCE_CSV"   --original_ranked_csv "$ORIGINAL_RANKED"   --scratch_ranked_csv "$SCRATCH_RANKED"   --rcfr_ranked_csv "$RCFR_RANKED"   --original_all_scored_csv "$ORIGINAL_ALL_SCORED"   --scratch_all_scored_csv "$SCRATCH_ALL_SCORED"   --rcfr_all_scored_csv "$RCFR_ALL_SCORED"   --original_summary_json "$ORIGINAL_SUMMARY_JSON"   --scratch_summary_json "$SCRATCH_SUMMARY_JSON"   --rcfr_summary_json "$RCFR_SUMMARY_JSON"   --original_sdf "$ORIGINAL_SDF"   --scratch_sdf "$SCRATCH_SDF"   --rcfr_sdf "$RCFR_SDF"   --outdir outputs/full_mrsa_generation_experiments   --top_n 200 --sampled_count 2000 --reference_active_only

run_and_log run_rcfr_generation_and_closeness_experiments   "$PY" evaluation/figures/run_rcfr_generation_and_closeness_experiments.py   --reference_csv "$REFERENCE_CSV"   --original_ranked_csv "$ORIGINAL_RANKED"   --scratch_ranked_csv "$SCRATCH_RANKED"   --rcfr_ranked_csv "$RCFR_RANKED"   --original_all_scored_csv "$ORIGINAL_ALL_SCORED"   --scratch_all_scored_csv "$SCRATCH_ALL_SCORED"   --rcfr_all_scored_csv "$RCFR_ALL_SCORED"   --original_summary_json "$ORIGINAL_SUMMARY_JSON"   --scratch_summary_json "$SCRATCH_SUMMARY_JSON"   --rcfr_summary_json "$RCFR_SUMMARY_JSON"   --outdir outputs/rcfr_generation_and_closeness_experiments   --top_n 200 --sampled_count 2000 --reference_active_only

# =========================
# 2) Explicit four-line / three-line / scatter / top molecules figures
# =========================
run_and_log compare_four_group_mrsa_database_closeness   "$PY" evaluation/figures/compare_four_group_mrsa_database_closeness.py   --reference_csv "$REFERENCE_CSV"   --original_csv "$ORIGINAL_RANKED"   --scratch_csv "$SCRATCH_RANKED"   --rcfr_csv "$RCFR_RANKED"   --outdir outputs/compare_four_group_mrsa_database_closeness   --top_n 200 --reference_active_only

run_and_log analyze_three_line_database_closeness   "$PY" evaluation/figures/analyze_three_line_database_closeness.py   --reference_csv "$REFERENCE_CSV"   --original_csv "$ORIGINAL_RANKED"   --scratch_csv "$SCRATCH_RANKED"   --finetune_csv "$RCFR_RANKED"   --outdir outputs/analyze_three_line_database_closeness   --top_n 200 --reference_active_only

run_and_log plot_mrsa_distribution_comparison_four_groups   "$PY" evaluation/figures/plot_mrsa_distribution_comparison_four_groups.py   --original_csv "$REFERENCE_CSV"   --pretrained_baseline_csv "$ORIGINAL_RANKED"   --scratch_csv "$SCRATCH_RANKED"   --finetune_csv "$RCFR_RANKED"   --outdir outputs/plot_mrsa_distribution_comparison_four_groups   --original_active_only --top_n_generated 200

run_and_log plot_three_line_distribution_compare   "$PY" evaluation/figures/plot_three_line_distribution_compare.py   --reference_csv "$REFERENCE_CSV"   --rcfr_csv "$RCFR_RANKED"   --scratch_csv "$SCRATCH_RANKED"   --rcfr_top_n 100 --scratch_top_n 100   --outdir outputs/plot_three_line_distribution_compare

run_and_log plot_three_line_distribution_compare_top100   "$PY" evaluation/figures/plot_three_line_distribution_compare_top100.py   --original_dataset_csv "$REFERENCE_CSV"   --scratch_ranked_csv "$SCRATCH_RANKED"   --rcfr_ranked_csv "$RCFR_RANKED"   --outdir outputs/plot_three_line_distribution_compare_top100   --top_n 100 --original_active_only

run_and_log plot_three_line_distribution_top100   "$PY" evaluation/figures/plot_three_line_distribution_top100.py   --reference_csv "$REFERENCE_CSV"   --scratch_csv "$SCRATCH_RANKED"   --rcfr_csv "$RCFR_RANKED"   --outdir outputs/plot_three_line_distribution_top100   --top_n 100 --reference_active_only

run_and_log plot_mrsa_distribution_comparison_v3   "$PY" evaluation/figures/plot_mrsa_distribution_comparison_v3.py   --original_csv "$REFERENCE_CSV"   --generated_csv "$RCFR_RANKED"   --outdir outputs/plot_mrsa_distribution_comparison_v3   --original_active_only

run_and_log compare_sampling_properties_scatter_three_lines   "$PY" evaluation/figures/compare_sampling_properties_scatter_three_lines.py   --pretrained_baseline_sdf "$ORIGINAL_SDF"   --scratch_sdf "$SCRATCH_SDF"   --finetune_sdf "$RCFR_SDF"   --outdir outputs/compare_sampling_properties_scatter_three_lines   --top_n 200

run_and_log compare_sdf_properties_scatter   "$PY" evaluation/figures/compare_sdf_properties_scatter.py   --original_sdf "$ORIGINAL_SDF"   --mrsa_sdf "$RCFR_SDF"   --outdir outputs/compare_sdf_properties_scatter   --top_n 200

run_and_log plot_top10_molecules   "$PY" evaluation/figures/plot_top10_molecules.py   --ranked_csv "$RCFR_RANKED"   --sdf "$RCFR_SDF"   --outdir outputs/plot_top10_molecules   --top_n 10

# =========================
# 3) Collect all SVGs and build clean English/Chinese folders
# =========================
run_and_log export_svg_bilingual_labels   "$PY" evaluation/figures/export_svg_bilingual_labels.py   --input_root outputs   --output_root outputs/all_svg_bilingual

echo "All done."
echo "English SVG folder: outputs/all_svg_bilingual/en_svg"
echo "Chinese SVG folder: outputs/all_svg_bilingual/zh_svg"
