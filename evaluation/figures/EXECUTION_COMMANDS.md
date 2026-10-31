
# One-click execution

## Run all figure scripts and export bilingual SVGs

```bash
cd /path/to/Diff-AMRSA
bash evaluation/figures/run_all_figures_and_collect_svgs.sh
```

This will sequentially run all figure-generation scripts in this package using the following fixed input files:

- Reference dataset: `../autodl-fs/data/processed_drug_mic_dataset_schemeA.csv`
- General-model ranked molecules: `outputs/compare_guided/pretrained_baseline/pretrained_baseline_guided_all_sdf_ranked.csv`
- MRSA-trained ranked molecules: `outputs/compare_guided/scratch_mrsa/scratch_mrsa_guided_all_sdf_ranked.csv`
- RCFR ranked molecules: `outputs/compare_guided/finetune_mrsa/finetune_mrsa_guided_all_sdf_ranked.csv`
- Their `all_scored.csv`, `summary.json`, and `top2000.sdf` companion files under the same folders

Logs are written to `outputs/logs/`.

## Final bilingual SVG folders

- English-only SVG figures: `outputs/all_svg_bilingual/en_svg`
- Chinese-only SVG figures: `outputs/all_svg_bilingual/zh_svg`

## Unified display names

### English
- Known MRSA Actives
- General-Model Generated
- MRSA-Trained Generated
- RCFR-Diffusion for MRSA

### Chinese
- 已知抗 MRSA 活性分子
- 通用模型生成分子
- MRSA 数据从零训练生成分子
- 本文模型
