# Diff-AMRSA: Anti-MRSA Molecule Generation with 3D Diffusion Fine-Tuning

Diff-AMRSA is a reproducible project for anti-MRSA molecule generation. It adapts
the Megalodon large diffusion model to an anti-MRSA activity dataset, builds
Scheme A training annotations, generates 3D conformer data, fine-tunes the
diffusion model, trains an activity proxy model, and samples ranked candidate
molecules.

This repository is derived from the Megalodon codebase and keeps compatibility
with Megalodon pretrained checkpoints. The canonical package in this repository
is `diff_amrsa`; a minimal `megalodon` shim is kept only for legacy checkpoint
loading and backward-compatible imports.

Upstream Megalodon repository: https://github.com/NVIDIA-BioNeMo/megalodon

Upstream paper:

```bibtex
@article{reidenbach2025applications,
  title={Applications of Modular Co-Design for De Novo 3D Molecule Generation},
  author={Reidenbach, Danny and Nikitin, Filipp and Isayev, Olexandr and Paliwal, Saee},
  journal={arXiv preprint arXiv:2505.18392},
  year={2025}
}
```

## Repository structure

```text
Diff-AMRSA/
├── README.md
├── MODEL_CARD.md
├── CITATION.cff
├── CLEANED_MANIFEST.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── diff_amrsa_paths.env
├── data/
│   └── README.md
├── data_processing/
│   ├── prepare_raw_tables.py
│   ├── prepare_dataset.py
│   ├── build_conformers.py
│   └── build_conformers_core.py
├── scripts/
│   ├── run_diff_amrsa_pipeline.sh
│   ├── train.py
│   ├── finetune.py
│   ├── train_activity_predictor.py
│   ├── sample_diff_amrsa.py
│   ├── sample_baseline.py
│   ├── learn_bin_weights.py
│   └── conf/drugs/
│       ├── baseline_diffusion.yaml
│       ├── diff_amrsa_finetune.yaml
│       └── diff_amrsa_scratch.yaml
├── src/
│   ├── diff_amrsa/
│   └── megalodon/
├── draw/
└── table/
```

## Setup

```bash
conda create -n diff-amrsa python=3.10 -y
conda activate diff-amrsa
pip install -e .
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

## External data and checkpoints

Large data files, checkpoints, processed `.pt` files, pickles, generated SDF files,
and trained predictors are not stored in this repository.

1. Download the three raw Figshare data files from:

   https://figshare.com/s/5fb5879ee14c8d2a1a5d

2. Place them in the folder configured by `DIFF_AMRSA_RAW_DIR` in:

   ```bash
   configs/diff_amrsa_paths.env
   ```

3. Download the upstream Megalodon large diffusion checkpoint and set:

   ```bash
   export DIFF_AMRSA_PRETRAINED_CKPT=/path/to/megalodon_large_diffusion.ckpt
   ```

## End-to-end pipeline

Edit `configs/diff_amrsa_paths.env` once, then run each step.

### 1. Prepare raw MRSA activity table

```bash
bash scripts/run_diff_amrsa_pipeline.sh raw
```

This runs:

```bash
python data_processing/prepare_raw_tables.py
```

Input: raw Figshare CSV/XLSX files.  
Output: `processed_drug_mic_dataset.csv`.

### 2. Build Scheme A annotations

```bash
bash scripts/run_diff_amrsa_pipeline.sh scheme_a
```

This runs:

```bash
python data_processing/prepare_dataset.py
```

Input: `processed_drug_mic_dataset.csv`.  
Output: `processed_drug_mic_dataset_schemeA.csv`, summary JSON, and count table.

Scheme A is the project-specific annotation policy that unifies activity
classification, MIC regression, and molecule-generation training masks.

### 3. Generate 3D conformers and processed training files

```bash
bash scripts/run_diff_amrsa_pipeline.sh conformers
```

This runs:

```bash
python data_processing/build_conformers.py
```

Input: Scheme A CSV.  
Output: raw split pickles and processed PyTorch Geometric files under the
configured dataset root.

### 4. Train activity predictor

```bash
bash scripts/run_diff_amrsa_pipeline.sh predictor
```

This trains the Scheme A activity proxy model used for candidate ranking.

### 5. Fine-tune Diff-AMRSA from Megalodon large diffusion

```bash
bash scripts/run_diff_amrsa_pipeline.sh finetune
```

The fine-tuning command uses `scripts/conf/drugs/diff_amrsa_finetune.yaml` and
loads weights from `DIFF_AMRSA_PRETRAINED_CKPT`.

### 6. Sample and rank anti-MRSA candidates

```bash
bash scripts/run_diff_amrsa_pipeline.sh sample
```

This generates molecules from the fine-tuned checkpoint and ranks them with the
activity proxy score, QED, Lipinski, alert, and MIC-related components.

### 7. Baseline Megalodon sampling

```bash
bash scripts/run_diff_amrsa_pipeline.sh baseline
```

This samples from the upstream Megalodon checkpoint for comparison.

### 8. Analysis tables

```bash
bash scripts/run_diff_amrsa_pipeline.sh analyze
```

## GitHub release policy

The following runtime files are excluded by `.gitignore`:

```text
*.ckpt
*.pt
*.pickle
*.pkl
*.joblib
*.sdf
*.npy
data/raw/
data/processed/
results/
generated/
figures/
```

## Notes on package naming

The repository name is `Diff-AMRSA`. Python import names cannot contain hyphens,
so the importable package is `diff_amrsa`. The `src/megalodon` folder is a
compatibility shim only.
