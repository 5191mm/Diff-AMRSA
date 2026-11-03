# Data preparation

The repository does not distribute raw activity tables, processed CSV files, conformer datasets, or training/test splits. Obtain the underlying source records from the respective original data providers under their terms and keep them outside the Git repository.

## Local configuration

From the repository root:

```bash
cp configs/paths.example.env configs/local.env
```

Edit `configs/local.env` to set `DIFF_AMRSA_RAW_DIR`, `DIFF_AMRSA_PROCESSED_DIR`, `DIFF_AMRSA_DATASET_ROOT`, and the remaining paths. The configuration file is ignored by Git.

The optional baseline data-preparation stages can be run separately:

```bash
bash scripts/run_legacy_pipeline.sh --config configs/local.env raw
bash scripts/run_legacy_pipeline.sh --config configs/local.env scheme_a
bash scripts/run_legacy_pipeline.sh --config configs/local.env conformers
```

The processed activity table and Scheme A CSV are created in the configured local output directories. They are not downloaded from this repository. For the complete three-module method, use `docs/EXPERIMENT_GUIDE.md` and the existing, authoritative PyG train/validation/test split files.

**Data isolation:** Build the retrieval library exclusively from the training split. Do not build the library or activity proxy from a combined train/validation/test table.
