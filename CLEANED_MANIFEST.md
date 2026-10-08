# Cleaned Release Manifest

This release keeps source code, configurations, and documentation required to
reproduce the Diff-AMRSA pipeline. Runtime artifacts and large files are ignored.

## Kept

- `src/diff_amrsa/`: canonical Python package.
- `src/megalodon/`: minimal compatibility shim for legacy checkpoint imports.
- `data_processing/`: raw table preparation, Scheme A construction, and 3D conformer building.
- `scripts/`: training, fine-tuning, sampling, predictor training, and pipeline entrypoints.
- `scripts/conf/drugs/`: baseline, fine-tuning, and scratch configurations.
- `draw/` and `table/`: reproducible analysis scripts used for paper figures and tables.

## Removed from release

- Checkpoints, PyTorch datasets, pickles, generated SDF files, NumPy arrays, model files,
  logs, generated outputs, and cached figures.
- Backup scripts with `v2`, `.bak`, or temporary experimental names.
