# MRSA Dataset Preparation

Primary entrypoints: `prepare_raw_tables.py`, `prepare_dataset.py`, `build_conformers.py`, and `assign_sample_weights.py`.

`prepare_dataset.py` produces Scheme A metadata and repeat/sample-weight columns. `build_conformers.py` performs bounded-time, resumable RDKit 3D conformer construction and can export PyG training splits. The helper `build_conformers_core.py` contains shared chemistry logic; `utils_data.py` provides the previously missing conversion and training-statistics helpers.

From the repository root:

```bash
python data_processing/prepare_dataset.py --help
python data_processing/build_conformers.py --help
python data_processing/assign_sample_weights.py --help
```

**Dataset protocol:** Do not recompute independent random train/test splits when you intend to reproduce a fixed published split. Export identities from the original PyG split files using `scripts/diff_amrsa/export_splits.py`. Do not use the entire 47,353-row source activity table as a train-only reference library.

**Statistics:** `utils_data.py` computes categorical priors over all processed conformers, while reference 3D geometry summaries use a deterministic subset of up to 5,000 conformers. Full historical datasets or exact paper metrics are not packaged here and must be independently validated.

**Scope note:** This repository concerns anti-MRSA data preparation. GEOM-Drugs is a legitimate upstream dataset name and remains in references to the pretrained model; `geom` has been removed from ambiguous legacy script filenames.
