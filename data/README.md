# Data

This repository does not store raw or processed datasets.

Put the three Figshare data files in the folder configured by `DIFF_AMRSA_RAW_DIR`
in `configs/diff_amrsa_paths.env`. The default is:

```bash
/autodl-fs/data/diff_amrsa_raw
```

Then run:

```bash
bash scripts/run_diff_amrsa_pipeline.sh raw
bash scripts/run_diff_amrsa_pipeline.sh scheme_a
bash scripts/run_diff_amrsa_pipeline.sh conformers
```

Generated raw and processed files are ignored by Git.
