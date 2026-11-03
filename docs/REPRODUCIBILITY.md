# Reproducibility and verification

## Public release scope

This repository contains the Diff-AMRSA source code, configuration examples, tests, and the model overview figure. It does not contain intermediate activity CSVs, raw datasets, generated SDFs, trained Diff-AMRSA weights, or complete processed three-dimensional train/validation/test files.

The methods described in the paper comprise (A) graph contrastive representation learning, (B) training-only reference retrieval with conditional three-dimensional diffusion, and (C) graph-preserving coordinate refinement through flow matching. The full workflow is documented in [`EXPERIMENT_GUIDE.md`](EXPERIMENT_GUIDE.md).

## Inputs required for full reproduction

1. The original source activity records and preprocessing criteria, with any applicable data-use permissions.
2. The original processed train, validation, and test conformer splits and corresponding model statistics.
3. A checkpoint compatible with the Megalodon architecture and the model configuration.
4. Separately trained graph encoder, retrieval-conditioned diffusion model, and flow-refinement checkpoints.
5. The sampling settings, random seeds, hardware/software environment, and raw generated outputs for each reported experiment.

Keep the reference retrieval index restricted to the training split. Do not silently regenerate different train/test splits or treat the combined source activity table as a training-only library.

## Verified locally

- CPU component tests (`PYTHONPATH=src pytest -q tests/diff_amrsa`)
- Python compilation and shell-script syntax checks
- Python package wheel construction without installing runtime dependencies

## Not established by these checks

- End-to-end training on the complete three-dimensional dataset and matching CUDA/PyG environment
- Checkpoint compatibility for all model variants and sampling configurations
- Numerical agreement with the quantitative results reported in the paper
- Official PoseBusters PB-valid evaluation; the `posebusters_inspired` functions are heuristics and are not substitutes for the official tool
- Experimental anti-MRSA activity of generated molecules

A passing unit-test workflow is evidence of component-level software behavior, not proof of full paper reproduction. Record the exact commit SHA, model checkpoints, input dataset versions, and evaluation commands when reporting final experiments.

## Licensing

The root [`LICENSE.md`](../LICENSE.md) provides the Apache License 2.0. Upstream Megalodon, incorporated third-party components, and model weights retain their respective copyright and licensing requirements; see [`../LICENSE/`](../LICENSE/) and [`../MODEL_CARD_UPSTREAM.md`](../MODEL_CARD_UPSTREAM.md).
