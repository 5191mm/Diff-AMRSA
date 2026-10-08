# Diff-AMRSA Model Card

Diff-AMRSA fine-tunes the publicly released Megalodon large diffusion model for
anti-MRSA molecule generation. The model is intended for computational candidate
generation and prioritization. It does not provide clinical or experimental activity
confirmation.

## Intended use

- Generate and rank anti-MRSA candidate molecules.
- Reproduce dissertation experiments involving molecule validity, novelty,
  domain consistency, and candidate ranking.
- Support downstream wet-lab or docking validation workflows.

## Out-of-scope use

- Clinical decision making.
- Direct medical treatment recommendation.
- Claims of experimentally confirmed antibacterial activity without validation.

## Required external assets

- Raw MRSA activity tables from Figshare.
- Megalodon large diffusion pretrained checkpoint.
- Generated checkpoints and datasets are not stored in the Git repository.
