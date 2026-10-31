# Diff-AMRSA model card

## Scope
Diff-AMRSA combines graph contrastive molecular representation learning, anti-MRSA reference retrieval, retrieval-conditioned 3D molecular diffusion and coordinate-only flow-matching refinement. Its diffusion backbone is based on upstream NVIDIA Megalodon.

## Status
**Research implementation; not an experimentally validated drug discovery system.** Independent unit tests are available. The uploaded repository contains no validated end-to-end Diff-AMRSA checkpoints, complete processed training splits, or independently reproduced clinical activity measurements. The reported paper metrics must be evaluated from newly generated outputs.

## Inputs and outputs
Inputs: curated anti-MRSA structures/labels, Megalodon processed conformers, matching pretrained checkpoint, and a molecule for *retrieval-only conditioning*. Outputs: initial and coordinate-refined candidate structures (SDF), ranking tables, and run manifests.

## Known limitations
- Retrieval library must contain training identities only; do not construct it from the full activity table.
- `posebusters_inspired` is a heuristic and does not compute official PoseBusters PB-valid.
- Activity and MIC values are predictions, not laboratory-verified antibacterial efficacy.
- Synthetic candidates must be checked for stereochemistry, geometry, toxicity, and physical validity before further research.
- Code has not been trained end-to-end in the present execution environment.

## Attribution and licensing
See `LICENSE/`, `LICENSE/third_party.txt`, `MODEL_CARD_UPSTREAM.md`, and `CITATION.cff`. Upstream Megalodon pretrained weights have separate licensing and are not included.
