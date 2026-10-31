
# Figure label replacement and bilingual SVG export

## Unified labels
- Known MRSA Actives -> 已知抗 MRSA 活性分子
- General-Model Generated -> 通用模型生成分子
- MRSA-Trained Generated -> MRSA 数据从零训练生成分子
- RCFR-Diffusion for MRSA -> 本文模型

## What was changed
1. Hard-coded figure legend/group display names in the plotting scripts were replaced with the unified English names above.
2. SVG text export is preserved by setting `matplotlib.rcParams['svg.fonttype'] = 'none'`.
3. A new utility script `export_svg_bilingual_labels.py` was added to collect all generated `.svg` files and export:
   - `en_svg/` : English SVG copies
   - `zh_svg/` : Chinese SVG copies with only label names translated

## How to run
Step 1: Re-run your plotting scripts **with the same commands as before**. The experimental content is unchanged; only displayed names are updated.

Step 2: After all figures are generated, run one command to collect bilingual SVG figures:

```bash
python evaluation/figures/export_svg_bilingual_labels.py   --input_root <your_figure_output_root>   --output_root <your_figure_output_root>/all_svg_bilingual
```

Example:

```bash
python evaluation/figures/export_svg_bilingual_labels.py   --input_root outputs   --output_root outputs/all_svg_bilingual
```

This will put all `.svg` files into two separate folders:
- `outputs/all_svg_bilingual/en_svg/`
- `outputs/all_svg_bilingual/zh_svg/`
