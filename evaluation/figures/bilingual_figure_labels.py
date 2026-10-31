
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import argparse, re

GROUP_EN = {
    'Known MRSA Actives': '已知抗 MRSA 活性分子',
    'General-Model Generated': '通用模型生成分子',
    'MRSA-Trained Generated': 'MRSA 数据从零训练生成分子',
    'RCFR-Diffusion for MRSA': '本文模型',
}

PHRASE_EN_TO_ZH = {
    # core group names and frequent variants
    'Known MRSA Actives': '已知抗 MRSA 活性分子',
    'General-Model Generated': '通用模型生成分子',
    'MRSA-Trained Generated': 'MRSA 数据从零训练生成分子',
    'RCFR-Diffusion for MRSA': '本文模型',
    'General-Model Generated Molecules': '通用模型生成分子',
    'MRSA-Trained Generated Molecules': 'MRSA 数据从零训练生成分子',
    'RCFR-Diffusion for MRSA Molecules': '本文模型',
    'Known MRSA Active Molecules': '已知抗 MRSA 活性分子',
    'Known MRSA Active': '已知抗 MRSA 活性分子',
    # titles/subtitles
    'Distribution': '分布',
    'Comparison': '比较',
    'Density': '密度',
    'Metric': '指标',
    'Score': '得分',
    'Property': '性质',
    'Properties': '性质',
    'Molecular Weight': '分子量',
    'Molecule Weight': '分子量',
    'SA Score': 'SA 分数',
    'QED Score': 'QED 分数',
    'log10 MIC (ug/mL)': 'log10 MIC（ug/mL）',
    'log10 MIC': 'log10 MIC',
    'Predicted MIC': '预测 MIC',
    'Activity probability': '活性概率',
    'Top activity probability mean': 'Top 活性概率均值',
    'Top active ratio': 'Top 高活性比例',
    'Top predicted MIC median': 'Top 预测 MIC 中位数',
    'Top QED mean': 'Top QED 均值',
    'Top SA mean': 'Top SA 均值',
    'Top MW mean': 'Top MW 均值',
    'Top LogP mean': 'Top LogP 均值',
    'Top TPSA mean': 'Top TPSA 均值',
    'Validity': '有效性',
    'Connectivity': '连通性',
    'Uniqueness': '唯一性',
    'Novelty': '新颖性',
    'Diversity': '多样性',
    'Nearest-neighbor': '最近邻',
    'Tanimoto': 'Tanimoto',
    'Similarity': '相似度',
    'Structural similarity': '结构相似度',
    'Scaffold': '骨架',
    'Fragment': '片段',
    'Coverage': '覆盖率',
    'New scaffold ratio': '新骨架比例',
    'Mean maximum Tanimoto': '平均最大 Tanimoto 相似度',
    'Median maximum Tanimoto': '中位最大 Tanimoto 相似度',
    'Scaffold coverage': '骨架覆盖率',
    'Key fragment coverage': '关键片段覆盖率',
    'Mean activity proxy score': '平均活性代理评分',
    'High-score candidate ratio': '高评分候选比例',
    'Chemical space': '化学空间',
    'Structure-level proximity and novelty': '结构层面的接近性与新颖性',
    'Chemical-space proximity to MRSA molecules': '与 MRSA 分子的化学空间接近性',
    'Distribution matching to original MRSA data': '与原始 MRSA 数据的分布匹配',
    'Generation Quality Comparison': '生成质量比较',
    'Generation Quality Heatmap': '生成质量热图',
    'Original MRSA data': '原始 MRSA 数据',
    'Original MRSA dataset density': '原始 MRSA 数据集密度',
    'Descriptor distribution distance to MRSA dataset': '相对 MRSA 数据集的描述符分布距离',
    'Mean property distance': '平均性质距离',
    'Quantile distance, lower is better': '分位数距离，越低越好',
    'Distance, lower is better': '距离，越低越好',
    'Hit ratio, higher is better': '命中率，越高越好',
    'Ratio': '比例',
    'Ratio above threshold': '高于阈值的比例',
    'Tanimoto threshold': 'Tanimoto 阈值',
    'Max Tanimoto to MRSA': '相对 MRSA 的最大 Tanimoto',
    'Useful analogue region': '有效类似物区间',
    'Frequency in generated molecules': '在生成分子中的频率',
    'Molecule scaffold coverage': '分子骨架覆盖率',
    'Fragment group coverage': '片段组覆盖率',
    '3D conformation plausibility': '三维构象合理性',
    'UFF success ratio': 'UFF 优化成功率',
    'Energy after mean': '优化后能量均值',
    'Relaxation RMSD mean': '构象优化 RMSD 均值',
    'PCA 1': '主成分 1',
    'PCA 2': '主成分 2',
    'PC1': '主成分 1',
    'PC2': '主成分 2',
    'UMAP 1': 'UMAP 1',
    'UMAP 2': 'UMAP 2',
    'Figure': '图',
    'Supplementary Figure': '补充图',
    'Representative generated candidates': '代表性生成候选分子',
    'Top 100': '前 100',
    'Top 200': '前 200',
}

# also remove specific top annotations in group labels by preserving them if present
GROUP_VARIANTS = [
    ('Known MRSA Actives', '已知抗 MRSA 活性分子'),
    ('General-Model Generated', '通用模型生成分子'),
    ('MRSA-Trained Generated', 'MRSA 数据从零训练生成分子'),
    ('RCFR-Diffusion for MRSA', '本文模型'),
]

EN_FONT_FAMILY = "DejaVu Sans, Arial, Helvetica, sans-serif"
ZH_FONT_FAMILY = "'Noto Sans CJK SC','Microsoft YaHei','SimHei','WenQuanYi Micro Hei','Arial Unicode MS',sans-serif"


def _replace_all(text: str, mapping: dict[str, str]) -> str:
    out = text
    for a, b in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        out = out.replace(a, b)
    return out


def translate_svg_text(svg_text: str) -> str:
    out = svg_text
    # group names with optional suffixes like (Top 100), Top100, top100
    for en, zh in GROUP_VARIANTS:
        out = re.sub(rf'{re.escape(en)}\s*\((Top\s*\d+)\)', lambda m: f"{zh}（{m.group(1).replace('Top','前').replace(' ', '')}）", out)
        out = re.sub(rf'{re.escape(en)}\s+(Top\s*\d+)', lambda m: f"{zh} {m.group(1).replace('Top','前')}", out)
        out = out.replace(en, zh)
    out = _replace_all(out, PHRASE_EN_TO_ZH)
    # make font Chinese-capable
    out = re.sub(r'font-family:[^;"]+', 'font-family:' + ZH_FONT_FAMILY, out)
    if '<svg' in out and 'font-family:' not in out:
        out = out.replace('<svg', '<svg style="font-family:' + ZH_FONT_FAMILY + ';"', 1)
    return out


def normalize_english_svg(svg_text: str) -> str:
    out = svg_text
    # remove any Chinese display names back to English where present
    zh_to_en = {v: k for k, v in GROUP_EN.items()}
    for zh, en in sorted(zh_to_en.items(), key=lambda kv: len(kv[0]), reverse=True):
        out = out.replace(zh, en)
    out = re.sub(r'font-family:[^;"]+', 'font-family:' + EN_FONT_FAMILY, out)
    if '<svg' in out and 'font-family:' not in out:
        out = out.replace('<svg', '<svg style="font-family:' + EN_FONT_FAMILY + ';"', 1)
    return out


def collect_and_translate_svgs(input_root: Path, output_root: Path, pattern: str = '*.svg'):
    en_dir = output_root / 'en_svg'
    zh_dir = output_root / 'zh_svg'
    en_dir.mkdir(parents=True, exist_ok=True)
    zh_dir.mkdir(parents=True, exist_ok=True)
    svgs = sorted(input_root.rglob(pattern))
    svgs = [p for p in svgs if p.is_file() and output_root not in p.parents]
    if not svgs:
        raise SystemExit(f'No SVG files found under {input_root}')
    for src in svgs:
        rel = src.relative_to(input_root)
        safe_name = '__'.join(rel.parts)
        en_target = en_dir / safe_name
        zh_target = zh_dir / safe_name
        text = src.read_text(encoding='utf-8', errors='ignore')
        en_target.write_text(normalize_english_svg(text), encoding='utf-8')
        zh_target.write_text(translate_svg_text(text), encoding='utf-8')
    return len(svgs), en_dir, zh_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_root', required=True)
    ap.add_argument('--output_root', required=True)
    ap.add_argument('--pattern', default='*.svg')
    args = ap.parse_args()
    n, en_dir, zh_dir = collect_and_translate_svgs(Path(args.input_root), Path(args.output_root), args.pattern)
    print(f'Collected {n} SVG files')
    print(f'English SVG folder: {en_dir}')
    print(f'Chinese SVG folder: {zh_dir}')

if __name__ == '__main__':
    main()
