#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = False
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

RDLogger.DisableLog('rdApp.*')

PROPERTY_LABELS = {
    'mw': 'Molecular Weight',
    'logp': 'LogP',
    'tpsa': 'TPSA',
    'qed': 'QED',
    'hba': 'HBA',
    'hbd': 'HBD',
    'rotb': 'Rotatable Bonds',
    'rings': 'Ring Count',
    'fsp3': 'Fraction Csp3',
    'heavy': 'Heavy Atom Count',
}

ORIGINAL_LABEL = 'General-Model Generated'
MRSA_LABEL = 'RCFR-Diffusion for MRSA'
GROUP_ORDER = [ORIGINAL_LABEL, MRSA_LABEL]
GROUP_COLORS = {
    ORIGINAL_LABEL: '#1f77b4',
    MRSA_LABEL: '#ff7f0e',
}
MEAN_COLOR = '#2ca02c'
MEDIAN_COLOR = '#d62728'


def sanitize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol, canonical=True)
        clean = Chem.MolFromSmiles(smi)
        return clean
    except Exception:
        return None


def calc_property(mol: Chem.Mol, prop: str) -> float:
    prop = prop.lower()
    if prop == 'mw':
        return float(Descriptors.MolWt(mol))
    if prop == 'logp':
        return float(Crippen.MolLogP(mol))
    if prop == 'tpsa':
        return float(rdMolDescriptors.CalcTPSA(mol))
    if prop == 'qed':
        return float(QED.qed(mol))
    if prop == 'hba':
        return float(rdMolDescriptors.CalcNumHBA(mol))
    if prop == 'hbd':
        return float(rdMolDescriptors.CalcNumHBD(mol))
    if prop == 'rotb':
        return float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    if prop == 'rings':
        return float(rdMolDescriptors.CalcNumRings(mol))
    if prop == 'fsp3':
        return float(rdMolDescriptors.CalcFractionCSP3(mol))
    if prop == 'heavy':
        return float(rdMolDescriptors.CalcNumHeavyAtoms(mol))
    raise ValueError(f'Unsupported property: {prop}')


def load_sdf_table(sdf_path: Path, label: str, top_n: int, properties: List[str]) -> pd.DataFrame:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    rows = []
    seen = set()
    valid_count = 0

    for mol in supplier:
        mol = sanitize_mol(mol)
        if mol is None:
            continue
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            continue
        if smiles in seen:
            continue
        seen.add(smiles)
        valid_count += 1

        row = {
            'group': label,
            'index': valid_count,
            'smiles': smiles,
        }
        for prop in properties:
            try:
                row[prop] = calc_property(mol, prop)
            except Exception:
                row[prop] = np.nan
        rows.append(row)

        if top_n > 0 and valid_count >= top_n:
            break

    if not rows:
        raise RuntimeError(f'No valid molecules loaded from {sdf_path}')
    return pd.DataFrame(rows)


def plot_property_scatter(df: pd.DataFrame, properties: List[str], outdir: Path, prefix: str, seed: int = 42):
    n_props = len(properties)
    ncols = 4
    nrows = math.ceil(n_props / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.9 * nrows))
    axes = np.array(axes).reshape(-1)

    rng = np.random.default_rng(seed)
    x_centers = {GROUP_ORDER[0]: 0.0, GROUP_ORDER[1]: 1.0}

    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=7,
               markerfacecolor=GROUP_COLORS[GROUP_ORDER[0]], markeredgecolor=GROUP_COLORS[GROUP_ORDER[0]],
               label=GROUP_ORDER[0]),
        Line2D([0], [0], marker='o', linestyle='None', markersize=7,
               markerfacecolor=GROUP_COLORS[GROUP_ORDER[1]], markeredgecolor=GROUP_COLORS[GROUP_ORDER[1]],
               label=GROUP_ORDER[1]),
        Line2D([0], [0], marker='D', linestyle='None', markersize=7,
               markerfacecolor=MEAN_COLOR, markeredgecolor=MEAN_COLOR, label='Mean'),
        Line2D([0], [0], marker='s', linestyle='None', markersize=7,
               markerfacecolor=MEDIAN_COLOR, markeredgecolor=MEDIAN_COLOR, label='Median'),
    ]

    for i, prop in enumerate(properties):
        ax = axes[i]
        for group in GROUP_ORDER:
            sub = df[df['group'] == group].copy()
            x = np.full(len(sub), x_centers[group], dtype=float)
            x = x + rng.uniform(-0.12, 0.12, size=len(sub))
            y = pd.to_numeric(sub[prop], errors='coerce').to_numpy(dtype=float)
            color = GROUP_COLORS[group]

            ax.scatter(x, y, s=24, alpha=0.68, color=color, edgecolors='none')

            mean_y = np.nanmean(y)
            median_y = np.nanmedian(y)
            ax.scatter(
                [x_centers[group]], [mean_y], s=92, marker='D',
                facecolors=MEAN_COLOR, edgecolors=MEAN_COLOR, linewidths=0.8, zorder=5
            )
            ax.scatter(
                [x_centers[group]], [median_y], s=82, marker='s',
                facecolors=MEDIAN_COLOR, edgecolors=MEDIAN_COLOR, linewidths=0.8, zorder=5
            )

        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_ylabel(PROPERTY_LABELS.get(prop, prop))
        ax.set_title(PROPERTY_LABELS.get(prop, prop), fontsize=12, pad=10)
        ax.grid(True, linestyle='--', alpha=0.22)

    for j in range(n_props, len(axes)):
        axes[j].axis('off')

    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=True,
        fontsize=10,
        handlelength=1.4,
        columnspacing=1.8,
        borderaxespad=0.4,
    )

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.86, wspace=0.24, hspace=0.30)
    fig_path = outdir / f'{prefix}_property_scatter.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig_path


def main():
    parser = argparse.ArgumentParser(description='Compare molecular property distributions between two SDF files using scatter subplots')
    parser.add_argument('--original_sdf', required=True, help='Path to original Megalodon SDF')
    parser.add_argument('--mrsa_sdf', required=True, help='Path to MRSA generated SDF')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--top_n', type=int, default=200, help='Number of valid unique molecules to use from each SDF')
    parser.add_argument('--properties', nargs='+', default=['mw', 'logp', 'tpsa', 'qed', 'hba', 'hbd', 'rotb', 'rings'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    original_df = load_sdf_table(Path(args.original_sdf), ORIGINAL_LABEL, args.top_n, args.properties)
    mrsa_df = load_sdf_table(Path(args.mrsa_sdf), MRSA_LABEL, args.top_n, args.properties)
    all_df = pd.concat([original_df, mrsa_df], ignore_index=True)

    fig_path = plot_property_scatter(all_df, args.properties, outdir, prefix='original_vs_mrsa', seed=args.seed)

    all_df.to_csv(outdir / 'original_vs_mrsa_property_table.csv', index=False, encoding='utf-8-sig')

    summary_rows = []
    for group_name, sub in all_df.groupby('group'):
        row = {'group': group_name, 'molecule_count': int(len(sub))}
        for prop in args.properties:
            vals = pd.to_numeric(sub[prop], errors='coerce')
            row[f'{prop}_mean'] = float(np.nanmean(vals))
            row[f'{prop}_median'] = float(np.nanmedian(vals))
            row[f'{prop}_std'] = float(np.nanstd(vals))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(outdir / 'original_vs_mrsa_property_summary.csv', index=False, encoding='utf-8-sig')

    print(f'Original molecules used: {len(original_df)}')
    print(f'MRSA molecules used: {len(mrsa_df)}')
    print(f'Saved figure: {fig_path}')
    print(f'Saved property table: {outdir / "original_vs_mrsa_property_table.csv"}')
    print(f'Saved summary table: {outdir / "original_vs_mrsa_property_summary.csv"}')


if __name__ == '__main__':
    main()
