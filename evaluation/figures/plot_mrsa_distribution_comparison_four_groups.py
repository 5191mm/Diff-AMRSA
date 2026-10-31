#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = False
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED

RDLogger.DisableLog('rdApp.*')


def load_sascorer():
    try:
        import sascorer
        return sascorer
    except Exception:
        pass
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer
    except Exception:
        pass

    import sys
    import rdkit
    rdkit_base = Path(rdkit.__file__).resolve().parent
    candidates = [
        rdkit_base / 'Contrib' / 'SA_Score',
        rdkit_base.parent / 'Contrib' / 'SA_Score',
        Path(os.environ.get('CONDA_PREFIX', '')) / 'share' / 'RDKit' / 'Contrib' / 'SA_Score',
        Path(os.environ.get('CONDA_PREFIX', '')) / 'Lib' / 'site-packages' / 'rdkit' / 'Contrib' / 'SA_Score',
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            sys.path.insert(0, str(candidate))
            try:
                import sascorer
                return sascorer
            except Exception:
                continue
    raise ImportError('Cannot import sascorer from RDKit Contrib SA_Score')


SASCORER = load_sascorer()

GROUP_COLORS = {
    'Known MRSA Actives': '#1f77b4',
    'General-Model Generated': '#7f7f7f',
    'MRSA-Trained Generated': '#ff7f0e',
    'RCFR-Diffusion for MRSA': '#2ca02c',
}


def remove_isolated_hydrogen_atoms(mol: Chem.Mol) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    remove_ids = []
    for atom in rw.GetAtoms():
        if atom.GetSymbol() == 'H' and atom.GetDegree() == 0:
            remove_ids.append(atom.GetIdx())
    for idx in sorted(remove_ids, reverse=True):
        rw.RemoveAtom(idx)
    mol2 = rw.GetMol()
    if mol2.GetNumAtoms() == 0:
        return None
    return mol2


def choose_largest_organic_fragment(mol: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return None
    if not frags:
        return None

    def score_frag(f: Chem.Mol):
        atoms = list(f.GetAtoms())
        has_carbon = int(any(a.GetSymbol() == 'C' for a in atoms))
        heavy = sum(1 for a in atoms if a.GetAtomicNum() > 1)
        total = len(atoms)
        return (has_carbon, heavy, total)

    frags = sorted(frags, key=score_frag, reverse=True)
    return frags[0]


def mol_from_smiles_robust(smiles: str, keep_largest_fragment: bool = True) -> Optional[Chem.Mol]:
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    except Exception:
        return None
    if mol is None:
        return None

    mol = remove_isolated_hydrogen_atoms(mol)
    if mol is None:
        return None

    if keep_largest_fragment:
        mol = choose_largest_organic_fragment(mol)
        if mol is None:
            return None
        mol = remove_isolated_hydrogen_atoms(mol)
        if mol is None:
            return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    try:
        mol = Chem.RemoveHs(mol, sanitize=True)
    except Exception:
        pass
    return mol


def canonicalize_smiles(smiles: str, keep_largest_fragment: bool = True) -> Optional[str]:
    mol = mol_from_smiles_robust(smiles, keep_largest_fragment=keep_largest_fragment)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def compute_descriptors_from_smiles(smiles: str, keep_largest_fragment: bool = True) -> Optional[Dict[str, float]]:
    can = canonicalize_smiles(smiles, keep_largest_fragment=keep_largest_fragment)
    if can is None:
        return None
    mol = mol_from_smiles_robust(can, keep_largest_fragment=False)
    if mol is None:
        return None
    try:
        return {
            'smiles': can,
            'mw': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'tpsa': rdMolDescriptors.CalcTPSA(mol),
            'qed': QED.qed(mol),
            'sa': float(SASCORER.calculateScore(mol)),
        }
    except Exception:
        return None


def build_descriptor_df_from_smiles(smiles_list: List[str], keep_largest_fragment: bool = True) -> pd.DataFrame:
    rows = []
    for smiles in smiles_list:
        rec = compute_descriptors_from_smiles(smiles, keep_largest_fragment=keep_largest_fragment)
        if rec is not None:
            rows.append(rec)
    return pd.DataFrame(rows)


def kde_curve(values: np.ndarray, grid_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([]), np.array([])
    if np.allclose(values.min(), values.max()):
        x = np.linspace(values.min() - 1.0, values.max() + 1.0, grid_size)
        y = np.zeros_like(x)
        y[np.argmin(np.abs(x - float(values.mean())))] = 1.0
        return x, y
    try:
        from scipy.stats import gaussian_kde
        x = np.linspace(values.min(), values.max(), grid_size)
        y = gaussian_kde(values)(x)
        return x, y
    except Exception:
        hist, edges = np.histogram(values, bins=40, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        x = np.linspace(centers.min(), centers.max(), grid_size)
        y = np.interp(x, centers, hist)
        return x, y


def plot_panel(ax, group_to_values, title, xlabel):
    for label, values in group_to_values.items():
        x, y = kde_curve(np.asarray(values, dtype=float))
        if x.size > 0:
            ax.plot(x, y, linewidth=2.1, label=label, color=GROUP_COLORS[label])
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.28)


def prepare_original_metrics(original_csv: Path, original_active_only: bool, max_original: int, keep_largest_fragment: bool) -> pd.DataFrame:
    original_df = pd.read_csv(original_csv)
    if original_active_only and 'activity_label' in original_df.columns:
        original_df = original_df[original_df['activity_label'] == 1].copy()

    original_df['canonical_smiles'] = original_df['smiles'].map(lambda x: canonicalize_smiles(x, keep_largest_fragment))
    original_df = original_df.dropna(subset=['canonical_smiles']).drop_duplicates(subset=['canonical_smiles']).copy()

    if max_original and len(original_df) > max_original:
        original_df = original_df.iloc[:max_original].copy()

    desc = build_descriptor_df_from_smiles(original_df['canonical_smiles'].tolist(), keep_largest_fragment)
    if desc.empty:
        raise RuntimeError('No valid original molecules after descriptor calculation')

    original_map = original_df.set_index('canonical_smiles')['pmic_ug_ml'].astype(float)
    desc['mic_log10'] = np.log10(original_map.loc[desc['smiles']].clip(lower=1e-12).to_numpy())
    desc['group'] = 'Known MRSA Actives'
    return desc


def prepare_generated_metrics(generated_csv: Path, label: str, top_n: int, keep_largest_fragment: bool) -> pd.DataFrame:
    generated_df = pd.read_csv(generated_csv)

    if top_n > 0 and len(generated_df) > top_n:
        generated_df = generated_df.head(top_n).copy()

    generated_df['canonical_smiles'] = generated_df['smiles'].map(lambda x: canonicalize_smiles(x, keep_largest_fragment))
    generated_df = generated_df.dropna(subset=['canonical_smiles']).drop_duplicates(subset=['canonical_smiles']).copy()

    desc = build_descriptor_df_from_smiles(generated_df['canonical_smiles'].tolist(), keep_largest_fragment)
    if desc.empty:
        raise RuntimeError(f'No valid generated molecules for {label}')

    generated_index = generated_df.set_index('canonical_smiles')
    if 'pred_log10_pmic' in generated_df.columns:
        desc['mic_log10'] = generated_index.loc[desc['smiles'], 'pred_log10_pmic'].astype(float).to_numpy()
    elif 'pred_pmic_ug_ml' in generated_df.columns:
        desc['mic_log10'] = np.log10(
            generated_index.loc[desc['smiles'], 'pred_pmic_ug_ml'].astype(float).clip(lower=1e-12).to_numpy()
        )
    else:
        desc['mic_log10'] = np.nan

    desc['group'] = label
    return desc


def main():
    parser = argparse.ArgumentParser(description='Plot MRSA property distributions for real dataset and three generated groups')
    parser.add_argument('--original_csv', required=True)
    parser.add_argument('--pretrained_baseline_csv', required=True)
    parser.add_argument('--scratch_csv', required=True)
    parser.add_argument('--finetune_csv', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--original_active_only', action='store_true')
    parser.add_argument('--max_original', type=int, default=0)
    parser.add_argument('--top_n_generated', type=int, default=200)
    parser.add_argument('--keep_largest_fragment', action='store_true', default=True)
    parser.add_argument('--no_largest_fragment', action='store_true')
    args = parser.parse_args()

    keep_largest_fragment = args.keep_largest_fragment and not args.no_largest_fragment
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    real_df = prepare_original_metrics(
        original_csv=Path(args.original_csv),
        original_active_only=args.original_active_only,
        max_original=args.max_original,
        keep_largest_fragment=keep_largest_fragment,
    )
    geom_df = prepare_generated_metrics(Path(args.pretrained_baseline_csv), 'General-Model Generated', args.top_n_generated, keep_largest_fragment)
    scratch_df = prepare_generated_metrics(Path(args.scratch_csv), 'MRSA-Trained Generated', args.top_n_generated, keep_largest_fragment)
    finetune_df = prepare_generated_metrics(Path(args.finetune_csv), 'RCFR-Diffusion for MRSA', args.top_n_generated, keep_largest_fragment)

    all_df = pd.concat([real_df, geom_df, scratch_df, finetune_df], ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.4))
    axes = axes.flatten()

    for ax, col, title, xlabel in [
        (axes[0], 'mw', 'MW Distribution', 'Molecular Weight'),
        (axes[1], 'logp', 'LogP Distribution', 'LogP'),
        (axes[2], 'tpsa', 'TPSA Distribution', 'TPSA'),
        (axes[3], 'qed', 'QED Distribution', 'QED'),
        (axes[4], 'sa', 'SA Score Distribution', 'SA Score'),
        (axes[5], 'mic_log10', 'log10 MIC Distribution', 'log10 MIC (ug/mL)'),
    ]:
        group_to_values = {}
        for label in ['Known MRSA Actives', 'General-Model Generated', 'MRSA-Trained Generated', 'RCFR-Diffusion for MRSA']:
            vals = pd.to_numeric(all_df.loc[all_df['group'] == label, col], errors='coerce').dropna().to_numpy()
            group_to_values[label] = vals
        plot_panel(ax, group_to_values, title, xlabel)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=True,
        fontsize=10,
        borderpad=0.6,
        handlelength=2.0,
        columnspacing=1.5,
    )
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor('white')

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.075, top=0.88, wspace=0.18, hspace=0.28)

    fig_path = outdir / 'mrsa_distribution_comparison_four_groups.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    summary_rows = []
    for source, sub in all_df.groupby('group'):
        for col in ['mw', 'logp', 'tpsa', 'qed', 'sa', 'mic_log10']:
            values = pd.to_numeric(sub[col], errors='coerce').dropna()
            summary_rows.append({
                'group': source,
                'property': col,
                'count': int(values.shape[0]),
                'mean': float(values.mean()) if len(values) else np.nan,
                'std': float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                'median': float(values.median()) if len(values) else np.nan,
                'min': float(values.min()) if len(values) else np.nan,
                'max': float(values.max()) if len(values) else np.nan,
            })

    pd.DataFrame(summary_rows).to_csv(outdir / 'mrsa_distribution_comparison_four_groups_summary.csv', index=False, encoding='utf-8-sig')
    all_df.to_csv(outdir / 'mrsa_distribution_comparison_four_groups_descriptor_table.csv', index=False, encoding='utf-8-sig')

    print(f'Saved figure: {fig_path}')
    print(f'Saved summary: {outdir / "mrsa_distribution_comparison_four_groups_summary.csv"}')
    print(f'Saved descriptor table: {outdir / "mrsa_distribution_comparison_four_groups_descriptor_table.csv"}')


if __name__ == '__main__':
    main()