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
        import sascorer  # type: ignore
        return sascorer
    except Exception:
        pass
    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore
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
                import sascorer  # type: ignore
                return sascorer
            except Exception:
                continue
    raise ImportError('Cannot import sascorer from RDKit Contrib SA_Score')


SASCORER = load_sascorer()


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
        y[np.argmin(np.abs(x - values.mean()))] = 1.0
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


def plot_panel(ax, original_values, generated_values, title, xlabel):
    blue = '#1f77b4'
    yellow = '#f5b000'
    x1, y1 = kde_curve(np.asarray(original_values, dtype=float))
    x2, y2 = kde_curve(np.asarray(generated_values, dtype=float))
    if x1.size > 0:
        ax.plot(x1, y1, color=blue, linewidth=2.2, label='Known MRSA Actives')
    if x2.size > 0:
        ax.plot(x2, y2, color=yellow, linewidth=2.2, label='RCFR-Diffusion for MRSA')
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.28)


def main():
    parser = argparse.ArgumentParser(description='Plot original-vs-generated MRSA property distributions')
    parser.add_argument('--original_csv', required=True)
    parser.add_argument('--generated_csv', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--original_active_only', action='store_true')
    parser.add_argument('--max_original', type=int, default=0)
    parser.add_argument('--keep_largest_fragment', action='store_true', default=True)
    parser.add_argument('--no_largest_fragment', action='store_true')
    args = parser.parse_args()

    keep_largest_fragment = args.keep_largest_fragment and not args.no_largest_fragment
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    original_df = pd.read_csv(args.original_csv)
    if args.original_active_only and 'activity_label' in original_df.columns:
        original_df = original_df[original_df['activity_label'] == 1].copy()
    original_df['canonical_smiles'] = original_df['smiles'].map(lambda x: canonicalize_smiles(x, keep_largest_fragment))
    original_df = original_df.dropna(subset=['canonical_smiles']).drop_duplicates(subset=['canonical_smiles']).copy()
    if args.max_original and len(original_df) > args.max_original:
        original_df = original_df.iloc[:args.max_original].copy()

    generated_df = pd.read_csv(args.generated_csv)
    generated_df['canonical_smiles'] = generated_df['smiles'].map(lambda x: canonicalize_smiles(x, keep_largest_fragment))
    generated_df = generated_df.dropna(subset=['canonical_smiles']).drop_duplicates(subset=['canonical_smiles']).copy()

    original_desc = build_descriptor_df_from_smiles(original_df['canonical_smiles'].tolist(), keep_largest_fragment)
    generated_desc = build_descriptor_df_from_smiles(generated_df['canonical_smiles'].tolist(), keep_largest_fragment)

    if original_desc.empty:
        raise RuntimeError('No valid original molecules after descriptor calculation')
    if generated_desc.empty:
        raise RuntimeError('No valid generated molecules after descriptor calculation')

    original_map = original_df.set_index('canonical_smiles')['pmic_ug_ml'].astype(float)
    original_metrics = original_desc.copy()
    original_metrics['mic_log10'] = np.log10(original_map.loc[original_desc['smiles']].clip(lower=1e-12).to_numpy())

    generated_metrics = generated_desc.copy()
    generated_index = generated_df.set_index('canonical_smiles')
    if 'pred_log10_pmic' in generated_df.columns:
        generated_metrics['mic_log10'] = generated_index.loc[generated_desc['smiles'], 'pred_log10_pmic'].astype(float).to_numpy()
    elif 'pred_pmic_ug_ml' in generated_df.columns:
        generated_metrics['mic_log10'] = np.log10(generated_index.loc[generated_desc['smiles'], 'pred_pmic_ug_ml'].astype(float).clip(lower=1e-12).to_numpy())
    else:
        generated_metrics['mic_log10'] = np.nan

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    axes = axes.flatten()

    plot_panel(axes[0], original_metrics['mw'], generated_metrics['mw'], '(a) MW Distribution', 'Molecular Weight')
    plot_panel(axes[1], original_metrics['logp'], generated_metrics['logp'], '(b) LogP Distribution', 'LogP')
    plot_panel(axes[2], original_metrics['tpsa'], generated_metrics['tpsa'], '(c) TPSA Distribution', 'TPSA')
    plot_panel(axes[3], original_metrics['qed'], generated_metrics['qed'], '(d) QED Distribution', 'QED')
    plot_panel(axes[4], original_metrics['sa'], generated_metrics['sa'], '(e) SA Score Distribution', 'SA Score')
    plot_panel(axes[5], original_metrics['mic_log10'], generated_metrics['mic_log10'], '(f) log10 MIC Distribution', 'log10 MIC (ug/mL)')

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=True,
        fontsize=11,
        borderpad=0.6,
        handlelength=2.0,
        columnspacing=1.6,
    )
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor('white')

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.075, top=0.88, wspace=0.18, hspace=0.28)

    fig_path = outdir / 'mrsa_distribution_comparison.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    summary_rows = []
    for source, df in [('Known MRSA Actives', original_metrics), ('RCFR-Diffusion for MRSA', generated_metrics)]:
        for col in ['mw', 'logp', 'tpsa', 'qed', 'sa', 'mic_log10']:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            summary_rows.append({
                'source': source,
                'property': col,
                'count': int(values.shape[0]),
                'mean': float(values.mean()) if len(values) else np.nan,
                'std': float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                'median': float(values.median()) if len(values) else np.nan,
                'min': float(values.min()) if len(values) else np.nan,
                'max': float(values.max()) if len(values) else np.nan,
            })

    pd.DataFrame(summary_rows).to_csv(outdir / 'mrsa_distribution_summary.csv', index=False, encoding='utf-8-sig')
    original_metrics.to_csv(outdir / 'original_mrsa_descriptor_table.csv', index=False, encoding='utf-8-sig')
    generated_metrics.to_csv(outdir / 'generated_mrsa_descriptor_table.csv', index=False, encoding='utf-8-sig')

    print(f'Original valid molecules: {len(original_metrics)}')
    print(f'Generated valid molecules: {len(generated_metrics)}')
    print(f'Saved figure: {fig_path}')


if __name__ == '__main__':
    main()
