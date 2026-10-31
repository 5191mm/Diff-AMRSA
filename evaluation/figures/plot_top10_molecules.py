#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import textwrap
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

ATOM_COLORS = {
    'H': '#DDDDDD', 'C': '#404040', 'N': '#1f77b4', 'O': '#d62728', 'F': '#2ca02c',
    'P': '#ff7f0e', 'S': '#bcbd22', 'Cl': '#2ca02c', 'Br': '#8c564b', 'I': '#9467bd',
    'B': '#17becf', 'Si': '#7f7f7f'
}


def canonicalize(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def set_equal_axis_3d(ax, xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
    if max_range == 0:
        max_range = 1.0
    x_mid = 0.5 * (x.max() + x.min())
    y_mid = 0.5 * (y.max() + y.min())
    z_mid = 0.5 * (z.max() + z.min())
    half = max_range / 2.0
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)


def remove_3d_background(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    try:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
            axis.line.set_color((1, 1, 1, 0))
    except Exception:
        pass
    try:
        ax.set_axis_off()
    except Exception:
        pass


def draw_mol_3d(ax, mol, bond_linewidth=3.4, atom_scale=0.72):
    """
    Original 3D layout kept.
    Only adjusted to a clearer ball-and-stick style:
    1. thicker sticks
    2. slightly smaller balls so balls do not touch each other
    3. remove 3D background frame
    """
    conf = mol.GetConformer()
    xyz = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz.append([pos.x, pos.y, pos.z])

    import numpy as np
    xyz = np.array(xyz, dtype=float)

    # draw sticks first
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ax.plot(
            [xyz[i, 0], xyz[j, 0]],
            [xyz[i, 1], xyz[j, 1]],
            [xyz[i, 2], xyz[j, 2]],
            color='gray', linewidth=bond_linewidth, alpha=0.95, solid_capstyle='round'
        )

    # draw balls with a smaller size so adjacent atoms are separated by visible sticks
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        color = ATOM_COLORS.get(sym, '#7f7f7f')
        base_size = 120 if sym != 'H' else 45
        size = base_size * atom_scale
        ax.scatter(
            xyz[idx, 0], xyz[idx, 1], xyz[idx, 2],
            s=size, color=color, edgecolors='black', linewidths=0.3
        )

    set_equal_axis_3d(ax, xyz)
    ax.set_box_aspect([1, 1, 1])
    remove_3d_background(ax)


def draw_2d_structure_image(mol: Chem.Mol, width: int = 420, height: int = 220):
    mol2d = Chem.Mol(mol)
    try:
        mol2d = Chem.RemoveHs(mol2d)
    except Exception:
        pass
    rdDepictor.SetPreferCoordGen(True)
    try:
        rdDepictor.Compute2DCoords(mol2d)
    except Exception:
        AllChem.Compute2DCoords(mol2d)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.04
    opts.fixedBondLength = 28
    opts.bondLineWidth = 2.0
    drawer.DrawMolecule(mol2d)
    drawer.FinishDrawing()

    import io
    return mpimg.imread(io.BytesIO(drawer.GetDrawingText()), format='png')


def build_sdf_rank_index(sdf_path: Path) -> Dict[int, Chem.Mol]:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    index = {}
    for mol in suppl:
        if mol is None:
            continue
        if mol.HasProp('rank'):
            try:
                rank = int(float(mol.GetProp('rank')))
                index[rank] = mol
            except Exception:
                pass
    return index


def get_mol_for_rank(rank: int, smiles: str, sdf_index: Dict[int, Chem.Mol]) -> Optional[Chem.Mol]:
    mol = sdf_index.get(rank)
    if mol is not None and mol.GetNumConformers() > 0:
        return mol

    can = canonicalize(smiles)
    if can:
        mol = Chem.MolFromSmiles(can)
        if mol is not None:
            mol = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
    if mol is None or mol.GetNumConformers() == 0:
        return None
    return mol


def main():
    parser = argparse.ArgumentParser(description='Plot top molecules with 2D on top and clearer ball-and-stick 3D below')
    parser.add_argument('--ranked_csv', required=True)
    parser.add_argument('--sdf', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--top_n', type=int, default=10)
    parser.add_argument('--bond_linewidth', type=float, default=3.4)
    parser.add_argument('--atom_scale', type=float, default=0.72)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ranked = pd.read_csv(args.ranked_csv).sort_values('rank').head(args.top_n).copy()
    sdf_index = build_sdf_rank_index(Path(args.sdf))

    meta_rows = []
    ncols = 5
    nrows = math.ceil(args.top_n / ncols)

    fig = plt.figure(figsize=(22.5, 11.2), constrained_layout=False)
    outer = GridSpec(nrows, ncols, figure=fig, wspace=0.18, hspace=0.24)

    cell_idx = 0
    for row in ranked.itertuples(index=False):
        rank = int(getattr(row, 'rank'))
        smiles = getattr(row, 'smiles')
        mol = get_mol_for_rank(rank, smiles, sdf_index)
        if mol is None:
            continue

        formula = rdMolDescriptors.CalcMolFormula(mol)
        meta_rows.append({
            'rank': rank,
            'smiles': smiles,
            'formula': formula,
            'pred_pmic_ug_ml': getattr(row, 'pred_pmic_ug_ml', None),
            'activity_label_probability': getattr(row, 'activity_label_probability', None),
            'qed': getattr(row, 'qed', None),
            'total_score': getattr(row, 'total_score', None),
        })

        r = cell_idx // ncols
        c = cell_idx % ncols
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[r, c], height_ratios=[0.90, 1.30], hspace=0.03)

        ax2d = fig.add_subplot(inner[0])
        ax2d.imshow(draw_2d_structure_image(mol))
        ax2d.axis('off')
        title_line_1 = f'Rank {rank} | Formula: {formula}'
        title_line_2 = textwrap.fill(f'SMILES: {smiles}', width=32)
        ax2d.set_title(title_line_1 + '\n' + title_line_2, fontsize=8.2, pad=2)

        ax3d = fig.add_subplot(inner[1], projection='3d')
        draw_mol_3d(ax3d, mol, bond_linewidth=args.bond_linewidth, atom_scale=args.atom_scale)

        cell_idx += 1

    fig_path = outdir / 'top10_generated_mrsa_2d3d.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame(meta_rows).to_csv(outdir / 'top10_generated_mrsa_metadata.csv', index=False, encoding='utf-8-sig')
    print(f'Saved figure: {fig_path}')
    print(f'Saved metadata: {outdir / "top10_generated_mrsa_metadata.csv"}')


if __name__ == '__main__':
    main()
