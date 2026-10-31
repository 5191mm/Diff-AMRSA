#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Property scatter comparison for the figure containing HBA, HBD,
Rotatable Bonds, and Ring Count.

This is the corrected version of the previous experiment script
`compare_sampling_properties_scatter_three_lines.py`.

It compares:
1. Known MRSA Actives              / 已知抗 MRSA 活性分子
2. MRSA-Trained Generated          / MRSA 数据从零训练生成分子
3. RCFR-Diffusion for MRSA         / 本文模型

It can export English-only and Chinese-only SVG figures.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["svg.fonttype"] = "none"      # keep SVG text edievaluation/tables/vector text
mpl.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# -----------------------------
# Display names
# -----------------------------
EN_LABELS = {
    "reference": "Known MRSA Actives",
    "scratch": "MRSA-Trained Generated",
    "rcfr": "RCFR-Diffusion for MRSA",
    "mean": "Mean",
    "median": "Median",
}

ZH_LABELS = {
    "reference": "已知抗 MRSA 活性分子",
    "scratch": "MRSA 数据从零训练生成分子",
    "rcfr": "本文模型",
    "mean": "均值",
    "median": "中位数",
}

PROPERTY_LABELS_EN = {
    "mw": "Molecular Weight",
    "logp": "LogP",
    "tpsa": "TPSA",
    "qed": "QED",
    "sa": "SA Score",
    "hba": "HBA",
    "hbd": "HBD",
    "rotb": "Rotatable Bonds",
    "rings": "Ring Count",
    "fsp3": "Fraction Csp3",
    "heavy": "Heavy Atom Count",
}

PROPERTY_LABELS_ZH = {
    "mw": "分子量",
    "logp": "脂水分配系数",
    "tpsa": "拓扑极性表面积",
    "qed": "类药性评分",
    "sa": "合成可及性评分",
    "hba": "氢键受体数",
    "hbd": "氢键供体数",
    "rotb": "可旋转键数",
    "rings": "环数量",
    "fsp3": "sp3 碳比例",
    "heavy": "重原子数",
}

COLORS = {
    "reference": "#7f7f7f",  # grey, matching the uploaded figure
    "scratch": "#ff7f0e",    # orange
    "rcfr": "#2ca02c",       # green
}
MEAN_COLOR = "#1f77b4"        # blue diamond
MEDIAN_COLOR = "#d62728"      # red square


def choose_font(language: str) -> None:
    """Set fonts. SVG keeps text; font family helps correct display."""
    if language == "zh":
        mpl.rcParams["font.family"] = [
            "Noto Sans CJK SC",
            "Microsoft YaHei",
            "SimHei",
            "WenQuanYi Micro Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
            "sans-serif",
        ]
    else:
        mpl.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]


def find_smiles_column(df: pd.DataFrame, user_col: str = "auto") -> str:
    if user_col != "auto":
        if user_col not in df.columns:
            raise ValueError(f"SMILES column '{user_col}' not found. Available columns: {list(df.columns)}")
        return user_col
    candidates = ["smiles", "SMILES", "canonical_smiles", "Canonical_SMILES", "mol", "molecule"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find SMILES column automatically. Available columns: {list(df.columns)}")


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
    if prop == "mw":
        return float(Descriptors.MolWt(mol))
    if prop == "logp":
        return float(Crippen.MolLogP(mol))
    if prop == "tpsa":
        return float(rdMolDescriptors.CalcTPSA(mol))
    if prop == "qed":
        return float(QED.qed(mol))
    if prop == "sa":
        # A real SA implementation is not always available in RDKit by default.
        # Keep NaN rather than using an inconsistent proxy.
        return float("nan")
    if prop == "hba":
        return float(rdMolDescriptors.CalcNumHBA(mol))
    if prop == "hbd":
        return float(rdMolDescriptors.CalcNumHBD(mol))
    if prop == "rotb":
        return float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    if prop == "rings":
        return float(rdMolDescriptors.CalcNumRings(mol))
    if prop == "fsp3":
        return float(rdMolDescriptors.CalcFractionCSP3(mol))
    if prop == "heavy":
        return float(rdMolDescriptors.CalcNumHeavyAtoms(mol))
    raise ValueError(f"Unsupported property: {prop}")


def row_from_mol(mol: Chem.Mol, group_key: str, index: int, properties: List[str]) -> Optional[Dict[str, object]]:
    mol = sanitize_mol(mol)
    if mol is None:
        return None
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
    row: Dict[str, object] = {"group_key": group_key, "index": index, "smiles": smiles}
    for prop in properties:
        try:
            row[prop] = calc_property(mol, prop)
        except Exception:
            row[prop] = np.nan
    return row


def load_csv_table(csv_path: Path, group_key: str, top_n: int, properties: List[str], smiles_col: str = "auto", active_only: bool = False, activity_col: str = "activity_label") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    smiles_col = find_smiles_column(df, smiles_col)

    if active_only and activity_col in df.columns:
        vals = pd.to_numeric(df[activity_col], errors="coerce")
        df = df[vals == 1].copy()

    rows = []
    seen = set()
    used = 0
    for _, r in df.iterrows():
        smi = r.get(smiles_col)
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        row = row_from_mol(mol, group_key, used + 1, properties)
        if row is None:
            continue
        can = row["smiles"]
        if can in seen:
            continue
        seen.add(can)
        used += 1
        rows.append(row)
        if top_n > 0 and used >= top_n:
            break

    if not rows:
        raise RuntimeError(f"No valid molecules loaded from CSV: {csv_path}")
    return pd.DataFrame(rows)


def load_sdf_table(sdf_path: Path, group_key: str, top_n: int, properties: List[str]) -> pd.DataFrame:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    rows = []
    seen = set()
    used = 0
    for mol in supplier:
        row = row_from_mol(mol, group_key, used + 1, properties)
        if row is None:
            continue
        can = row["smiles"]
        if can in seen:
            continue
        seen.add(can)
        used += 1
        rows.append(row)
        if top_n > 0 and used >= top_n:
            break

    if not rows:
        raise RuntimeError(f"No valid molecules loaded from SDF: {sdf_path}")
    return pd.DataFrame(rows)


def lang_labels(language: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    return (ZH_LABELS, PROPERTY_LABELS_ZH) if language == "zh" else (EN_LABELS, PROPERTY_LABELS_EN)


def plot_property_scatter(df: pd.DataFrame, properties: List[str], outdir: Path, prefix: str, language: str, seed: int = 42, ncols: int = 0) -> Path:
    choose_font(language)
    group_labels, prop_labels = lang_labels(language)
    group_order = ["reference", "scratch", "rcfr"]

    n_props = len(properties)
    if ncols <= 0:
        ncols = 2 if n_props <= 4 else 4
    nrows = math.ceil(n_props / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.7 * nrows))
    axes = np.array(axes).reshape(-1)

    rng = np.random.default_rng(seed)
    x_centers = {g: i for i, g in enumerate(group_order)}

    legend_handles = []
    for g in group_order:
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="None", markersize=6.5,
                   markerfacecolor=COLORS[g], markeredgecolor=COLORS[g], label=group_labels[g])
        )
    legend_handles.extend([
        Line2D([0], [0], marker="D", linestyle="None", markersize=6.5,
               markerfacecolor=MEAN_COLOR, markeredgecolor=MEAN_COLOR, label=group_labels["mean"]),
        Line2D([0], [0], marker="s", linestyle="None", markersize=6.5,
               markerfacecolor=MEDIAN_COLOR, markeredgecolor=MEDIAN_COLOR, label=group_labels["median"]),
    ])

    for i, prop in enumerate(properties):
        ax = axes[i]
        for g in group_order:
            sub = df[df["group_key"] == g].copy()
            y = pd.to_numeric(sub[prop], errors="coerce").to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            x = np.full(len(y), x_centers[g], dtype=float) + rng.uniform(-0.12, 0.12, size=len(y))
            ax.scatter(x, y, s=18, alpha=0.68, color=COLORS[g], edgecolors="none")

            if len(y) > 0:
                ax.scatter([x_centers[g]], [np.nanmean(y)], s=72, marker="D",
                           facecolors=MEAN_COLOR, edgecolors=MEAN_COLOR, linewidths=0.7, zorder=5)
                ax.scatter([x_centers[g]], [np.nanmedian(y)], s=66, marker="s",
                           facecolors=MEDIAN_COLOR, edgecolors=MEDIAN_COLOR, linewidths=0.7, zorder=5)

        ax.set_xlim(-0.45, len(group_order) - 0.55)
        ax.set_xticks([x_centers[g] for g in group_order])
        ax.set_xticklabels([group_labels[g] for g in group_order], rotation=0)
        ax.set_ylabel(prop_labels.get(prop, prop))
        ax.set_title(prop_labels.get(prop, prop), fontsize=12, pad=8)
        ax.grid(True, linestyle="--", alpha=0.22)

    for j in range(n_props, len(axes)):
        axes[j].axis("off")

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=True,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.4,
        borderaxespad=0.35,
    )

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.085, top=0.86, wspace=0.24, hspace=0.34)
    suffix = "zh" if language == "zh" else "en"
    outpath = outdir / f"{prefix}_{suffix}.svg"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(description="Property scatter comparison: Known MRSA Actives vs Scratch-MRSA vs RCFR-Diffusion for MRSA")
    parser.add_argument("--reference_csv", required=True, help="Known MRSA active molecule CSV")
    parser.add_argument("--scratch_sdf", default="", help="Scratch/MRSA-trained generated SDF")
    parser.add_argument("--rcfr_sdf", default="", help="RCFR generated SDF")
    parser.add_argument("--scratch_csv", default="", help="Alternative Scratch/MRSA-trained generated CSV containing SMILES")
    parser.add_argument("--rcfr_csv", default="", help="Alternative RCFR generated CSV containing SMILES")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--top_n", type=int, default=200)
    parser.add_argument("--reference_smiles_col", default="auto")
    parser.add_argument("--generated_smiles_col", default="auto")
    parser.add_argument("--reference_active_only", action="store_true")
    parser.add_argument("--activity_col", default="activity_label")
    parser.add_argument("--properties", nargs="+", default=["hba", "hbd", "rotb", "rings"],
                        help="Default reproduces the uploaded 2x2 figure")
    parser.add_argument("--language", choices=["en", "zh", "both"], default="both")
    parser.add_argument("--ncols", type=int, default=0, help="0 means auto: 2 columns for <=4 properties, otherwise 4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reference_df = load_csv_table(
        Path(args.reference_csv), "reference", args.top_n, args.properties,
        smiles_col=args.reference_smiles_col,
        active_only=args.reference_active_only,
        activity_col=args.activity_col,
    )

    if args.scratch_sdf:
        scratch_df = load_sdf_table(Path(args.scratch_sdf), "scratch", args.top_n, args.properties)
    elif args.scratch_csv:
        scratch_df = load_csv_table(Path(args.scratch_csv), "scratch", args.top_n, args.properties, smiles_col=args.generated_smiles_col)
    else:
        raise ValueError("Provide either --scratch_sdf or --scratch_csv")

    if args.rcfr_sdf:
        rcfr_df = load_sdf_table(Path(args.rcfr_sdf), "rcfr", args.top_n, args.properties)
    elif args.rcfr_csv:
        rcfr_df = load_csv_table(Path(args.rcfr_csv), "rcfr", args.top_n, args.properties, smiles_col=args.generated_smiles_col)
    else:
        raise ValueError("Provide either --rcfr_sdf or --rcfr_csv")

    all_df = pd.concat([reference_df, scratch_df, rcfr_df], ignore_index=True)
    all_df.to_csv(outdir / "reference_scratch_rcfr_property_table.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for group_key, sub in all_df.groupby("group_key"):
        row: Dict[str, object] = {"group_key": group_key, "molecule_count": int(len(sub))}
        for prop in args.properties:
            vals = pd.to_numeric(sub[prop], errors="coerce")
            row[f"{prop}_mean"] = float(np.nanmean(vals))
            row[f"{prop}_median"] = float(np.nanmedian(vals))
            row[f"{prop}_std"] = float(np.nanstd(vals))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(outdir / "reference_scratch_rcfr_property_summary.csv", index=False, encoding="utf-8-sig")

    outputs = []
    langs = ["en", "zh"] if args.language == "both" else [args.language]
    for lang in langs:
        outputs.append(plot_property_scatter(
            all_df, args.properties, outdir,
            prefix="reference_scratch_rcfr_property_scatter",
            language=lang,
            seed=args.seed,
            ncols=args.ncols,
        ))

    print("Saved figures:")
    for p in outputs:
        print(f"  {p}")
    print(f"Saved data table: {outdir / 'reference_scratch_rcfr_property_table.csv'}")
    print(f"Saved summary table: {outdir / 'reference_scratch_rcfr_property_summary.csv'}")


if __name__ == "__main__":
    main()
