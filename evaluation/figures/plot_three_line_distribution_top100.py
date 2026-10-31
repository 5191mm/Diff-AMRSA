#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


COLORS = {
    "Known MRSA Actives": "#1f77b4",
    "MRSA-Trained Generated Top100": "#ff7f0e",
    "RCFR-Diffusion for MRSA Top100": "#2ca02c",
}

PROPS = [
    "MW",
    "LogP",
    "TPSA",
    "QED",
    "SA",
    "log10_MIC",
]

TITLES = {
    "MW": "(a) MW Distribution",
    "LogP": "(b) LogP Distribution",
    "TPSA": "(c) TPSA Distribution",
    "QED": "(d) QED Distribution",
    "SA": "(e) SA Score Distribution",
    "log10_MIC": "(f) log10 MIC Distribution",
}

XLABELS = {
    "MW": "Molecular Weight",
    "LogP": "LogP",
    "TPSA": "TPSA",
    "QED": "QED",
    "SA": "SA Score",
    "log10_MIC": "log10 MIC (ug/mL)",
}


def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    except Exception:
        return None
    if mol is None:
        return None

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
            return mol
        except Exception:
            return None


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
    return None


SASCORER = load_sascorer()


def calc_sa(mol):
    if SASCORER is None:
        return np.nan
    try:
        return float(SASCORER.calculateScore(mol))
    except Exception:
        return np.nan


def calc_desc(mol):
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "QED": float(QED.qed(mol)),
        "SA": calc_sa(mol),
    }


def get_log10_mic_from_row(row: pd.Series, is_reference: bool):
    if is_reference:
        for c in [
            "pmic_ug_ml",
            "standard_concentration_ug_ml",
            "standard_concentration",
            "mic_ug_ml",
            "MIC_ug_ml",
        ]:
            if c in row.index:
                v = pd.to_numeric(row[c], errors="coerce")
                if pd.notna(v) and float(v) > 0:
                    return float(np.log10(float(v)))
        return np.nan
    else:
        if "pred_log10_pmic" in row.index:
            v = pd.to_numeric(row["pred_log10_pmic"], errors="coerce")
            if pd.notna(v):
                return float(v)

        for c in ["pred_pmic_ug_ml", "pred_mic_ug_ml", "pred_MIC_ug_ml"]:
            if c in row.index:
                v = pd.to_numeric(row[c], errors="coerce")
                if pd.notna(v) and float(v) > 0:
                    return float(np.log10(float(v)))
        return np.nan


def load_reference(path: Path, active_only: bool, activity_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in reference file: {path}")

    if active_only and activity_col in df.columns:
        df = df[df[activity_col] == 1].copy()

    rows = []
    for _, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])
        if mol is None:
            continue

        item = {"Group": "Known MRSA Actives"}
        item.update(calc_desc(mol))
        item["log10_MIC"] = get_log10_mic_from_row(row, is_reference=True)
        rows.append(item)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No valid molecules loaded from reference dataset")
    return out


def load_generated_topn(path: Path, group_name: str, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in generated file: {path}")

    df = df.head(top_n).copy()

    rows = []
    for _, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])
        if mol is None:
            continue

        item = {"Group": group_name}
        item.update(calc_desc(mol))
        item["log10_MIC"] = get_log10_mic_from_row(row, is_reference=False)
        rows.append(item)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No valid molecules loaded from {group_name}")
    return out


def kde_line(ax, values, label, color, bw_adjust=1.0):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values.astype(float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals)
        kde.set_bandwidth(kde.factor * bw_adjust)

        xs = np.linspace(vals.min(), vals.max(), 300)
        ys = kde(xs)
        ax.plot(xs, ys, label=label, color=color, linewidth=2.0)
    except Exception:
        ax.hist(vals, bins=30, density=True, histtype="step", label=label, color=color, linewidth=2.0)


def combined_xlim(arrays, q_low=0.01, q_high=0.99, pad_ratio=0.05):
    vals = []
    for arr in arrays:
        x = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().values.astype(float)
        x = x[np.isfinite(x)]
        if len(x):
            vals.append(x)
    if not vals:
        return None

    allv = np.concatenate(vals)
    lo = np.quantile(allv, q_low)
    hi = np.quantile(allv, q_high)
    if hi <= lo:
        lo, hi = float(np.min(allv)), float(np.max(allv))
    if hi <= lo:
        return None

    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad


def save_fig(fig, outprefix: Path):
    outprefix.parent.mkdir(parents=True, exist_ok=True)
    with open(outprefix.with_suffix(".mplfig.pkl"), "wb") as f:
        pickle.dump(fig, f)
    fig.savefig(outprefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outprefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outprefix.with_suffix(".pdf"), bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--scratch_csv", required=True)
    parser.add_argument("--rcfr_csv", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--reference_active_only", action="store_true")
    parser.add_argument("--activity_col", default="activity_label")
    parser.add_argument("--bw_adjust", type=float, default=1.15)
    parser.add_argument("--q_low", type=float, default=0.01)
    parser.add_argument("--q_high", type=float, default=0.99)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ref = load_reference(Path(args.reference_csv), args.reference_active_only, args.activity_col)
    scratch = load_generated_topn(Path(args.scratch_csv), "MRSA-Trained Generated Top100", args.top_n)
    rcfr = load_generated_topn(Path(args.rcfr_csv), "RCFR-Diffusion for MRSA Top100", args.top_n)

    ref.to_csv(outdir / "reference_used_for_plot.csv", index=False, encoding="utf-8-sig")
    scratch.to_csv(outdir / "scratch_top100_used_for_plot.csv", index=False, encoding="utf-8-sig")
    rcfr.to_csv(outdir / "rcfr_top100_used_for_plot.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    groups = {
        "Known MRSA Actives": ref,
        "MRSA-Trained Generated Top100": scratch,
        "RCFR-Diffusion for MRSA Top100": rcfr,
    }

    for ax, prop in zip(axes, PROPS):
        arrays = [groups[g][prop] for g in groups]
        xlim = combined_xlim(arrays, q_low=args.q_low, q_high=args.q_high, pad_ratio=0.05)

        for gname, gdf in groups.items():
            kde_line(
                ax,
                gdf[prop],
                gname,
                COLORS[gname],
                bw_adjust=args.bw_adjust,
            )

        if xlim is not None:
            ax.set_xlim(*xlim)

        ax.set_title(TITLES[prop], fontsize=10)
        ax.set_xlabel(XLABELS[prop])
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_fig(fig, outdir / "three_line_top100_property_distribution")
    plt.close(fig)

    summary_rows = []
    for gname, gdf in groups.items():
        row = {"Group": gname, "N": len(gdf)}
        for p in PROPS:
            vals = pd.to_numeric(gdf[p], errors="coerce")
            row[f"{p}_mean"] = float(vals.mean())
            row[f"{p}_median"] = float(vals.median())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "three_line_top100_property_distribution_summary.csv", index=False, encoding="utf-8-sig")

    print("Saved:")
    print(outdir / "three_line_top100_property_distribution.png")
    print(outdir / "three_line_top100_property_distribution.svg")
    print(outdir / "three_line_top100_property_distribution.pdf")
    print(outdir / "three_line_top100_property_distribution_summary.csv")


if __name__ == "__main__":
    main()
