#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


# =========================
# helper
# =========================
def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    except Exception:
        return None
    if mol is None:
        return None

    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if frags:
            def frag_score(frag):
                heavy = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() > 1)
                carbon = int(any(a.GetSymbol() == "C" for a in frag.GetAtoms()))
                return (carbon, heavy, frag.GetNumAtoms())
            mol = sorted(frags, key=frag_score, reverse=True)[0]
    except Exception:
        return None

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
        except Exception:
            return None

    try:
        smi = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        pass

    return mol


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

    try:
        import rdkit
        rdkit_base = Path(rdkit.__file__).resolve().parent
        candidates = [
            rdkit_base / "Contrib" / "SA_Score",
            rdkit_base.parent / "Contrib" / "SA_Score",
            Path(os.environ.get("CONDA_PREFIX", "")) / "share" / "RDKit" / "Contrib" / "SA_Score",
        ]
        for c in candidates:
            if c.exists():
                sys.path.insert(0, str(c))
                import sascorer
                return sascorer
    except Exception:
        pass

    return None


SASCORER = load_sascorer()


def calc_sa(mol: Chem.Mol) -> float:
    if SASCORER is None:
        return float("nan")
    try:
        return float(SASCORER.calculateScore(mol))
    except Exception:
        return float("nan")


def calc_desc(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "QED": float(QED.qed(mol)),
        "SA": float(calc_sa(mol)),
    }


def get_numeric(row: pd.Series, candidates: List[str]) -> float:
    for c in candidates:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v):
                return float(v)
    return float("nan")


def get_log10_mic_reference(row: pd.Series) -> float:
    # 优先使用已有标准浓度列
    candidates = [
        "pmic_ug_ml",
        "standard_concentration_ug_ml",
        "standard_concentration",
        "mic_ug_ml",
        "MIC_ug_ml",
        "concentration_ug_ml",
        "value",
        "Value",
    ]
    for c in candidates:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v) and float(v) > 0:
                return float(np.log10(float(v)))
    return float("nan")


def get_log10_mic_generated(row: pd.Series) -> float:
    # 先用 pred_log10_pmic
    if "pred_log10_pmic" in row.index:
        v = pd.to_numeric(row["pred_log10_pmic"], errors="coerce")
        if pd.notna(v):
            return float(v)

    candidates = [
        "pred_pmic_ug_ml",
        "pred_mic_ug_ml",
        "pred_MIC_ug_ml",
    ]
    for c in candidates:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v) and float(v) > 0:
                return float(np.log10(float(v)))
    return float("nan")


# =========================
# load data
# =========================
def load_reference_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {path}")

    rows = []
    for _, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])
        if mol is None:
            continue
        desc = calc_desc(mol)
        rows.append({
            "Group": "Known MRSA Actives",
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            **desc,
            "log10_MIC": get_log10_mic_reference(row),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No valid reference molecules loaded from {path}")
    return out


def load_generated_data(path: Path, group_name: str, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {path}")

    if top_n > 0:
        df = df.head(top_n).copy()

    rows = []
    for _, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])
        if mol is None:
            continue
        desc = calc_desc(mol)
        rows.append({
            "Group": group_name,
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            **desc,
            "log10_MIC": get_log10_mic_generated(row),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No valid generated molecules loaded from {path}")
    return out


# =========================
# kde
# =========================
def density_curve(values: np.ndarray, xs: np.ndarray) -> np.ndarray:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values.astype(float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.zeros_like(xs)

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals)
        ys = kde(xs)
        return ys
    except Exception:
        # fallback histogram density interpolation
        hist, edges = np.histogram(vals, bins=50, density=True)
        mids = (edges[:-1] + edges[1:]) / 2
        return np.interp(xs, mids, hist, left=0, right=0)


def choose_xlim(all_series: List[np.ndarray], metric: str) -> Tuple[float, float]:
    vals = []
    for arr in all_series:
        arr = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().values.astype(float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            vals.append(arr)
    if not vals:
        return (0.0, 1.0)

    merged = np.concatenate(vals)

    if metric == "QED":
        return (0.0, 1.0)

    q1 = np.percentile(merged, 1)
    q99 = np.percentile(merged, 99)

    if metric == "SA":
        lo = max(0.0, q1 - 0.5)
        hi = q99 + 0.5
        return (lo, hi)

    if metric == "LogP":
        lo = min(q1 - 1.0, np.min(merged))
        hi = max(q99 + 1.0, np.max(merged))
        return (lo, hi)

    if metric == "log10_MIC":
        lo = min(q1 - 0.5, np.min(merged))
        hi = max(q99 + 0.5, np.max(merged))
        return (lo, hi)

    lo = max(0.0, q1 - 0.05 * abs(q99 - q1 + 1e-8))
    hi = q99 + 0.08 * abs(q99 - q1 + 1e-8)
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


# =========================
# plot
# =========================
def plot_distributions(
    ref_df: pd.DataFrame,
    rcfr_df: pd.DataFrame,
    scratch_df: pd.DataFrame,
    outdir: Path,
):
    metrics = [
        ("MW", "Molecular Weight", "(a) MW Distribution"),
        ("LogP", "LogP", "(b) LogP Distribution"),
        ("TPSA", "TPSA", "(c) TPSA Distribution"),
        ("QED", "QED", "(d) QED Distribution"),
        ("SA", "SA Score", "(e) SA Score Distribution"),
        ("log10_MIC", "log10 MIC (ug/mL)", "(f) log10 MIC Distribution"),
    ]

    colors = {
        "Known MRSA Actives": "#1f77b4",
        "RCFR-Diffusion for MRSA (Top 100)": "#f2a900",
        "MRSA-Trained Generated (Top 100)": "#2ca02c",
    }

    groups = [
        ("Known MRSA Actives", ref_df),
        ("RCFR-Diffusion for MRSA (Top 100)", rcfr_df),
        ("MRSA-Trained Generated (Top 100)", scratch_df),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16.8, 8.6))
    axes = axes.flatten()

    for ax, (metric, xlabel, title) in zip(axes, metrics):
        all_series = [gdf[metric].values for _, gdf in groups]
        x_min, x_max = choose_xlim(all_series, metric)
        xs = np.linspace(x_min, x_max, 500)

        for group_name, gdf in groups:
            ys = density_curve(gdf[metric].values, xs)
            ax.plot(xs, ys, linewidth=2.0, label=group_name, color=colors[group_name])

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(outdir / "three_line_distribution_compare.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "three_line_distribution_compare.svg", bbox_inches="tight")
    fig.savefig(outdir / "three_line_distribution_compare.pdf", bbox_inches="tight")
    plt.close(fig)


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--rcfr_csv", required=True)
    parser.add_argument("--scratch_csv", required=True)
    parser.add_argument("--rcfr_top_n", type=int, default=100)
    parser.add_argument("--scratch_top_n", type=int, default=500)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ref_df = load_reference_data(Path(args.reference_csv))
    rcfr_df = load_generated_data(
        Path(args.rcfr_csv),
        group_name="RCFR-Diffusion for MRSA (Top 100)",
        top_n=args.rcfr_top_n,
    )
    scratch_df = load_generated_data(
        Path(args.scratch_csv),
        group_name="MRSA-Trained Generated (Top 100)",
        top_n=args.scratch_top_n,
    )

    merged = pd.concat([ref_df, rcfr_df, scratch_df], ignore_index=True)
    merged.to_csv(outdir / "three_line_distribution_compare_source_data.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for group_name, gdf in [
        ("Known MRSA Actives", ref_df),
        ("RCFR-Diffusion for MRSA (Top 100)", rcfr_df),
        ("MRSA-Trained Generated (Top 100)", scratch_df),
    ]:
        for metric in ["MW", "LogP", "TPSA", "QED", "SA", "log10_MIC"]:
            vals = pd.to_numeric(gdf[metric], errors="coerce").dropna().values.astype(float)
            if len(vals) == 0:
                continue
            summary_rows.append({
                "Group": group_name,
                "Metric": metric,
                "N": len(vals),
                "Mean": float(np.mean(vals)),
                "Median": float(np.median(vals)),
                "Std": float(np.std(vals)),
            })
    pd.DataFrame(summary_rows).to_csv(
        outdir / "three_line_distribution_compare_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_distributions(ref_df, rcfr_df, scratch_df, outdir)

    print("Done.")
    print(f"Reference N = {len(ref_df)}")
    print(f"RCFR top N = {len(rcfr_df)}")
    print(f"Scratch top N = {len(scratch_df)}")
    print(f"Saved figure: {outdir / 'three_line_distribution_compare.png'}")


if __name__ == "__main__":
    main()
