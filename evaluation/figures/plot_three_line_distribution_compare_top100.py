#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


# ---------- optional SA scorer ----------
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


def calc_sa(mol):
    if SASCORER is None:
        return np.nan
    try:
        return float(SASCORER.calculateScore(mol))
    except Exception:
        return np.nan


# ---------- helpers ----------
def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
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
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
            return mol
        except Exception:
            return None


def compute_props_from_mol(mol):
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "QED": float(QED.qed(mol)),
        "SA": calc_sa(mol),
    }


def get_log10_mic_from_row(row: pd.Series, dataset_type: str):
    """
    dataset_type:
      - reference: original dataset
      - generated: generated top100 csv
    """
    if dataset_type == "reference":
        # priority order for original data
        candidates_direct_log = ["pred_log10_pmic", "log10_mic", "log10_MIC"]
        candidates_linear = [
            "pmic_ug_ml",
            "standard_concentration_ug_ml",
            "standard_concentration",
            "mic_ug_ml",
            "MIC_ug_ml",
            "concentration_ug_ml",
        ]
    else:
        candidates_direct_log = ["pred_log10_pmic", "log10_pred_mic", "log10_MIC"]
        candidates_linear = [
            "pred_pmic_ug_ml",
            "pred_mic_ug_ml",
            "pred_MIC_ug_ml",
            "pmic_ug_ml",
        ]

    for c in candidates_direct_log:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v):
                return float(v)

    for c in candidates_linear:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v) and float(v) > 0:
                return float(np.log10(float(v)))

    return np.nan


def load_original_dataset(path: Path, active_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)

    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if smiles_col is None:
        raise ValueError(f"Cannot find smiles column in original dataset: {path}")

    # if activity label exists and active_only=True, only keep positives
    label_col = detect_col(df, ["activity_label", "label", "activity", "y"])
    if active_only and label_col is not None:
        df = df[df[label_col] == 1].copy()

    rows = []
    for _, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])
        if mol is None:
            continue
        props = compute_props_from_mol(mol)
        props["log10_MIC"] = get_log10_mic_from_row(row, "reference")
        rows.append(props)

    out = pd.DataFrame(rows)
    return out


def load_generated_topn(path: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if smiles_col is None:
        raise ValueError(f"Cannot find smiles column in generated dataset: {path}")

    if top_n > 0:
        df = df.head(top_n).copy()

    rows = []
    for _, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])
        if mol is None:
            continue
        props = compute_props_from_mol(mol)
        props["log10_MIC"] = get_log10_mic_from_row(row, "generated")
        rows.append(props)

    out = pd.DataFrame(rows)
    return out


def clean_numeric(series: pd.Series):
    arr = pd.to_numeric(series, errors="coerce").values.astype(float)
    arr = arr[np.isfinite(arr)]
    return arr


def simple_kde(values: np.ndarray, xs: np.ndarray, bw_scale: float = 1.0):
    """
    Gaussian KDE implemented manually to avoid hard dependency on scipy.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.zeros_like(xs)

    std = np.std(values, ddof=1)
    if std <= 1e-12:
        std = 1.0

    n = len(values)
    # Silverman's rule
    bw = 1.06 * std * (n ** (-1 / 5))
    if bw <= 1e-8:
        bw = max(std * 0.1, 1e-3)

    bw *= bw_scale

    diffs = (xs[:, None] - values[None, :]) / bw
    ys = np.exp(-0.5 * diffs ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
    return ys


def get_display_range(all_arrays: List[np.ndarray], q_low=0.01, q_high=0.99, pad_ratio=0.08):
    merged = np.concatenate([a[np.isfinite(a)] for a in all_arrays if len(a) > 0])
    merged = merged[np.isfinite(merged)]
    if len(merged) == 0:
        return 0.0, 1.0

    lo = np.quantile(merged, q_low)
    hi = np.quantile(merged, q_high)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(merged))
        hi = float(np.max(merged))

    if hi <= lo:
        hi = lo + 1.0

    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad


def plot_kde_three(ax, ref_vals, scratch_vals, rcfr_vals, title, xlabel, bw_scale=1.0):
    ref_vals = clean_numeric(pd.Series(ref_vals))
    scratch_vals = clean_numeric(pd.Series(scratch_vals))
    rcfr_vals = clean_numeric(pd.Series(rcfr_vals))

    lo, hi = get_display_range([ref_vals, scratch_vals, rcfr_vals], q_low=0.01, q_high=0.99, pad_ratio=0.10)
    xs = np.linspace(lo, hi, 400)

    y_ref = simple_kde(ref_vals, xs, bw_scale=bw_scale)
    y_scratch = simple_kde(scratch_vals, xs, bw_scale=bw_scale)
    y_rcfr = simple_kde(rcfr_vals, xs, bw_scale=bw_scale)

    ax.plot(xs, y_ref, label="Known MRSA Actives", linewidth=2.0, color="#1f77b4")
    ax.plot(xs, y_scratch, label="MRSA-Trained Generated Top100", linewidth=2.0, color="#ff7f0e")
    ax.plot(xs, y_rcfr, label="RCFR-Diffusion for MRSA Top100", linewidth=2.0, color="#2ca02c")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.25)


def save_all(fig, outprefix: Path):
    outprefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outprefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outprefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outprefix.with_suffix(".svg"), bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_dataset_csv", required=True)
    parser.add_argument("--scratch_ranked_csv", required=True)
    parser.add_argument("--rcfr_ranked_csv", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--original_active_only", action="store_true")
    parser.add_argument("--bw_scale", type=float, default=1.15,
                        help="Common KDE bandwidth scale for all three lines. Slightly >1 gives smoother and more visually comparable curves without changing data.")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    original_df = load_original_dataset(Path(args.original_dataset_csv), active_only=args.original_active_only)
    scratch_df = load_generated_topn(Path(args.scratch_ranked_csv), args.top_n)
    rcfr_df = load_generated_topn(Path(args.rcfr_ranked_csv), args.top_n)

    # save source numeric tables
    original_df.to_csv(outdir / "original_dataset_numeric_props.csv", index=False, encoding="utf-8-sig")
    scratch_df.to_csv(outdir / "scratch_top100_numeric_props.csv", index=False, encoding="utf-8-sig")
    rcfr_df.to_csv(outdir / "rcfr_top100_numeric_props.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    plot_kde_three(
        axes[0, 0],
        original_df["MW"], scratch_df["MW"], rcfr_df["MW"],
        "(a) MW Distribution", "Molecular Weight", bw_scale=args.bw_scale
    )
    plot_kde_three(
        axes[0, 1],
        original_df["LogP"], scratch_df["LogP"], rcfr_df["LogP"],
        "(b) LogP Distribution", "LogP", bw_scale=args.bw_scale
    )
    plot_kde_three(
        axes[0, 2],
        original_df["TPSA"], scratch_df["TPSA"], rcfr_df["TPSA"],
        "(c) TPSA Distribution", "TPSA", bw_scale=args.bw_scale
    )
    plot_kde_three(
        axes[1, 0],
        original_df["QED"], scratch_df["QED"], rcfr_df["QED"],
        "(d) QED Distribution", "QED", bw_scale=args.bw_scale
    )
    plot_kde_three(
        axes[1, 1],
        original_df["SA"], scratch_df["SA"], rcfr_df["SA"],
        "(e) SA Score Distribution", "SA Score", bw_scale=args.bw_scale
    )
    plot_kde_three(
        axes[1, 2],
        original_df["log10_MIC"], scratch_df["log10_MIC"], rcfr_df["log10_MIC"],
        "(f) log10 MIC Distribution", "log10 MIC (ug/mL)", bw_scale=args.bw_scale
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.98))
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_all(fig, outdir / "three_line_distribution_compare_top100")
    plt.close(fig)

    # summary table
    metrics = ["MW", "LogP", "TPSA", "QED", "SA", "log10_MIC"]
    rows = []
    for name, df in [
        ("Known MRSA Actives", original_df),
        ("MRSA-Trained Generated Top100", scratch_df),
        ("RCFR-Diffusion for MRSA Top100", rcfr_df),
    ]:
        row = {"Group": name, "N": len(df)}
        for m in metrics:
            vals = clean_numeric(df[m])
            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{m}_median"] = float(np.median(vals)) if len(vals) else np.nan
        rows.append(row)

    pd.DataFrame(rows).to_csv(outdir / "three_line_distribution_compare_top100_summary.csv", index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Figure saved to: {outdir / 'three_line_distribution_compare_top100.png'}")
    print(f"Summary saved to: {outdir / 'three_line_distribution_compare_top100_summary.csv'}")


if __name__ == "__main__":
    main()
