#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Four-line property distribution comparison.

This script corresponds to the previous experiment:
    plot_three_line_distribution_top100.py

Difference in this version:
    1) Known MRSA Actives are split into two groups:
       - all known MRSA active molecules
       - top-N known MRSA active molecules
    2) Scratch-MRSA and RCFR are still plotted as generated Top-N groups.
    3) English and Chinese SVG files are generated separately.

Default four lines:
    English:
        Known MRSA Actives (All)
        Known MRSA Actives (Top 100)
        MRSA-Trained Generated (Top 100)
        RCFR-Diffusion for MRSA (Top 100)

    Chinese:
        已知抗 MRSA 活性分子（全部）
        已知抗 MRSA 活性分子（前100名）
        MRSA 数据从零训练生成分子（前100名）
        本文模型（前100名）
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


PROPS = ["MW", "LogP", "TPSA", "QED", "SA", "log10_MIC"]

TITLES_EN = {
    "MW": "(a) MW Distribution",
    "LogP": "(b) LogP Distribution",
    "TPSA": "(c) TPSA Distribution",
    "QED": "(d) QED Distribution",
    "SA": "(e) SA Score Distribution",
    "log10_MIC": "(f) log10 MIC Distribution",
}

XLABELS_EN = {
    "MW": "Molecular Weight",
    "LogP": "LogP",
    "TPSA": "TPSA",
    "QED": "QED",
    "SA": "SA Score",
    "log10_MIC": "log10 MIC (ug/mL)",
}

TITLES_ZH = {
    "MW": "（a）分子量分布",
    "LogP": "（b）LogP 分布",
    "TPSA": "（c）拓扑极性表面积分布",
    "QED": "（d）QED 分布",
    "SA": "（e）合成可及性分数分布",
    "log10_MIC": "（f）log10 MIC 分布",
}

XLABELS_ZH = {
    "MW": "分子量",
    "LogP": "LogP",
    "TPSA": "拓扑极性表面积",
    "QED": "QED",
    "SA": "合成可及性分数",
    "log10_MIC": "log10 MIC（ug/mL）",
}

COLORS_EN = {
    "Known MRSA Actives (All)": "#1f77b4",
    "Known MRSA Actives (Top 100)": "#9467bd",
    "MRSA-Trained Generated (Top 100)": "#ff7f0e",
    "RCFR-Diffusion for MRSA (Top 100)": "#2ca02c",
}

COLORS_ZH = {
    "已知抗 MRSA 活性分子（全部）": "#1f77b4",
    "已知抗 MRSA 活性分子（前100名）": "#9467bd",
    "MRSA 数据从零训练生成分子（前100名）": "#ff7f0e",
    "本文模型（前100名）": "#2ca02c",
}

SCORE_DESC_CANDIDATES = [
    "activity_score",
    "standardized_activity_score",
    "activity_label_probability",
    "total_score",
    "score",
    "pActivity",
    "p_activity",
]

MIC_ASC_CANDIDATES = [
    "pmic_ug_ml",
    "standard_concentration_ug_ml",
    "standard_concentration",
    "mic_ug_ml",
    "MIC_ug_ml",
    "MIC",
    "mic",
]


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
        for c in MIC_ASC_CANDIDATES:
            if c in row.index:
                v = pd.to_numeric(row[c], errors="coerce")
                if pd.notna(v) and float(v) > 0:
                    return float(np.log10(float(v)))
        return np.nan

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


def choose_reference_top_column(
    df: pd.DataFrame,
    reference_top_col: str,
    reference_top_order: str,
) -> Tuple[Optional[str], str, str]:
    """
    Returns:
        (column, order, reason)

    order is either "ascending" or "descending".
    """
    if reference_top_col:
        if reference_top_col not in df.columns:
            raise ValueError(f"--reference_top_col not found in reference CSV: {reference_top_col}")
        if reference_top_order == "auto":
            # default user-specified score column: larger is better
            return reference_top_col, "descending", "user-specified score column, default descending"
        return reference_top_col, reference_top_order, "user-specified column and order"

    score_col = detect_col(df, SCORE_DESC_CANDIDATES)
    if score_col is not None:
        return score_col, "descending", "auto-detected score/probability column, larger is better"

    mic_col = detect_col(df, MIC_ASC_CANDIDATES)
    if mic_col is not None:
        return mic_col, "ascending", "auto-detected MIC column, lower MIC is better"

    return None, "none", "no score/MIC column detected; using the first Top-N rows after filtering"


def prepare_reference_frames(
    path: Path,
    active_only: bool,
    activity_col: str,
    known_top_n: int,
    reference_top_col: str,
    reference_top_order: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    raw = pd.read_csv(path)
    smiles_col = detect_col(raw, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in reference file: {path}")

    df = raw.copy()
    if active_only and activity_col in df.columns:
        df = df[df[activity_col] == 1].copy()

    top_col, order, reason = choose_reference_top_column(df, reference_top_col, reference_top_order)
    if top_col is None:
        top_df = df.head(known_top_n).copy()
    else:
        values = pd.to_numeric(df[top_col], errors="coerce")
        ranked = df.assign(__top_sort_value__=values).dropna(subset=["__top_sort_value__"]).copy()
        ascending = order == "ascending"
        ranked = ranked.sort_values("__top_sort_value__", ascending=ascending)
        top_df = ranked.head(known_top_n).drop(columns=["__top_sort_value__"]).copy()

    meta = {
        "reference_csv": str(path),
        "reference_total_rows_after_active_filter": int(len(df)),
        "known_top_n": int(known_top_n),
        "reference_top_column": top_col,
        "reference_top_order": order,
        "reference_top_reason": reason,
        "reference_active_only": bool(active_only),
        "activity_col": activity_col,
    }

    return df, top_df, meta


def reference_to_metrics(df: pd.DataFrame, group_name: str, smiles_col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])
        if mol is None:
            continue
        item = {"Group": group_name}
        item.update(calc_desc(mol))
        item["log10_MIC"] = get_log10_mic_from_row(row, is_reference=True)
        rows.append(item)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No valid molecules loaded for {group_name}")
    return out


def load_generated_topn(path: Path, group_name: str, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in generated file: {path}")

    # Generated ranked CSV is assumed to be already sorted by generation score.
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


def set_language_font(language: str):
    if language == "zh":
        matplotlib.rcParams["font.family"] = [
            "Noto Sans CJK SC",
            "Microsoft YaHei",
            "SimHei",
            "WenQuanYi Micro Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
            "sans-serif",
        ]
    else:
        matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]


def plot_distribution(
    groups: Dict[str, pd.DataFrame],
    colors: Dict[str, str],
    outprefix: Path,
    language: str,
    bw_adjust: float,
    q_low: float,
    q_high: float,
):
    set_language_font(language)

    if language == "zh":
        titles = TITLES_ZH
        xlabels = XLABELS_ZH
        ylabel = "密度"
        ncol = 2
    else:
        titles = TITLES_EN
        xlabels = XLABELS_EN
        ylabel = "Density"
        ncol = 2

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.2))
    axes = axes.flatten()

    for ax, prop in zip(axes, PROPS):
        arrays = [groups[g][prop] for g in groups]
        xlim = combined_xlim(arrays, q_low=q_low, q_high=q_high, pad_ratio=0.05)

        for gname, gdf in groups.items():
            kde_line(
                ax,
                gdf[prop],
                gname,
                colors[gname],
                bw_adjust=bw_adjust,
            )

        if xlim is not None:
            ax.set_xlim(*xlim)

        ax.set_title(titles[prop], fontsize=10)
        ax.set_xlabel(xlabels[prop])
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        frameon=True,
        bbox_to_anchor=(0.5, 1.035),
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    save_fig(fig, outprefix)
    plt.close(fig)


def write_summary(groups: Dict[str, pd.DataFrame], outpath: Path):
    summary_rows = []
    for gname, gdf in groups.items():
        row = {"Group": gname, "N": len(gdf)}
        for p in PROPS:
            vals = pd.to_numeric(gdf[p], errors="coerce")
            row[f"{p}_mean"] = float(vals.mean())
            row[f"{p}_median"] = float(vals.median())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(outpath, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(
        description="Four-line distribution: known all vs known top-N vs generated top-N."
    )

    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--scratch_csv", required=True)
    parser.add_argument("--rcfr_csv", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--known_top_n", type=int, default=100)
    parser.add_argument("--generated_top_n", type=int, default=100)
    parser.add_argument("--reference_active_only", action="store_true")
    parser.add_argument("--activity_col", default="activity_label")

    parser.add_argument(
        "--reference_top_col",
        default="",
        help=(
            "Column used to rank known MRSA molecules. "
            "If omitted, the script automatically uses score/probability columns descending, "
            "or MIC columns ascending."
        ),
    )
    parser.add_argument(
        "--reference_top_order",
        default="auto",
        choices=["auto", "ascending", "descending"],
        help="Sort order for --reference_top_col. For MIC, use ascending; for scores, use descending.",
    )

    parser.add_argument("--bw_adjust", type=float, default=1.15)
    parser.add_argument("--q_low", type=float, default=0.01)
    parser.add_argument("--q_high", type=float, default=0.99)
    parser.add_argument("--language", choices=["en", "zh", "both"], default="both")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reference_raw = pd.read_csv(args.reference_csv)
    smiles_col = detect_col(reference_raw, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in reference file: {args.reference_csv}")

    reference_all_df, reference_top_df, meta = prepare_reference_frames(
        path=Path(args.reference_csv),
        active_only=args.reference_active_only,
        activity_col=args.activity_col,
        known_top_n=args.known_top_n,
        reference_top_col=args.reference_top_col,
        reference_top_order=args.reference_top_order,
    )

    # Need smiles_col after possible filtering; column name is unchanged.
    ref_all_en = reference_to_metrics(reference_all_df, "Known MRSA Actives (All)", smiles_col)
    ref_top_en = reference_to_metrics(reference_top_df, f"Known MRSA Actives (Top {args.known_top_n})", smiles_col)
    scratch_en = load_generated_topn(
        Path(args.scratch_csv),
        f"MRSA-Trained Generated (Top {args.generated_top_n})",
        args.generated_top_n,
    )
    rcfr_en = load_generated_topn(
        Path(args.rcfr_csv),
        f"RCFR-Diffusion for MRSA (Top {args.generated_top_n})",
        args.generated_top_n,
    )

    ref_all_en.to_csv(outdir / "known_mrsa_actives_all_used_for_plot.csv", index=False, encoding="utf-8-sig")
    ref_top_en.to_csv(outdir / f"known_mrsa_actives_top{args.known_top_n}_used_for_plot.csv", index=False, encoding="utf-8-sig")
    scratch_en.to_csv(outdir / f"mrsa_trained_generated_top{args.generated_top_n}_used_for_plot.csv", index=False, encoding="utf-8-sig")
    rcfr_en.to_csv(outdir / f"rcfr_diffusion_top{args.generated_top_n}_used_for_plot.csv", index=False, encoding="utf-8-sig")

    groups_en = {
        "Known MRSA Actives (All)": ref_all_en,
        f"Known MRSA Actives (Top {args.known_top_n})": ref_top_en,
        f"MRSA-Trained Generated (Top {args.generated_top_n})": scratch_en,
        f"RCFR-Diffusion for MRSA (Top {args.generated_top_n})": rcfr_en,
    }
    colors_en = {
        "Known MRSA Actives (All)": "#1f77b4",
        f"Known MRSA Actives (Top {args.known_top_n})": "#9467bd",
        f"MRSA-Trained Generated (Top {args.generated_top_n})": "#ff7f0e",
        f"RCFR-Diffusion for MRSA (Top {args.generated_top_n})": "#2ca02c",
    }

    groups_zh = {
        "已知抗 MRSA 活性分子（全部）": ref_all_en.rename(columns={}),
        f"已知抗 MRSA 活性分子（前{args.known_top_n}名）": ref_top_en.rename(columns={}),
        f"MRSA 数据从零训练生成分子（前{args.generated_top_n}名）": scratch_en.rename(columns={}),
        f"本文模型（前{args.generated_top_n}名）": rcfr_en.rename(columns={}),
    }
    colors_zh = {
        "已知抗 MRSA 活性分子（全部）": "#1f77b4",
        f"已知抗 MRSA 活性分子（前{args.known_top_n}名）": "#9467bd",
        f"MRSA 数据从零训练生成分子（前{args.generated_top_n}名）": "#ff7f0e",
        f"本文模型（前{args.generated_top_n}名）": "#2ca02c",
    }

    if args.language in ["en", "both"]:
        plot_distribution(
            groups_en,
            colors_en,
            outdir / "four_line_known_all_vs_top100_property_distribution_en",
            language="en",
            bw_adjust=args.bw_adjust,
            q_low=args.q_low,
            q_high=args.q_high,
        )
        write_summary(groups_en, outdir / "four_line_known_all_vs_top100_property_distribution_summary_en.csv")

    if args.language in ["zh", "both"]:
        plot_distribution(
            groups_zh,
            colors_zh,
            outdir / "four_line_known_all_vs_top100_property_distribution_zh",
            language="zh",
            bw_adjust=args.bw_adjust,
            q_low=args.q_low,
            q_high=args.q_high,
        )
        write_summary(groups_zh, outdir / "four_line_known_all_vs_top100_property_distribution_summary_zh.csv")

    meta.update({
        "generated_top_n": int(args.generated_top_n),
        "outputs": {
            "english_svg": str(outdir / "four_line_known_all_vs_top100_property_distribution_en.svg"),
            "chinese_svg": str(outdir / "four_line_known_all_vs_top100_property_distribution_zh.svg"),
        },
    })
    (outdir / "four_line_known_all_vs_top100_selection_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Reference Top-N selection:")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("Saved SVG files:")
    if args.language in ["en", "both"]:
        print(outdir / "four_line_known_all_vs_top100_property_distribution_en.svg")
    if args.language in ["zh", "both"]:
        print(outdir / "four_line_known_all_vs_top100_property_distribution_zh.svg")


if __name__ == "__main__":
    main()
