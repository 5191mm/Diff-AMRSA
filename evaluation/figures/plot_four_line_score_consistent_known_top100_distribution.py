#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Four-line property distribution plot:
1) Known MRSA Actives (All)
2) Known MRSA Actives (Top 100 by the same composite score as generated molecules)
3) MRSA-Trained Generated (Top 100)
4) RCFR-Diffusion for MRSA (Top 100)

Key correction:
- The known MRSA Top 100 group is NOT selected only by the raw MIC/concentration.
- It is ranked by the same composite scoring formula used for generated candidates:

  total_score =
      w_mic * mic_score
    + w_activity * activity_score
    + w_qed * qed
    + w_lipinski * lipinski_score
    + w_alert * alert_score

Default weights:
  w_mic = 0.40
  w_activity = 0.35
  w_qed = 0.15
  w_lipinski = 0.05
  w_alert = 0.05

For generated molecules, if total_score already exists, the script uses it.
For known MRSA molecules, the script computes the same score from raw MIC/concentration
and RDKit descriptors. If activity probability is unavailable, known active molecules
use activity_score = 1.0 by default.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED


EN_LABELS = {
    "known_all": "Known MRSA Actives (All)",
    "known_top": "Known MRSA Actives (Top 100)",
    "scratch": "MRSA-Trained Generated (Top 100)",
    "rcfr": "RCFR-Diffusion for MRSA (Top 100)",
}
ZH_LABELS = {
    "known_all": "已知抗 MRSA 活性分子（全部）",
    "known_top": "已知抗 MRSA 活性分子（前100）",
    "scratch": "MRSA 数据从零训练生成分子（前100）",
    "rcfr": "本文模型（前100）",
}
EN_PROP_LABELS = {
    "MW": "MW",
    "LogP": "LogP",
    "TPSA": "TPSA",
    "QED": "QED",
    "SA": "SA Score",
    "log10_MIC": "log10 MIC",
}
ZH_PROP_LABELS = {
    "MW": "分子量",
    "LogP": "LogP",
    "TPSA": "TPSA",
    "QED": "QED",
    "SA": "SA 分数",
    "log10_MIC": "log10 MIC",
}

GROUP_COLORS = {
    "known_all": "#7f7f7f",
    "known_top": "#1f77b4",
    "scratch": "#ff7f0e",
    "rcfr": "#2ca02c",
}


def first_existing(columns, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for name in candidates:
        if name in columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    # fuzzy contains
    for c in columns:
        cl = c.lower()
        for name in candidates:
            nl = name.lower()
            if nl in cl:
                return c
    return None


def smiles_col(df: pd.DataFrame) -> str:
    c = first_existing(df.columns, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if c is None:
        raise ValueError("Cannot find SMILES column.")
    return c


def mic_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "pred_pmic_ug_ml", "pmic_ug_ml", "standard_concentration_ug_ml",
        "concentration_ug_ml", "mic_ug_ml", "MIC_ug_ml", "MIC (ug/mL)",
        "mic", "MIC", "value", "activity_value", "standard_value",
        "standard_concentration", "activity_num", "activity_value_ug_ml",
    ]
    return first_existing(df.columns, candidates)


def activity_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "activity_label_probability", "activity_probability", "prob_active",
        "active_probability", "activity_score", "activity_label", "label", "active",
        "activity_label_binary",
    ]
    return first_existing(df.columns, candidates)


def canonicalize(s: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def lipinski_score(mol: Chem.Mol) -> float:
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rules = [
        mw <= 500,
        logp <= 5,
        hbd <= 5,
        hba <= 10,
    ]
    return float(sum(rules)) / 4.0


def alert_score_default(mol: Chem.Mol) -> float:
    # Conservative default without a full PAINS/alert SMARTS list:
    # keep same scoring slot, but do not punish if explicit alerts are unavailable.
    return 1.0


def compute_basic_props(df: pd.DataFrame, smi_col: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        smi = r.get(smi_col)
        can = canonicalize(smi)
        if can is None:
            continue
        mol = Chem.MolFromSmiles(can)
        if mol is None:
            continue
        d = r.to_dict()
        d["smiles"] = can
        d["MW"] = Descriptors.MolWt(mol)
        d["LogP"] = Crippen.MolLogP(mol)
        d["TPSA"] = rdMolDescriptors.CalcTPSA(mol)
        d["QED"] = float(QED.qed(mol))
        # SA may exist in generated CSV. For reference without SA model, keep NaN unless present.
        if "sa" in df.columns:
            d["SA"] = r.get("sa")
        elif "SA" in df.columns:
            d["SA"] = r.get("SA")
        elif "sa_score" in df.columns:
            d["SA"] = r.get("sa_score")
        else:
            d["SA"] = np.nan
        d["lipinski_score_computed"] = lipinski_score(mol)
        d["alert_score_computed"] = alert_score_default(mol)
        rows.append(d)
    return pd.DataFrame(rows)


def normalize_mic_to_score(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    # MIC lower is better. Convert to log scale then min-max invert.
    logx = np.log10(x.clip(lower=1e-12))
    valid = logx.replace([np.inf, -np.inf], np.nan)
    mn, mx = valid.min(skipna=True), valid.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.where(valid.notna(), 1.0, np.nan), index=values.index)
    return (mx - valid) / (mx - mn)


def get_generated_total_score(df: pd.DataFrame,
                              w_mic: float,
                              w_activity: float,
                              w_qed: float,
                              w_lipinski: float,
                              w_alert: float) -> pd.Series:
    if "total_score" in df.columns:
        return pd.to_numeric(df["total_score"], errors="coerce")
    mic_c = mic_col(df)
    act_c = activity_col(df)
    mic_score = pd.to_numeric(df.get("mic_score", np.nan), errors="coerce") if "mic_score" in df.columns else (
        normalize_mic_to_score(df[mic_c]) if mic_c else pd.Series(np.nan, index=df.index)
    )
    activity = pd.to_numeric(df[act_c], errors="coerce") if act_c else pd.Series(1.0, index=df.index)
    qed = pd.to_numeric(df.get("qed", df.get("QED", np.nan)), errors="coerce")
    lip = pd.to_numeric(df.get("lipinski_score", df.get("lipinski_score_computed", np.nan)), errors="coerce")
    alert = pd.to_numeric(df.get("alert_score", df.get("alert_score_computed", 1.0)), errors="coerce")
    return w_mic * mic_score + w_activity * activity + w_qed * qed + w_lipinski * lip + w_alert * alert


def compute_reference_total_score(df: pd.DataFrame,
                                  w_mic: float,
                                  w_activity: float,
                                  w_qed: float,
                                  w_lipinski: float,
                                  w_alert: float) -> Tuple[pd.Series, Dict]:
    mic_c = mic_col(df)
    act_c = activity_col(df)
    if mic_c is None:
        raise ValueError("Reference CSV has no recognizable MIC/concentration column. Please pass --reference_mic_col.")
    mic_score = normalize_mic_to_score(df[mic_c])
    if act_c is not None:
        activity = pd.to_numeric(df[act_c], errors="coerce")
        # If labels are string actives, map common terms to 1.
        if activity.isna().mean() > 0.8:
            raw = df[act_c].astype(str).str.lower()
            activity = raw.apply(lambda x: 1.0 if any(k in x for k in ["active", "inhib", "yes", "true", "1"]) else 0.0)
    else:
        # These are known MRSA actives after filtering; use activity term = 1.0.
        activity = pd.Series(1.0, index=df.index)
    qed = pd.to_numeric(df["QED"], errors="coerce")
    lip = pd.to_numeric(df["lipinski_score_computed"], errors="coerce")
    alert = pd.to_numeric(df["alert_score_computed"], errors="coerce")
    total = w_mic * mic_score + w_activity * activity + w_qed * qed + w_lipinski * lip + w_alert * alert
    meta = {
        "reference_mic_column": mic_c,
        "reference_activity_column": act_c,
        "activity_term_rule": "Use activity column if available; otherwise use 1.0 for known active molecules.",
        "score_formula": "total_score = w_mic*mic_score + w_activity*activity_score + w_qed*qed + w_lipinski*lipinski_score + w_alert*alert_score",
        "weights": {
            "w_mic": w_mic,
            "w_activity": w_activity,
            "w_qed": w_qed,
            "w_lipinski": w_lipinski,
            "w_alert": w_alert,
        },
        "mic_score_rule": "MIC/concentration lower is better; score is min-max inverted after log10 transform within reference set.",
    }
    return total, meta


def add_log10_mic(df: pd.DataFrame) -> pd.DataFrame:
    c = mic_col(df)
    if c is not None:
        vals = pd.to_numeric(df[c], errors="coerce")
        df["log10_MIC"] = np.log10(vals.clip(lower=1e-12))
    else:
        df["log10_MIC"] = np.nan
    return df


def filter_active_reference(df: pd.DataFrame, active_only: bool) -> pd.DataFrame:
    if not active_only:
        return df
    c = activity_col(df)
    if c is None:
        return df
    s = df[c]
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() > 0:
        return df[num.fillna(0) > 0].copy()
    ss = s.astype(str).str.lower()
    mask = ss.str.contains("active|inhib|yes|true|1", regex=True, na=False)
    return df[mask].copy()


def prepare_reference(reference_csv: Path, active_only: bool, args) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    raw = pd.read_csv(reference_csv)
    raw = filter_active_reference(raw, active_only)
    sc = smiles_col(raw)
    ref = compute_basic_props(raw, sc)
    ref = add_log10_mic(ref)
    ref_total, meta = compute_reference_total_score(
        ref, args.w_mic, args.w_activity, args.w_qed, args.w_lipinski, args.w_alert
    )
    ref["total_score_consistent"] = ref_total
    ref = ref.sort_values("total_score_consistent", ascending=False).reset_index(drop=True)
    top = ref.head(args.known_top_n).copy()
    return ref, top, meta


def prepare_generated(csv_path: Path, top_n: int, args) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    sc = smiles_col(df)
    gen = compute_basic_props(df, sc)
    gen = add_log10_mic(gen)
    gen["total_score_consistent"] = get_generated_total_score(
        gen, args.w_mic, args.w_activity, args.w_qed, args.w_lipinski, args.w_alert
    )
    gen = gen.sort_values("total_score_consistent", ascending=False).reset_index(drop=True)
    return gen.head(top_n).copy()


def kde_curve(vals, xs):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return np.zeros_like(xs)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals)
        y = kde(xs)
        return y
    except Exception:
        hist, edges = np.histogram(vals, bins=30, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        return np.interp(xs, centers, hist, left=0, right=0)


def plot_distribution(groups: Dict[str, pd.DataFrame], outdir: Path, lang: str):
    labels = ZH_LABELS if lang == "zh" else EN_LABELS
    prop_labels = ZH_PROP_LABELS if lang == "zh" else EN_PROP_LABELS
    font_family = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"] if lang == "zh" else ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["font.family"] = font_family
    props = ["MW", "LogP", "TPSA", "QED", "SA", "log10_MIC"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    axes = axes.flatten()
    for ax, prop in zip(axes, props):
        all_vals = []
        for df in groups.values():
            if prop in df.columns:
                v = pd.to_numeric(df[prop], errors="coerce").dropna()
                if len(v):
                    all_vals.append(v)
        if not all_vals:
            ax.axis("off")
            continue
        merged = pd.concat(all_vals)
        q_low, q_high = merged.quantile(0.01), merged.quantile(0.99)
        if q_low == q_high:
            q_low, q_high = merged.min(), merged.max()
        pad = (q_high - q_low) * 0.08 if q_high > q_low else 1.0
        xs = np.linspace(q_low - pad, q_high + pad, 350)
        for key, df in groups.items():
            if prop not in df.columns:
                continue
            y = kde_curve(df[prop], xs)
            ax.plot(xs, y, linewidth=2.0, color=GROUP_COLORS[key], label=labels[key])
        ax.set_title(prop_labels[prop], fontsize=11)
        ax.set_xlabel(prop_labels[prop], fontsize=9)
        ax.set_ylabel("Density" if lang == "en" else "密度", fontsize=9)
        ax.grid(alpha=0.25)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=2, frameon=True, fontsize=9)
    title = "Four-line property distribution with score-consistent known Top 100" if lang == "en" else "四组分子性质分布：已知分子前100按相同综合得分筛选"
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    suffix = "en" if lang == "en" else "zh"
    fig.savefig(outdir / f"four_line_score_consistent_property_distribution_{suffix}.svg", bbox_inches="tight")
    fig.savefig(outdir / f"four_line_score_consistent_property_distribution_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_csv", required=True)
    ap.add_argument("--scratch_csv", required=True)
    ap.add_argument("--rcfr_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--known_top_n", type=int, default=100)
    ap.add_argument("--generated_top_n", type=int, default=100)
    ap.add_argument("--reference_active_only", action="store_true")
    ap.add_argument("--language", choices=["en", "zh", "both"], default="both")

    ap.add_argument("--w_mic", type=float, default=0.40)
    ap.add_argument("--w_activity", type=float, default=0.35)
    ap.add_argument("--w_qed", type=float, default=0.15)
    ap.add_argument("--w_lipinski", type=float, default=0.05)
    ap.add_argument("--w_alert", type=float, default=0.05)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    known_all, known_top, meta = prepare_reference(Path(args.reference_csv), args.reference_active_only, args)
    scratch_top = prepare_generated(Path(args.scratch_csv), args.generated_top_n, args)
    rcfr_top = prepare_generated(Path(args.rcfr_csv), args.generated_top_n, args)

    groups = {
        "known_all": known_all,
        "known_top": known_top,
        "scratch": scratch_top,
        "rcfr": rcfr_top,
    }

    known_all.to_csv(outdir / "known_mrsa_actives_all_scored.csv", index=False, encoding="utf-8-sig")
    known_top.to_csv(outdir / "known_mrsa_actives_top100_same_score.csv", index=False, encoding="utf-8-sig")
    scratch_top.to_csv(outdir / "mrsa_trained_generated_top100_same_score.csv", index=False, encoding="utf-8-sig")
    rcfr_top.to_csv(outdir / "rcfr_diffusion_top100_same_score.csv", index=False, encoding="utf-8-sig")

    meta["known_all_count"] = int(len(known_all))
    meta["known_top_n"] = int(len(known_top))
    meta["generated_top_n"] = int(args.generated_top_n)
    (outdir / "score_consistent_selection_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    if args.language in ("en", "both"):
        plot_distribution(groups, outdir, "en")
    if args.language in ("zh", "both"):
        plot_distribution(groups, outdir, "zh")

    print(f"Saved results to: {outdir}")
    print("Known MRSA Actives Top 100 is selected by total_score_consistent, not by raw MIC alone.")


if __name__ == "__main__":
    main()
