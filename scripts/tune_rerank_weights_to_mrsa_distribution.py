#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors, Crippen, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


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


def detect_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None


def mol_from_smiles(smi):
    if pd.isna(smi):
        return None

    try:
        mol = Chem.MolFromSmiles(str(smi), sanitize=False)
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


def calc_sa(mol):
    if mol is None or SASCORER is None:
        return np.nan
    try:
        return float(SASCORER.calculateScore(mol))
    except Exception:
        return np.nan


def calc_props_from_mol(mol):
    if mol is None:
        return {
            "MW": np.nan,
            "LogP": np.nan,
            "TPSA": np.nan,
            "QED": np.nan,
            "SA": np.nan,
        }

    try:
        mw = float(Descriptors.MolWt(mol))
    except Exception:
        mw = np.nan

    try:
        logp = float(Crippen.MolLogP(mol))
    except Exception:
        logp = np.nan

    try:
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    except Exception:
        tpsa = np.nan

    try:
        qed = float(QED.qed(mol))
    except Exception:
        qed = np.nan

    sa = calc_sa(mol)

    return {
        "MW": mw,
        "LogP": logp,
        "TPSA": tpsa,
        "QED": qed,
        "SA": sa,
    }


def get_log10_mic(row, is_generated):
    if is_generated:
        direct = ["pred_log10_pmic", "log10_pred_mic", "log10_MIC", "log10_mic"]
        linear = ["pred_pmic_ug_ml", "pred_mic_ug_ml", "pmic_ug_ml", "mic_ug_ml"]
    else:
        direct = ["pred_log10_pmic", "log10_mic", "log10_MIC"]
        linear = [
            "pmic_ug_ml",
            "standard_concentration_ug_ml",
            "standard_concentration",
            "mic_ug_ml",
            "MIC_ug_ml",
        ]

    for c in direct:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v):
                return float(v)

    for c in linear:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v) and float(v) > 0:
                return float(np.log10(float(v)))

    return np.nan


def load_dataset(path, is_generated, active_only=False):
    df = pd.read_csv(path)

    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if smiles_col is None:
        raise ValueError(f"Cannot find SMILES column in {path}. Columns: {list(df.columns)}")

    if active_only:
        label_col = detect_col(df, ["activity_label", "label", "activity", "y"])
        if label_col is not None:
            df = df[df[label_col] == 1].copy()

    rows = []

    for idx, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])
        if mol is None:
            continue

        props = calc_props_from_mol(mol)
        props["log10_MIC"] = get_log10_mic(row, is_generated=is_generated)

        out = row.to_dict()
        out.update(props)
        out["_source_index"] = idx
        out["_valid_mol"] = 1

        rows.append(out)

    out_df = pd.DataFrame(rows)
    return out_df


def clip01(x):
    return np.clip(pd.to_numeric(x, errors="coerce").astype(float), 0.0, 1.0)


def robust_ref_stats(ref_df, metrics):
    stats = {}
    for m in metrics:
        vals = pd.to_numeric(ref_df[m], errors="coerce").dropna().astype(float).values
        if len(vals) == 0:
            stats[m] = {"median": 0.0, "iqr": 1.0}
            continue

        q25, q50, q75 = np.quantile(vals, [0.25, 0.50, 0.75])
        iqr = q75 - q25
        if not np.isfinite(iqr) or iqr <= 1e-9:
            iqr = np.std(vals) if np.std(vals) > 1e-9 else 1.0

        stats[m] = {
            "median": float(q50),
            "iqr": float(iqr),
        }
    return stats


def add_component_scores(df, ref_stats, metrics):
    out = df.copy()

    # mic_score: MIC 越低越好
    if "mic_score" in out.columns:
        out["_mic_score"] = clip01(out["mic_score"])
    else:
        log_mic = pd.to_numeric(out["log10_MIC"], errors="coerce").astype(float)
        # 0.01 到 100 ug/mL 映射到 1 到 0
        out["_mic_score"] = np.clip((2.0 - log_mic) / 4.0, 0.0, 1.0)

    prob_col = detect_col(out, [
        "activity_label_probability",
        "activity_probability",
        "activity_prob",
        "pred_activity_prob",
        "probability",
    ])
    if prob_col is not None:
        out["_activity_score"] = clip01(out[prob_col])
    else:
        out["_activity_score"] = 0.5

    out["_qed_score"] = clip01(out["QED"])

    # SA 越低越好。SA=2 -> 1, SA=8 -> 0。
    out["_sa_score"] = np.clip((8.0 - pd.to_numeric(out["SA"], errors="coerce").astype(float)) / 6.0, 0.0, 1.0)

    lip_col = detect_col(out, ["lipinski_score", "Lipinski_score", "lipinski"])
    if lip_col is not None:
        out["_lipinski_score"] = clip01(out[lip_col])
    else:
        out["_lipinski_score"] = 1.0

    alert_col = detect_col(out, ["alert_score", "Alert_score"])
    alert_count_col = detect_col(out, ["alert_count", "alerts", "num_alerts"])

    if alert_col is not None:
        out["_alert_score"] = clip01(out[alert_col])
    elif alert_count_col is not None:
        alert_count = pd.to_numeric(out[alert_count_col], errors="coerce").fillna(0)
        out["_alert_score"] = (alert_count <= 0).astype(float)
    else:
        out["_alert_score"] = 1.0

    # 与原始 MRSA 分布的单分子接近性分数
    dists = []
    for m in metrics:
        vals = pd.to_numeric(out[m], errors="coerce").astype(float)
        median = ref_stats[m]["median"]
        iqr = ref_stats[m]["iqr"]
        z = np.abs(vals - median) / iqr
        z = np.clip(z, 0.0, 10.0)
        dists.append(z)

    dist_mat = np.vstack(dists).T
    mean_robust_distance = np.nanmean(dist_mat, axis=1)

    # 距离越小越好，转成 0~1 分数
    out["_property_match_score"] = 1.0 / (1.0 + mean_robust_distance)
    out["_property_match_distance_single"] = mean_robust_distance

    return out


def distribution_distance(selected_df, ref_df, metrics):
    qs = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    total = 0.0
    used = 0

    for m in metrics:
        ref_vals = pd.to_numeric(ref_df[m], errors="coerce").dropna().astype(float).values
        sel_vals = pd.to_numeric(selected_df[m], errors="coerce").dropna().astype(float).values

        if len(ref_vals) < 5 or len(sel_vals) < 5:
            continue

        ref_q = np.quantile(ref_vals, qs)
        sel_q = np.quantile(sel_vals, qs)

        scale = np.quantile(ref_vals, 0.75) - np.quantile(ref_vals, 0.25)
        if scale <= 1e-9:
            scale = np.std(ref_vals) if np.std(ref_vals) > 1e-9 else 1.0

        d = np.mean(np.abs(ref_q - sel_q) / scale)

        total += float(d)
        used += 1

    return total / used if used else np.nan


def apply_score(df, weights):
    score = (
        weights["w_mic"] * df["_mic_score"]
        + weights["w_activity"] * df["_activity_score"]
        + weights["w_qed"] * df["_qed_score"]
        + weights["w_sa"] * df["_sa_score"]
        + weights["w_property"] * df["_property_match_score"]
        + weights["w_lipinski"] * df["_lipinski_score"]
        + weights["w_alert"] * df["_alert_score"]
    )
    return score


def normalize_weights(w):
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def parse_grid(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_dataset_csv", required=True)
    parser.add_argument("--scratch_ranked_csv", required=True)
    parser.add_argument("--rcfr_ranked_csv", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--original_active_only", action="store_true")

    parser.add_argument("--metrics", default="MW,LogP,TPSA,QED,SA,log10_MIC")

    parser.add_argument("--grid_mic", default="0.20,0.25,0.30,0.35")
    parser.add_argument("--grid_activity", default="0.20,0.25,0.30,0.35")
    parser.add_argument("--grid_qed", default="0.05,0.10,0.15")
    parser.add_argument("--grid_sa", default="0.10,0.15,0.20,0.25")
    parser.add_argument("--grid_property", default="0.15,0.20,0.25,0.30,0.35")
    parser.add_argument("--w_lipinski", type=float, default=0.05)
    parser.add_argument("--w_alert", type=float, default=0.05)

    parser.add_argument(
        "--selection_mode",
        choices=["rcfr_closest", "rcfr_margin_over_scratch"],
        default="rcfr_margin_over_scratch",
    )
    parser.add_argument("--margin_weight", type=float, default=0.50)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]

    print("Loading datasets...")
    ref = load_dataset(Path(args.original_dataset_csv), is_generated=False, active_only=args.original_active_only)
    scratch = load_dataset(Path(args.scratch_ranked_csv), is_generated=True, active_only=False)
    rcfr = load_dataset(Path(args.rcfr_ranked_csv), is_generated=True, active_only=False)

    ref_stats = robust_ref_stats(ref, metrics)

    scratch = add_component_scores(scratch, ref_stats, metrics)
    rcfr = add_component_scores(rcfr, ref_stats, metrics)

    grids = {
        "w_mic": parse_grid(args.grid_mic),
        "w_activity": parse_grid(args.grid_activity),
        "w_qed": parse_grid(args.grid_qed),
        "w_sa": parse_grid(args.grid_sa),
        "w_property": parse_grid(args.grid_property),
    }

    records = []
    best = None

    for w_mic, w_activity, w_qed, w_sa, w_property in itertools.product(
        grids["w_mic"],
        grids["w_activity"],
        grids["w_qed"],
        grids["w_sa"],
        grids["w_property"],
    ):
        raw_w = {
            "w_mic": w_mic,
            "w_activity": w_activity,
            "w_qed": w_qed,
            "w_sa": w_sa,
            "w_property": w_property,
            "w_lipinski": args.w_lipinski,
            "w_alert": args.w_alert,
        }

        w = normalize_weights(raw_w)

        scratch_score = apply_score(scratch, w)
        rcfr_score = apply_score(rcfr, w)

        scratch_top = scratch.assign(total_score_distribution_fit=scratch_score).sort_values(
            "total_score_distribution_fit", ascending=False
        ).head(args.top_n)

        rcfr_top = rcfr.assign(total_score_distribution_fit=rcfr_score).sort_values(
            "total_score_distribution_fit", ascending=False
        ).head(args.top_n)

        scratch_dist = distribution_distance(scratch_top, ref, metrics)
        rcfr_dist = distribution_distance(rcfr_top, ref, metrics)

        margin = scratch_dist - rcfr_dist

        if args.selection_mode == "rcfr_closest":
            objective = rcfr_dist
        else:
            # 越小越好：既要求 RCFR 自身接近原数据，也鼓励它相对 Scratch 更接近
            objective = rcfr_dist - args.margin_weight * margin
            if margin <= 0:
                objective += 10.0

        rec = {
            **w,
            "scratch_distribution_distance": scratch_dist,
            "rcfr_distribution_distance": rcfr_dist,
            "margin_scratch_minus_rcfr": margin,
            "objective": objective,
            "scratch_top100_SA_mean": float(scratch_top["SA"].mean()),
            "rcfr_top100_SA_mean": float(rcfr_top["SA"].mean()),
            "scratch_top100_QED_mean": float(scratch_top["QED"].mean()),
            "rcfr_top100_QED_mean": float(rcfr_top["QED"].mean()),
            "scratch_top100_property_match_score_mean": float(scratch_top["_property_match_score"].mean()),
            "rcfr_top100_property_match_score_mean": float(rcfr_top["_property_match_score"].mean()),
        }

        records.append(rec)

        if best is None or rec["objective"] < best["objective"]:
            best = rec

    grid_df = pd.DataFrame(records).sort_values("objective").reset_index(drop=True)
    grid_df.to_csv(outdir / "weight_grid_search_results.csv", index=False, encoding="utf-8-sig")

    best_weights = {k: best[k] for k in [
        "w_mic",
        "w_activity",
        "w_qed",
        "w_sa",
        "w_property",
        "w_lipinski",
        "w_alert",
    ]}

    scratch["total_score_distribution_fit"] = apply_score(scratch, best_weights)
    rcfr["total_score_distribution_fit"] = apply_score(rcfr, best_weights)

    scratch_ranked = scratch.sort_values("total_score_distribution_fit", ascending=False).reset_index(drop=True)
    rcfr_ranked = rcfr.sort_values("total_score_distribution_fit", ascending=False).reset_index(drop=True)

    scratch_ranked["rank_distribution_fit"] = np.arange(1, len(scratch_ranked) + 1)
    rcfr_ranked["rank_distribution_fit"] = np.arange(1, len(rcfr_ranked) + 1)

    scratch_ranked.to_csv(outdir / "scratch_mrsa_ranked_distribution_fit.csv", index=False, encoding="utf-8-sig")
    rcfr_ranked.to_csv(outdir / "rcfr_mrsa_ranked_distribution_fit.csv", index=False, encoding="utf-8-sig")

    scratch_ranked.head(args.top_n).to_csv(outdir / "scratch_mrsa_top100_distribution_fit.csv", index=False, encoding="utf-8-sig")
    rcfr_ranked.head(args.top_n).to_csv(outdir / "rcfr_mrsa_top100_distribution_fit.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([best]).to_csv(outdir / "best_weight_summary.csv", index=False, encoding="utf-8-sig")

    ref.to_csv(outdir / "original_mrsa_numeric_props_used.csv", index=False, encoding="utf-8-sig")

    print("\nBest weights:")
    for k, v in best_weights.items():
        print(f"{k}: {v:.4f}")

    print("\nBest result:")
    print(pd.DataFrame([best]).to_string(index=False))

    print("\nSaved:")
    print(outdir / "weight_grid_search_results.csv")
    print(outdir / "best_weight_summary.csv")
    print(outdir / "scratch_mrsa_ranked_distribution_fit.csv")
    print(outdir / "rcfr_mrsa_ranked_distribution_fit.csv")
    print(outdir / "scratch_mrsa_top100_distribution_fit.csv")
    print(outdir / "rcfr_mrsa_top100_distribution_fit.csv")


if __name__ == "__main__":
    main()
