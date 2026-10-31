#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
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


def detect_col(df: pd.DataFrame, candidates):
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


def clip01(x):
    return np.clip(pd.to_numeric(x, errors="coerce").astype(float), 0.0, 1.0)


def get_or_make_qed(df, mols):
    col = detect_col(df, ["qed", "QED"])
    if col is not None:
        return pd.to_numeric(df[col], errors="coerce").astype(float)

    vals = []
    for mol in mols:
        if mol is None:
            vals.append(np.nan)
        else:
            try:
                vals.append(float(QED.qed(mol)))
            except Exception:
                vals.append(np.nan)
    return pd.Series(vals, index=df.index)


def get_or_make_sa(df, mols):
    col = detect_col(df, ["sa", "SA", "sa_score_raw", "SAscore", "SA_score"])
    if col is not None:
        return pd.to_numeric(df[col], errors="coerce").astype(float)

    vals = [calc_sa(mol) for mol in mols]
    return pd.Series(vals, index=df.index)


def get_or_make_mic_score(df):
    col = detect_col(df, ["mic_score", "MIC_score"])
    if col is not None:
        return clip01(df[col])

    mic_col = detect_col(df, ["pred_pmic_ug_ml", "pred_mic_ug_ml", "pmic_ug_ml", "mic_ug_ml"])
    if mic_col is not None:
        mic = pd.to_numeric(df[mic_col], errors="coerce").astype(float)
        log_mic = np.log10(np.clip(mic, 1e-9, None))
        # MIC 越低越好。这里把 0.01 到 100 ug/mL 映射到 1 到 0。
        score = (2.0 - log_mic) / 4.0
        return pd.Series(np.clip(score, 0.0, 1.0), index=df.index)

    log_col = detect_col(df, ["pred_log10_pmic", "log10_mic", "log10_MIC"])
    if log_col is not None:
        log_mic = pd.to_numeric(df[log_col], errors="coerce").astype(float)
        score = (2.0 - log_mic) / 4.0
        return pd.Series(np.clip(score, 0.0, 1.0), index=df.index)

    raise ValueError("Cannot find mic_score or predicted MIC column.")


def get_activity_prob(df):
    col = detect_col(df, [
        "activity_label_probability",
        "activity_probability",
        "activity_prob",
        "pred_activity_prob",
        "probability",
    ])
    if col is None:
        raise ValueError("Cannot find activity probability column.")
    return clip01(df[col])


def get_lipinski_score(df):
    col = detect_col(df, ["lipinski_score", "Lipinski_score", "lipinski"])
    if col is not None:
        return clip01(df[col])
    return pd.Series(1.0, index=df.index)


def get_alert_score(df):
    col = detect_col(df, ["alert_score", "Alert_score"])
    if col is not None:
        return clip01(df[col])

    alert_count_col = detect_col(df, ["alert_count", "alerts", "num_alerts"])
    if alert_count_col is not None:
        alert_count = pd.to_numeric(df[alert_count_col], errors="coerce").fillna(0).astype(float)
        return pd.Series((alert_count <= 0).astype(float), index=df.index)

    return pd.Series(1.0, index=df.index)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--w_mic", type=float, default=0.35)
    parser.add_argument("--w_activity", type=float, default=0.30)
    parser.add_argument("--w_qed", type=float, default=0.15)
    parser.add_argument("--w_sa", type=float, default=0.10)
    parser.add_argument("--w_lipinski", type=float, default=0.05)
    parser.add_argument("--w_alert", type=float, default=0.05)

    parser.add_argument("--sa_good", type=float, default=2.0)
    parser.add_argument("--sa_bad", type=float, default=8.0)

    parser.add_argument("--top_n", type=int, default=0)

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])
    if smiles_col is None:
        raise ValueError(f"Cannot find smiles column. Columns: {list(df.columns)}")

    mols = [mol_from_smiles(x) for x in df[smiles_col]]

    mic_score = get_or_make_mic_score(df)
    activity_prob = get_activity_prob(df)
    qed = get_or_make_qed(df, mols)
    sa = get_or_make_sa(df, mols)
    lipinski_score = get_lipinski_score(df)
    alert_score = get_alert_score(df)

    # SA 越低越好，映射为越高越好的 sa_score。
    # 默认 SA=2 得 1 分，SA=8 得 0 分。
    sa_score = (args.sa_bad - sa) / (args.sa_bad - args.sa_good)
    sa_score = pd.Series(np.clip(sa_score, 0.0, 1.0), index=df.index)

    total_weight = (
        args.w_mic
        + args.w_activity
        + args.w_qed
        + args.w_sa
        + args.w_lipinski
        + args.w_alert
    )

    if abs(total_weight - 1.0) > 1e-6:
        print(f"Warning: weights sum to {total_weight}, not 1.0")

    total_score_sa = (
        args.w_mic * mic_score
        + args.w_activity * activity_prob
        + args.w_qed * qed
        + args.w_sa * sa_score
        + args.w_lipinski * lipinski_score
        + args.w_alert * alert_score
    )

    out = df.copy()

    # 保留原来的 total_score，新增旧分数备份列。
    if "total_score" in out.columns and "total_score_original" not in out.columns:
        out["total_score_original"] = out["total_score"]

    out["mic_score"] = mic_score
    out["activity_label_probability"] = activity_prob
    out["qed"] = qed
    out["sa"] = sa
    out["sa_score"] = sa_score
    out["lipinski_score"] = lipinski_score
    out["alert_score"] = alert_score

    out["total_score_sa"] = total_score_sa
    out["total_score"] = total_score_sa

    out = out.sort_values("total_score_sa", ascending=False).reset_index(drop=True)
    out["rank_sa"] = np.arange(1, len(out) + 1)

    if args.top_n and args.top_n > 0:
        out = out.head(args.top_n).copy()

    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "n_input": len(df),
        "n_output": len(out),
        "w_mic": args.w_mic,
        "w_activity": args.w_activity,
        "w_qed": args.w_qed,
        "w_sa": args.w_sa,
        "w_lipinski": args.w_lipinski,
        "w_alert": args.w_alert,
        "sa_good": args.sa_good,
        "sa_bad": args.sa_bad,
        "top100_sa_mean": float(out.head(100)["sa"].mean()) if len(out) else np.nan,
        "top100_qed_mean": float(out.head(100)["qed"].mean()) if len(out) else np.nan,
        "top100_total_score_sa_mean": float(out.head(100)["total_score_sa"].mean()) if len(out) else np.nan,
    }

    pd.DataFrame([summary]).to_csv(
        output_csv.with_suffix(".summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("Done.")
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Rows: {len(df)} -> {len(out)}")
    print(f"Top100 SA mean: {summary['top100_sa_mean']:.4f}")
    print(f"Top100 QED mean: {summary['top100_qed_mean']:.4f}")
    print(f"Top100 total_score_sa mean: {summary['top100_total_score_sa_mean']:.4f}")


if __name__ == "__main__":
    main()
