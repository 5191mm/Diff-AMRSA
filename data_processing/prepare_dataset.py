#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Scheme A training table for MRSA molecule generation.

Scheme A rule
- Keep original pmic_ug_ml unchanged
- Use all activity_label 0 and 1 rows for activity classification
- Use MIC regression only for activity_label 1 with 0 < pmic_ug_ml < active_threshold
- Set pmic_train to NaN for activity_label 0 so inactive samples are not treated as exact MIC values
- Add train weights and repeat counts for generation model fine tuning
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Scheme A columns for MRSA cleaned table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default="processed_drug_mic_dataset.csv")
    parser.add_argument("--outdir", type=str, default="mrsa_scheme_a")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--pmic_col", type=str, default="pmic_ug_ml")
    parser.add_argument("--label_col", type=str, default="activity_label")
    parser.add_argument("--active_threshold", type=float, default=100.0)
    parser.add_argument("--very_low_mic", type=float, default=1.0)
    parser.add_argument("--low_mic", type=float, default=10.0)
    parser.add_argument("--inactive_weight", type=float, default=0.5)
    parser.add_argument("--inactive_repeat", type=int, default=1)
    parser.add_argument("--mid_active_repeat", type=int, default=2)
    parser.add_argument("--low_active_repeat", type=int, default=4)
    parser.add_argument("--very_low_active_repeat", type=int, default=6)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, args: argparse.Namespace) -> None:
    missing = [c for c in [args.smiles_col, args.pmic_col, args.label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Existing columns: {list(df.columns)}")


def make_scheme_a(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out[args.pmic_col] = pd.to_numeric(out[args.pmic_col], errors="coerce")
    out[args.label_col] = pd.to_numeric(out[args.label_col], errors="coerce").fillna(-1).astype(int)

    valid_label = out[args.label_col].isin([0, 1])
    valid_pmic = out[args.pmic_col].notna() & (out[args.pmic_col] > 0)
    active_exact = valid_label & (out[args.label_col] == 1) & valid_pmic & (out[args.pmic_col] < args.active_threshold)
    inactive = valid_label & (out[args.label_col] == 0)

    out["pmic_train"] = np.where(active_exact, out[args.pmic_col], np.nan)
    out["log10_pmic_train"] = np.where(active_exact, np.log10(out[args.pmic_col]), np.nan)
    out["pmic_regression_mask"] = active_exact.astype(int)

    censor_type = np.full(len(out), "invalid_label_or_pmic", dtype=object)
    censor_type[active_exact.to_numpy()] = "exact_active"
    censor_type[inactive.to_numpy()] = "inactive_over_threshold"
    out["mic_censor_type"] = censor_type

    train_weight = np.ones(len(out), dtype=float)
    generation_repeat = np.ones(len(out), dtype=int)

    active_values = out[args.pmic_col].to_numpy(dtype=float)
    very_low = active_exact & (out[args.pmic_col] <= args.very_low_mic)
    low = active_exact & (out[args.pmic_col] > args.very_low_mic) & (out[args.pmic_col] <= args.low_mic)
    mid = active_exact & (out[args.pmic_col] > args.low_mic) & (out[args.pmic_col] < args.active_threshold)

    train_weight[inactive.to_numpy()] = args.inactive_weight
    generation_repeat[inactive.to_numpy()] = max(1, int(args.inactive_repeat))

    train_weight[mid.to_numpy()] = float(args.mid_active_repeat)
    generation_repeat[mid.to_numpy()] = max(1, int(args.mid_active_repeat))

    train_weight[low.to_numpy()] = float(args.low_active_repeat)
    generation_repeat[low.to_numpy()] = max(1, int(args.low_active_repeat))

    train_weight[very_low.to_numpy()] = float(args.very_low_active_repeat)
    generation_repeat[very_low.to_numpy()] = max(1, int(args.very_low_active_repeat))

    invalid = ~valid_label | ~valid_pmic
    train_weight[invalid.to_numpy()] = 0.0
    generation_repeat[invalid.to_numpy()] = 0

    out["train_weight"] = train_weight
    out["generation_repeat"] = generation_repeat
    out["scheme_a_keep_for_activity_classifier"] = (valid_label & valid_pmic).astype(int)
    out["scheme_a_keep_for_mic_regression"] = out["pmic_regression_mask"]
    out["scheme_a_keep_for_generation"] = (generation_repeat > 0).astype(int)

    # Keep original column names and append explicit aliases for downstream scripts
    if args.smiles_col != "smiles":
        out["smiles"] = out[args.smiles_col]
    if args.pmic_col != "pmic_ug_ml":
        out["pmic_ug_ml"] = out[args.pmic_col]
    if args.label_col != "activity_label":
        out["activity_label"] = out[args.label_col]

    return out


def build_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    summary["rows"] = int(len(df))
    summary["activity_label_counts"] = {str(k): int(v) for k, v in df["activity_label"].value_counts(dropna=False).items()}
    summary["mic_censor_type_counts"] = {str(k): int(v) for k, v in df["mic_censor_type"].value_counts(dropna=False).items()}
    summary["pmic_regression_rows"] = int(df["pmic_regression_mask"].sum())
    summary["activity_classifier_rows"] = int(df["scheme_a_keep_for_activity_classifier"].sum())
    summary["generation_rows_before_repeat"] = int((df["generation_repeat"] > 0).sum())
    summary["generation_rows_after_repeat"] = int(df["generation_repeat"].sum())
    summary["pmic_ug_ml_min"] = float(df["pmic_ug_ml"].min())
    summary["pmic_ug_ml_max"] = float(df["pmic_ug_ml"].max())
    active_pmic = df.loc[df["pmic_regression_mask"] == 1, "pmic_train"]
    if len(active_pmic) > 0:
        summary["pmic_train_min"] = float(active_pmic.min())
        summary["pmic_train_median"] = float(active_pmic.median())
        summary["pmic_train_max"] = float(active_pmic.max())
    return summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    require_columns(df, args)
    out = make_scheme_a(df, args)

    out_csv = outdir / "processed_drug_mic_dataset_schemeA.csv"
    out_xlsx = outdir / "processed_drug_mic_dataset_schemeA.xlsx"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    try:
        out.to_excel(out_xlsx, index=False)
    except Exception as exc:
        print(f"Excel output skipped: {exc}")
        out_xlsx = None

    summary = build_summary(out)
    summary_path = outdir / "scheme_a_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    counts_path = outdir / "scheme_a_counts.csv"
    counts = []
    for col in ["activity_label", "mic_censor_type", "generation_repeat", "pmic_regression_mask"]:
        vc = out[col].value_counts(dropna=False).reset_index()
        vc.columns = ["value", "count"]
        vc.insert(0, "column", col)
        counts.append(vc)
    pd.concat(counts, ignore_index=True).to_csv(counts_path, index=False, encoding="utf-8-sig")

    print("Scheme A table prepared")
    print(f"Input: {input_path.resolve()}")
    print(f"Output CSV: {out_csv.resolve()}")
    if out_xlsx is not None:
        print(f"Output XLSX: {out_xlsx.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
