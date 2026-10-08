#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learn concentration-bin sample weights from generated molecules after sampling and scoring.

This script is intended for stage-wise self-training:
1 Train the first model without concentration weights.
2 Sample molecules and rank them with MIC activity QED alert scores.
3 Run this script on the ranked CSV to learn bin weights.
4 Rebuild processed .pt files with the learned weights and continue fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

BINS = ["inactive", "mic_0_1", "mic_1_10", "mic_10_100", "unknown"]
DEFAULT_WEIGHTS = {k: 1.0 for k in BINS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--generated_csv", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--previous_weights", type=str, default=None)
    parser.add_argument("--top_fraction", type=float, default=0.30)
    parser.add_argument("--min_top_rows", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.50, help="EMA update strength for new weights")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--min_weight", type=float, default=0.50)
    parser.add_argument("--max_weight", type=float, default=3.00)
    parser.add_argument("--inactive_weight", type=float, default=0.80)
    parser.add_argument("--score_col", type=str, default="total_score")
    parser.add_argument("--pmic_col", type=str, default="pred_pmic_ug_ml")
    parser.add_argument("--activity_prob_col", type=str, default="activity_label_probability")
    parser.add_argument("--alert_col", type=str, default="alert_count")
    parser.add_argument("--qed_col", type=str, default="qed")
    return parser.parse_args()


def load_previous(path: str | None) -> Dict[str, float]:
    if path is None:
        return dict(DEFAULT_WEIGHTS)
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    bins = obj.get("bins", obj)
    weights = dict(DEFAULT_WEIGHTS)
    for k in weights:
        v = bins.get(k, weights[k])
        if isinstance(v, dict):
            v = v.get("weight", weights[k])
        weights[k] = float(v)
    return weights


def mic_bin_from_pred(pmic: float, prob: float) -> str:
    if pd.isna(pmic) or pmic <= 0:
        return "unknown"
    if prob < 0.5 or pmic >= 100:
        return "inactive"
    if pmic <= 1:
        return "mic_0_1"
    if pmic <= 10:
        return "mic_1_10"
    return "mic_10_100"


def safe_col(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def robust_minmax_score(x: pd.Series, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = x.median(skipna=True)
    x = x.fillna(med if not pd.isna(med) else 0.0)
    lo, hi = x.quantile(0.05), x.quantile(0.95)
    if hi <= lo:
        return pd.Series([0.5] * len(x), index=x.index)
    y = ((x - lo) / (hi - lo)).clip(0.0, 1.0)
    if not higher_is_better:
        y = 1.0 - y
    return y


def compute_internal_score(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.score_col in df.columns:
        return robust_minmax_score(df[args.score_col], higher_is_better=True)
    pmic = safe_col(df, args.pmic_col, np.nan)
    prob = safe_col(df, args.activity_prob_col, 0.0).clip(0.0, 1.0)
    qed = safe_col(df, args.qed_col, 0.0).clip(0.0, 1.0)
    alert = safe_col(df, args.alert_col, 0.0)
    mic_score = robust_minmax_score(np.log10(pmic.clip(lower=1e-6)), higher_is_better=False)
    alert_score = robust_minmax_score(alert, higher_is_better=False)
    return 0.40 * mic_score + 0.35 * prob + 0.15 * qed + 0.10 * alert_score


def softmax_targets(utilities: Dict[str, float], args: argparse.Namespace) -> Dict[str, float]:
    active_bins = ["mic_0_1", "mic_1_10", "mic_10_100"]
    vals = np.array([utilities.get(k, np.nan) for k in active_bins], dtype=float)
    if np.all(np.isnan(vals)):
        return {"inactive": args.inactive_weight, "mic_0_1": 1.0, "mic_1_10": 1.0, "mic_10_100": 1.0, "unknown": 1.0}
    finite_mean = np.nanmean(vals)
    vals = np.where(np.isnan(vals), finite_mean, vals)
    vals = vals - vals.max()
    expv = np.exp(vals / max(args.temperature, 1e-6))
    expv = expv / max(expv.max(), 1e-8)
    targets = {}
    for k, e in zip(active_bins, expv):
        targets[k] = float(args.min_weight + (args.max_weight - args.min_weight) * e)
    targets["inactive"] = float(args.inactive_weight)
    targets["unknown"] = 1.0
    return targets


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.generated_csv)
    if df.empty:
        raise ValueError("generated_csv is empty")

    pmic = safe_col(df, args.pmic_col, np.nan)
    prob = safe_col(df, args.activity_prob_col, 0.0).clip(0.0, 1.0)
    df = df.copy()
    df["learned_bin"] = [mic_bin_from_pred(p, q) for p, q in zip(pmic, prob)]
    df["learned_internal_score"] = compute_internal_score(df, args)
    df = df.sort_values("learned_internal_score", ascending=False).reset_index(drop=True)

    n_top = max(args.min_top_rows, int(math.ceil(len(df) * args.top_fraction)))
    n_top = min(n_top, len(df))
    top = df.head(n_top).copy()

    utilities = {}
    counts = {}
    for b in BINS:
        sub = top[top["learned_bin"] == b]
        counts[b] = int(len(sub))
        utilities[b] = float(sub["learned_internal_score"].mean()) if len(sub) else float("nan")

    prev = load_previous(args.previous_weights)
    target = softmax_targets(utilities, args)
    new_weights = {}
    for b in BINS:
        new_weights[b] = float((1.0 - args.alpha) * prev.get(b, 1.0) + args.alpha * target.get(b, 1.0))
        new_weights[b] = float(np.clip(new_weights[b], args.min_weight, args.max_weight))

    out = {
        "version": "mrsa_adaptive_bin_weights_v1",
        "source_generated_csv": str(Path(args.generated_csv).resolve()),
        "method": "stagewise_generated_result_feedback",
        "top_fraction": args.top_fraction,
        "top_rows_used": n_top,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "note": "Use after an unweighted stage. Rebuild processed files with these weights and continue fine-tuning.",
        "bins": {
            b: {
                "weight": new_weights[b],
                "previous_weight": prev.get(b, 1.0),
                "target_weight": target.get(b, 1.0),
                "top_count": counts.get(b, 0),
                "top_mean_score": utilities.get(b, float("nan")),
            }
            for b in BINS
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_csv = out_path.with_suffix(".summary.csv")
    rows = []
    for b, info in out["bins"].items():
        r = {"bin": b}
        r.update(info)
        rows.append(r)
    pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print("Learned adaptive bin weights")
    print(f"Weights JSON: {out_path}")
    print(f"Summary CSV: {summary_csv}")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
