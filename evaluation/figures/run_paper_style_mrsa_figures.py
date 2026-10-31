#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


GROUPS = [
    "General-Model Generated",
    "MRSA-Trained Generated",
    "RCFR-Diffusion for MRSA",
]

ALL_GROUPS = [
    "Known MRSA Actives",
    "General-Model Generated",
    "MRSA-Trained Generated",
    "RCFR-Diffusion for MRSA",
]

COLORS = {
    "Known MRSA Actives": "#1f77b4",
    "General-Model Generated": "#7f7f7f",
    "MRSA-Trained Generated": "#ff7f0e",
    "RCFR-Diffusion for MRSA": "#2ca02c",
}

MARKERS = {
    "General-Model Generated": "o",
    "MRSA-Trained Generated": "^",
    "RCFR-Diffusion for MRSA": "s",
}

DESC_COLS = [
    "MW",
    "LogP",
    "TPSA",
    "QED",
    "SA",
    "HBA",
    "HBD",
    "RotB",
    "Rings",
    "AromaticRings",
    "HeavyAtoms",
]

DIST_COLS = [
    "MW",
    "LogP",
    "TPSA",
    "QED",
    "SA",
    "HBA",
    "HBD",
    "RotB",
    "Rings",
    "log10_MIC",
]

PLOT_PROPERTIES = [
    "MW",
    "LogP",
    "TPSA",
    "QED",
    "SA",
    "log10_MIC",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig, outprefix: Path) -> None:
    ensure_dir(outprefix.parent)

    with open(outprefix.with_suffix(".mplfig.pkl"), "wb") as f:
        pickle.dump(fig, f)

    fig.savefig(outprefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outprefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outprefix.with_suffix(".pdf"), bbox_inches="tight")


def safe_ratio(a: float, b: float) -> float:
    if b == 0 or pd.isna(b):
        return float("nan")
    return float(a) / float(b)


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
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


def build_filter_catalog():
    try:
        from rdkit.Chem import FilterCatalog
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
        try:
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        except Exception:
            pass
        return FilterCatalog.FilterCatalog(params)
    except Exception:
        return None


FILTER_CATALOG = build_filter_catalog()


def alert_count(mol: Chem.Mol) -> float:
    if FILTER_CATALOG is None:
        return float("nan")
    try:
        return float(len(FILTER_CATALOG.GetMatches(mol)))
    except Exception:
        return float("nan")


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
                return carbon, heavy, frag.GetNumAtoms()
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
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
        except Exception:
            return None

    try:
        smi = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)
        return Chem.MolFromSmiles(smi)
    except Exception:
        return mol


def canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)
    except Exception:
        return None


def get_sa(mol: Chem.Mol) -> float:
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
        "SA": get_sa(mol),
        "HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "RotB": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "Rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "AromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "HeavyAtoms": float(mol.GetNumHeavyAtoms()),
        "AlertCount": alert_count(mol),
    }


def get_numeric(row: pd.Series, candidates: List[str]) -> float:
    for c in candidates:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v):
                return float(v)
    return float("nan")


def get_log10_mic(row: pd.Series, is_reference: bool) -> float:
    if is_reference:
        candidates = [
            "pmic_ug_ml",
            "standard_concentration_ug_ml",
            "standard_concentration",
            "mic_ug_ml",
            "MIC_ug_ml",
            "concentration_ug_ml",
        ]
    else:
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
        if c not in row.index:
            continue
        v = pd.to_numeric(row[c], errors="coerce")
        if pd.notna(v) and float(v) > 0:
            return float(np.log10(float(v)))

    return float("nan")


def load_molecule_table(
    path: Path,
    group: str,
    top_n: int,
    is_reference: bool,
    reference_active_only: bool,
    activity_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])

    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {path}")

    if is_reference and reference_active_only and activity_col in df.columns:
        df = df[df[activity_col] == 1].copy()

    if not is_reference and top_n > 0:
        df = df.head(top_n).copy()

    rows = []
    seen = set()

    for idx, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])

        if mol is None:
            continue

        smi = canonical_smiles(mol)

        if smi is None:
            continue

        if is_reference and smi in seen:
            continue

        seen.add(smi)

        out = {
            "Group": group,
            "source_index": int(idx),
            "smiles": smi,
            "mol": mol,
            "log10_MIC": get_log10_mic(row, is_reference=is_reference),
            "activity_label_probability": get_numeric(row, ["activity_label_probability", "activity_probability", "activity_prob"]),
            "pred_pmic_ug_ml": get_numeric(row, ["pred_pmic_ug_ml", "pred_mic_ug_ml", "pred_MIC_ug_ml"]),
            "pred_log10_pmic": get_numeric(row, ["pred_log10_pmic"]),
            "mic_score": get_numeric(row, ["mic_score"]),
            "total_score": get_numeric(row, ["total_score"]),
            "lipinski_score": get_numeric(row, ["lipinski_score"]),
        }

        out.update(calc_desc(mol))
        rows.append(out)

    out_df = pd.DataFrame(rows)

    if out_df.empty:
        raise RuntimeError(f"No valid molecules loaded from {path}")

    return out_df.reset_index(drop=True)


def load_all(args):
    reference = load_molecule_table(
        Path(args.reference_csv),
        "Known MRSA Actives",
        top_n=0,
        is_reference=True,
        reference_active_only=args.reference_active_only,
        activity_col=args.activity_col,
    )

    ranked = {
        "General-Model Generated": load_molecule_table(Path(args.original_ranked_csv), "General-Model Generated", args.top_n, False, False, args.activity_col),
        "MRSA-Trained Generated": load_molecule_table(Path(args.scratch_ranked_csv), "MRSA-Trained Generated", args.top_n, False, False, args.activity_col),
        "RCFR-Diffusion for MRSA": load_molecule_table(Path(args.rcfr_ranked_csv), "RCFR-Diffusion for MRSA", args.top_n, False, False, args.activity_col),
    }

    all_paths = {
        "General-Model Generated": Path(args.original_all_scored_csv),
        "MRSA-Trained Generated": Path(args.scratch_all_scored_csv),
        "RCFR-Diffusion for MRSA": Path(args.rcfr_all_scored_csv),
    }

    all_scored = {}
    for group, path in all_paths.items():
        if path.exists():
            all_scored[group] = load_molecule_table(path, group, 0, False, False, args.activity_col)
        else:
            all_scored[group] = ranked[group].copy()

    summary_jsons = {
        "General-Model Generated": read_json(Path(args.original_summary_json)),
        "MRSA-Trained Generated": read_json(Path(args.scratch_summary_json)),
        "RCFR-Diffusion for MRSA": read_json(Path(args.rcfr_summary_json)),
    }

    return reference, ranked, all_scored, summary_jsons


def mol_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def fp_to_array(fp, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def sparse_bits(mol: Chem.Mol, radius: int = 2) -> set:
    try:
        fp = AllChem.GetMorganFingerprint(mol, radius=radius)
        return set(fp.GetNonzeroElements().keys())
    except Exception:
        return set()


def scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return "NO_SCAFFOLD"
        smi = Chem.MolToSmiles(scaf, canonical=True)
        return smi if smi else "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


def average_pairwise_diversity(smiles: List[str], max_pairs: int, seed: int) -> float:
    mols = []
    for smi in sorted(set(smiles)):
        mol = mol_from_smiles(smi)
        if mol is not None:
            mols.append(mol)

    if len(mols) < 2:
        return float("nan")

    fps = [mol_fp(m) for m in mols]
    n = len(fps)
    rng = np.random.default_rng(seed)
    total = n * (n - 1) // 2

    if total <= max_pairs:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = []
        for _ in range(max_pairs):
            i, j = rng.choice(n, size=2, replace=False)
            if i > j:
                i, j = j, i
            pairs.append((int(i), int(j)))

    dists = []
    for i, j in pairs:
        dists.append(1.0 - float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))

    return float(np.mean(dists))


def druglike_pass(df: pd.DataFrame, args) -> pd.Series:
    qed = pd.to_numeric(df["QED"], errors="coerce")
    sa = pd.to_numeric(df["SA"], errors="coerce")
    mw = pd.to_numeric(df["MW"], errors="coerce")
    logp = pd.to_numeric(df["LogP"], errors="coerce")
    tpsa = pd.to_numeric(df["TPSA"], errors="coerce")
    hba = pd.to_numeric(df["HBA"], errors="coerce")
    hbd = pd.to_numeric(df["HBD"], errors="coerce")
    alerts = pd.to_numeric(df["AlertCount"], errors="coerce").fillna(0)

    return (
        (qed >= args.qed_min)
        & (sa <= args.sa_max)
        & (mw <= args.mw_max)
        & (logp <= args.logp_max)
        & (tpsa <= args.tpsa_max)
        & (hba <= args.hba_max)
        & (hbd <= args.hbd_max)
        & (alerts <= args.alert_max)
    )


def quantile_distance(ref_values, gen_values) -> float:
    ref = np.asarray(ref_values, dtype=float)
    gen = np.asarray(gen_values, dtype=float)
    ref = ref[np.isfinite(ref)]
    gen = gen[np.isfinite(gen)]

    if len(ref) == 0 or len(gen) == 0:
        return float("nan")

    qs = np.linspace(0, 1, 101)
    rq = np.quantile(ref, qs)
    gq = np.quantile(gen, qs)

    scale = np.percentile(ref, 75) - np.percentile(ref, 25)
    if scale <= 1e-8:
        scale = np.std(ref)
    if scale <= 1e-8:
        scale = 1.0

    return float(np.mean(np.abs(rq - gq)) / scale)


def wasserstein_distance_safe(ref_values, gen_values) -> float:
    try:
        from scipy.stats import wasserstein_distance
    except Exception:
        return float("nan")

    ref = np.asarray(ref_values, dtype=float)
    gen = np.asarray(gen_values, dtype=float)
    ref = ref[np.isfinite(ref)]
    gen = gen[np.isfinite(gen)]

    if len(ref) == 0 or len(gen) == 0:
        return float("nan")

    scale = np.percentile(ref, 75) - np.percentile(ref, 25)
    if scale <= 1e-8:
        scale = np.std(ref)
    if scale <= 1e-8:
        scale = 1.0

    return float(wasserstein_distance(ref, gen) / scale)


def ks_stat_safe(ref_values, gen_values) -> float:
    try:
        from scipy.stats import ks_2samp
    except Exception:
        return float("nan")

    ref = np.asarray(ref_values, dtype=float)
    gen = np.asarray(gen_values, dtype=float)
    ref = ref[np.isfinite(ref)]
    gen = gen[np.isfinite(gen)]

    if len(ref) == 0 or len(gen) == 0:
        return float("nan")

    return float(ks_2samp(ref, gen).statistic)


def frechet_distance_safe(X: np.ndarray, Y: np.ndarray) -> float:
    try:
        from scipy.linalg import sqrtm
    except Exception:
        return float("nan")

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)]
    Y = Y[np.all(np.isfinite(Y), axis=1)]

    if len(X) < 2 or len(Y) < 2:
        return float("nan")

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)

    cov_prod = sqrtm(cov_x.dot(cov_y))

    if np.iscomplexobj(cov_prod):
        cov_prod = cov_prod.real

    diff = mu_x - mu_y
    val = diff.dot(diff) + np.trace(cov_x + cov_y - 2.0 * cov_prod)
    return float(np.real(val))


def density_curve(ax, values, label, color, linewidth=2.0):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values.astype(float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return

    try:
        from scipy.stats import gaussian_kde
        low, high = np.percentile(vals, [1, 99])
        if high <= low:
            low, high = vals.min(), vals.max()
        if high <= low:
            return
        xs = np.linspace(low, high, 250)
        kde = gaussian_kde(vals)
        ys = kde(xs)
        ax.plot(xs, ys, label=label, color=color, linewidth=linewidth)
    except Exception:
        ax.hist(vals, bins=35, density=True, histtype="step", label=label, color=color, linewidth=linewidth)


def build_table_1(reference, ranked, all_scored, summary_jsons, args, outdir):
    ref_set = set(reference["smiles"])
    rows = []

    for group in GROUPS:
        all_df = all_scored[group]
        top_df = ranked[group]
        unique_smiles = sorted(set(all_df["smiles"]))
        novel = sum(1 for s in unique_smiles if s not in ref_set)

        summary = summary_jsons.get(group, {})
        sampled = int(summary.get("generated_objects", args.sampled_count))
        valid = int(summary.get("valid_raw_molecules", len(all_df)))

        rows.append({
            "Method": group,
            "Sampled": sampled,
            "Valid": valid,
            "Validity ↑": safe_ratio(valid, sampled),
            "Unique ↑": safe_ratio(len(unique_smiles), len(all_df)),
            "Novel ↑": safe_ratio(novel, len(unique_smiles)),
            "Diversity ↑": average_pairwise_diversity(unique_smiles, args.max_pairs, args.seed),
            "QED ↑": float(top_df["QED"].mean()),
            "SA ↓": float(top_df["SA"].mean()),
            "MW": float(top_df["MW"].mean()),
            "LogP": float(top_df["LogP"].mean()),
            "TPSA": float(top_df["TPSA"].mean()),
            "Lipinski ↑": float(pd.to_numeric(top_df["lipinski_score"], errors="coerce").mean()) if "lipinski_score" in top_df.columns else float("nan"),
            "Drug-like ↑": float(druglike_pass(top_df, args).mean()),
            "Alert-free ↑": float((pd.to_numeric(top_df["AlertCount"], errors="coerce").fillna(0) <= args.alert_max).mean()),
        })

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "table_1_generation_druglikeness_benchmark.csv", index=False, encoding="utf-8-sig")
    return table


def build_table_2(ranked, args, outdir):
    rows = []

    for group in GROUPS:
        df = ranked[group]
        prob = pd.to_numeric(df["activity_label_probability"], errors="coerce")
        mic = pd.to_numeric(df["pred_pmic_ug_ml"], errors="coerce")
        top50 = df.head(min(50, len(df)))
        prob50 = pd.to_numeric(top50["activity_label_probability"], errors="coerce")
        mic50 = pd.to_numeric(top50["pred_pmic_ug_ml"], errors="coerce")

        rows.append({
            "Method": group,
            "Activity prob. ↑": float(prob.mean()),
            "Active ratio ↑": float((prob >= args.activity_threshold).mean()),
            "Median MIC ↓": float(mic.median()),
            "MIC score ↑": float(pd.to_numeric(df["mic_score"], errors="coerce").mean()),
            "Total score ↑": float(pd.to_numeric(df["total_score"], errors="coerce").mean()),
            "Top-50 active ↑": float((prob50 >= args.activity_threshold).mean()),
            "Top-50 MIC ↓": float(mic50.median()),
            "Top-50 QED ↑": float(pd.to_numeric(top50["QED"], errors="coerce").mean()),
            "Top-50 SA ↓": float(pd.to_numeric(top50["SA"], errors="coerce").mean()),
        })

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "table_2_mrsa_goal_directed_candidate_quality.csv", index=False, encoding="utf-8-sig")
    return table


def compute_property_distances(reference, ranked, outdir):
    detail_rows = []

    ref_X = reference[DESC_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    ref_mean = np.nanmean(ref_X, axis=0)
    ref_std = np.nanstd(ref_X, axis=0)
    ref_std[ref_std <= 1e-8] = 1.0
    ref_Z = (ref_X - ref_mean) / ref_std

    fd_rows = []

    for group in GROUPS:
        df = ranked[group]

        for col in DIST_COLS:
            ref_vals = pd.to_numeric(reference[col], errors="coerce").dropna().values.astype(float)
            gen_vals = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(float)

            detail_rows.append({
                "Method": group,
                "Descriptor": col,
                "Quantile distance ↓": quantile_distance(ref_vals, gen_vals),
                "Wasserstein distance ↓": wasserstein_distance_safe(ref_vals, gen_vals),
                "KS statistic ↓": ks_stat_safe(ref_vals, gen_vals),
            })

        gen_X = df[DESC_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
        gen_Z = (gen_X - ref_mean) / ref_std
        fd_rows.append({
            "Method": group,
            "Descriptor Frechet distance ↓": frechet_distance_safe(ref_Z, gen_Z),
        })

    detail = pd.DataFrame(detail_rows)
    fd = pd.DataFrame(fd_rows)

    summary = detail.groupby("Method").agg({
        "Quantile distance ↓": "mean",
        "Wasserstein distance ↓": "mean",
        "KS statistic ↓": "mean",
    }).reset_index().rename(columns={
        "Quantile distance ↓": "Mean property distance ↓",
        "Wasserstein distance ↓": "Mean Wasserstein distance ↓",
        "KS statistic ↓": "Mean KS statistic ↓",
    })

    summary = summary.merge(fd, on="Method", how="left")

    detail.to_csv(outdir / "property_distance_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "property_distance_summary.csv", index=False, encoding="utf-8-sig")

    return detail, summary


def compute_tanimoto(reference, ranked, args, outdir):
    ref_fps = [mol_fp(m) for m in reference["mol"]]
    detail_rows = []
    summary_rows = []

    for group in GROUPS:
        sims = []

        for _, row in ranked[group].iterrows():
            fp = mol_fp(row["mol"])
            vals = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)

            if vals:
                idx = int(np.argmax(vals))
                max_sim = float(vals[idx])
                nearest = reference.iloc[idx]["smiles"]
            else:
                max_sim = float("nan")
                nearest = ""

            sims.append(max_sim)
            detail_rows.append({
                "Method": group,
                "smiles": row["smiles"],
                "nearest_reference_smiles": nearest,
                "Max Tanimoto": max_sim,
                "Activity probability": row.get("activity_label_probability", np.nan),
                "Predicted MIC": row.get("pred_pmic_ug_ml", np.nan),
                "QED": row.get("QED", np.nan),
                "SA": row.get("SA", np.nan),
                "MW": row.get("MW", np.nan),
                "LogP": row.get("LogP", np.nan),
                "TPSA": row.get("TPSA", np.nan),
                "Total score": row.get("total_score", np.nan),
            })

        arr = np.asarray(sims, dtype=float)
        arr = arr[np.isfinite(arr)]

        useful = (arr >= args.useful_tanimoto_low) & (arr < args.useful_tanimoto_high)

        summary_rows.append({
            "Method": group,
            "Mean max Tanimoto ↑": float(np.mean(arr)) if len(arr) else np.nan,
            "Median max Tanimoto ↑": float(np.median(arr)) if len(arr) else np.nan,
            "Useful analogue ratio ↑": float(np.mean(useful)) if len(arr) else np.nan,
            "Tanimoto >= 0.3 ↑": float(np.mean(arr >= 0.3)) if len(arr) else np.nan,
            "Tanimoto >= 0.4 ↑": float(np.mean(arr >= 0.4)) if len(arr) else np.nan,
            "Tanimoto >= 0.5 ↑": float(np.mean(arr >= 0.5)) if len(arr) else np.nan,
        })

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)

    detail.to_csv(outdir / "tanimoto_nearest_neighbor_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "tanimoto_nearest_neighbor_summary.csv", index=False, encoding="utf-8-sig")

    return detail, summary


def compute_scaffold_fragment(reference, ranked, args, outdir):
    ref_scaffolds = set(scaffold_smiles(m) for m in reference["mol"])
    ref_scaffolds.discard("NO_SCAFFOLD")

    ref_scaf_counter = Counter(scaffold_smiles(m) for m in reference["mol"])
    ref_scaf_counter.pop("NO_SCAFFOLD", None)
    top_ref_scaffolds = [s for s, _ in ref_scaf_counter.most_common(args.top_scaffolds)]

    ref_bit_counter = Counter()
    for mol in reference["mol"]:
        ref_bit_counter.update(sparse_bits(mol))

    top_bits = set(bit for bit, _ in ref_bit_counter.most_common(args.top_fragments))

    rows = []
    freq_rows = []

    for group in GROUPS:
        df = ranked[group]
        scafs = []
        group_bits = set()
        mol_scaf_hits = []
        mol_frag_cov = []

        for _, row in df.iterrows():
            mol = row["mol"]
            scaf = scaffold_smiles(mol)
            bits = sparse_bits(mol)

            scafs.append(scaf)
            group_bits.update(bits)
            mol_scaf_hits.append(int(scaf in ref_scaffolds))
            mol_frag_cov.append(safe_ratio(len(bits & top_bits), len(top_bits)))

        scaf_set = set(scafs)
        scaf_set.discard("NO_SCAFFOLD")
        overlap = scaf_set & ref_scaffolds

        rows.append({
            "Method": group,
            "Generated scaffolds": len(scaf_set),
            "Overlapped scaffolds": len(overlap),
            "Scaffold overlap ↑": safe_ratio(len(overlap), len(scaf_set)),
            "Molecule scaffold coverage ↑": float(np.mean(mol_scaf_hits)) if mol_scaf_hits else np.nan,
            "Fragment group coverage ↑": safe_ratio(len(group_bits & top_bits), len(top_bits)),
            "Mean fragment coverage ↑": float(np.mean(mol_frag_cov)) if mol_frag_cov else np.nan,
        })

        scaf_counter = Counter(scafs)
        n = max(1, len(scafs))

        for rank, scaf in enumerate(top_ref_scaffolds, start=1):
            freq_rows.append({
                "Method": group,
                "Reference scaffold rank": rank,
                "Scaffold": scaf,
                "Frequency": scaf_counter.get(scaf, 0) / n,
            })

    summary = pd.DataFrame(rows)
    freq = pd.DataFrame(freq_rows)

    summary.to_csv(outdir / "scaffold_fragment_summary.csv", index=False, encoding="utf-8-sig")
    freq.to_csv(outdir / "top_scaffold_frequency_lollipop_data.csv", index=False, encoding="utf-8-sig")

    return summary, freq


def chemical_features(reference, ranked, args):
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError("scikit-learn is required for chemical space analysis") from e

    ref = reference.copy()

    if args.max_reference_plot > 0 and len(ref) > args.max_reference_plot:
        ref = ref.sample(n=args.max_reference_plot, random_state=args.seed).copy()

    records = []

    for _, row in ref.iterrows():
        fp = fp_to_array(mol_fp(row["mol"]))
        desc = row[["MW", "LogP", "TPSA", "QED", "SA", "HBA", "HBD", "RotB", "Rings"]].astype(float).values
        records.append(("Known MRSA Actives", row["smiles"], fp, desc))

    for group in GROUPS:
        for _, row in ranked[group].iterrows():
            fp = fp_to_array(mol_fp(row["mol"]))
            desc = row[["MW", "LogP", "TPSA", "QED", "SA", "HBA", "HBD", "RotB", "Rings"]].astype(float).values
            records.append((group, row["smiles"], fp, desc))

    fps = np.vstack([r[2] for r in records])
    descs = np.vstack([r[3] for r in records])
    descs = np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)
    descs = StandardScaler().fit_transform(descs)

    X = np.hstack([fps, descs])
    meta = pd.DataFrame({
        "Method": [r[0] for r in records],
        "smiles": [r[1] for r in records],
    })

    return X, meta


def chemical_space(reference, ranked, args, outdir):
    X, meta = chemical_features(reference, ranked, args)

    method = "UMAP"

    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.seed,
        )
        coords = reducer.fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA
        method = "PCA"
        coords = PCA(n_components=2, random_state=args.seed).fit_transform(X)

    coord = meta.copy()
    coord["X1"] = coords[:, 0]
    coord["X2"] = coords[:, 1]
    coord.to_csv(outdir / f"chemical_space_{method.lower()}_coordinates.csv", index=False, encoding="utf-8-sig")

    ref_pts = coord[coord["Method"] == "Known MRSA Actives"][["X1", "X2"]].values.astype(float)

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(ref_pts.T)
        ref_density = kde(ref_pts.T)
        density_threshold = np.quantile(ref_density, args.high_density_quantile)
    except Exception:
        kde = None
        density_threshold = None

    rows = []

    ref_centroid = ref_pts.mean(axis=0)

    for group in GROUPS:
        pts = coord[coord["Method"] == group][["X1", "X2"]].values.astype(float)

        nn = []
        for p in pts:
            nn.append(float(np.min(np.linalg.norm(ref_pts - p, axis=1))))

        centroid_distance = float(np.linalg.norm(pts.mean(axis=0) - ref_centroid))

        if kde is not None and density_threshold is not None:
            dens = kde(pts.T)
            hit_ratio = float(np.mean(dens >= density_threshold))
        else:
            hit_ratio = float("nan")

        rows.append({
            "Method": group,
            f"{method} mean nearest distance ↓": float(np.mean(nn)),
            f"{method} median nearest distance ↓": float(np.median(nn)),
            f"{method} centroid distance ↓": centroid_distance,
            f"{method} MRSA high-density hit ratio ↑": hit_ratio,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / f"chemical_space_{method.lower()}_summary.csv", index=False, encoding="utf-8-sig")

    return coord, summary, method


def build_table_3(property_summary, tanimoto_summary, chem_summary, chem_method, scaffold_summary, outdir):
    table = property_summary.copy()
    table = table.merge(tanimoto_summary, on="Method", how="left")
    table = table.merge(chem_summary, on="Method", how="left")
    table = table.merge(scaffold_summary, on="Method", how="left")

    keep = [
        "Method",
        "Mean property distance ↓",
        "Descriptor Frechet distance ↓",
        "Mean max Tanimoto ↑",
        "Useful analogue ratio ↑",
        f"{chem_method} mean nearest distance ↓",
        f"{chem_method} MRSA high-density hit ratio ↑",
        "Molecule scaffold coverage ↑",
        "Fragment group coverage ↑",
    ]

    keep = [c for c in keep if c in table.columns]
    table = table[keep]
    table.to_csv(outdir / "table_3_mrsa_space_proximity_benchmark.csv", index=False, encoding="utf-8-sig")
    return table


def plot_figure_1(table1, table2, ranked, args, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))
    axes = axes.flatten()

    key_metrics = ["Validity ↑", "Diversity ↑", "QED ↑", "SA ↓"]
    x = np.arange(len(GROUPS))
    width = 0.18

    for i, metric in enumerate(key_metrics):
        vals = table1.set_index("Method").loc[GROUPS, metric].values.astype(float)
        if metric == "SA ↓":
            vals_norm = 1.0 / vals
            label = "1 / SA"
        else:
            vals_norm = vals
            label = metric
        axes[0].bar(x + (i - 1.5) * width, vals_norm, width=width, label=label)

    axes[0].set_title("A. Key generation and drug-likeness metrics")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(GROUPS, rotation=20, ha="right")
    axes[0].set_ylabel("Metric value")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(fontsize=8)

    for group in GROUPS:
        df = ranked[group]
        prob = pd.to_numeric(df["activity_label_probability"], errors="coerce")
        mic = pd.to_numeric(df["pred_pmic_ug_ml"], errors="coerce")
        y = np.log10(mic.where(mic > 0))
        axes[1].scatter(prob, y, s=28, alpha=0.65, label=group, color=COLORS[group], marker=MARKERS[group])

    axes[1].set_title("B. Activity-MIC objective landscape")
    axes[1].set_xlabel("Activity probability")
    axes[1].set_ylabel("log10 predicted MIC")
    axes[1].grid(True, linestyle="--", alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].text(0.72, axes[1].get_ylim()[0] + 0.05, "Ideal region\nhigh probability\nlow MIC", fontsize=8)

    ks = [20, 50, 100, 200]
    top_rows = []
    for group in GROUPS:
        for k in ks:
            sub = ranked[group].head(min(k, len(ranked[group])))
            mic = pd.to_numeric(sub["pred_pmic_ug_ml"], errors="coerce")
            prob = pd.to_numeric(sub["activity_label_probability"], errors="coerce")
            top_rows.append({
                "Method": group,
                "TopK": k,
                "Median MIC": float(mic.median()),
                "Active ratio": float((prob >= args.activity_threshold).mean()),
            })

    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(outdir / "figure_1_topk_curve_data.csv", index=False, encoding="utf-8-sig")

    for group in GROUPS:
        sub = top_df[top_df["Method"] == group]
        axes[2].plot(sub["TopK"], sub["Median MIC"], marker="o", linewidth=2, color=COLORS[group], label=group)

    axes[2].set_title("C. Top-K MIC enrichment")
    axes[2].set_xlabel("Top-K candidates")
    axes[2].set_ylabel("Median predicted MIC")
    axes[2].grid(True, linestyle="--", alpha=0.25)
    axes[2].legend(fontsize=8)

    for group in GROUPS:
        df = ranked[group]
        qed = pd.to_numeric(df["QED"], errors="coerce")
        mic = pd.to_numeric(df["pred_pmic_ug_ml"], errors="coerce")
        sa = pd.to_numeric(df["SA"], errors="coerce")
        y = -np.log10(mic.where(mic > 0))
        size = np.clip(80 - 8 * sa, 20, 80)
        axes[3].scatter(qed, y, s=size, alpha=0.6, label=group, color=COLORS[group], marker=MARKERS[group])

    axes[3].set_title("D. Pareto view: QED vs predicted potency")
    axes[3].set_xlabel("QED")
    axes[3].set_ylabel("-log10 predicted MIC")
    axes[3].grid(True, linestyle="--", alpha=0.25)
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, outdir / "figure_1_multi_objective_generation_performance")
    plt.close(fig)


def plot_figure_2(reference, ranked, prop_detail, prop_summary, outdir):
    fig = plt.figure(figsize=(14.5, 12.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.05])

    all_tables = {"Known MRSA Actives": reference, **ranked}

    for idx, prop in enumerate(PLOT_PROPERTIES):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        for group in ALL_GROUPS:
            df = all_tables[group]
            vals = pd.to_numeric(df[prop], errors="coerce")
            lw = 2.6 if group == "Known MRSA Actives" else 1.9
            density_curve(ax, vals, group, COLORS[group], linewidth=lw)

        ax.set_title(prop)
        ax.set_xlabel(prop)
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.25)

        if idx == 0:
            ax.legend(fontsize=7)

    ax_hm = fig.add_subplot(gs[2, :2])
    pivot = prop_detail.pivot(index="Descriptor", columns="Method", values="Quantile distance ↓")
    pivot = pivot.reindex(index=DIST_COLS, columns=GROUPS)
    data = pivot.to_numpy(dtype=float)

    im = ax_hm.imshow(data, aspect="auto")
    ax_hm.set_title("B. Descriptor distribution distance to MRSA dataset")
    ax_hm.set_xticks(np.arange(len(GROUPS)))
    ax_hm.set_xticklabels(GROUPS, rotation=25, ha="right")
    ax_hm.set_yticks(np.arange(len(DIST_COLS)))
    ax_hm.set_yticklabels(DIST_COLS)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.02)
    cbar.set_label("Quantile distance, lower is better")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax_hm.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)

    ax_bar = fig.add_subplot(gs[2, 2])
    vals = prop_summary.set_index("Method").loc[GROUPS, "Mean property distance ↓"]
    ax_bar.bar(vals.index, vals.values, color=[COLORS[g] for g in GROUPS])
    ax_bar.set_title("C. Mean property distance")
    ax_bar.set_ylabel("Distance, lower is better")
    ax_bar.tick_params(axis="x", rotation=25)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.25)

    for i, v in enumerate(vals.values):
        if np.isfinite(v):
            ax_bar.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle("Figure 2. Distribution matching to original MRSA data", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, outdir / "figure_2_distribution_matching_to_mrsa")
    plt.close(fig)


def plot_density_background(ax, ref_pts, method_name):
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(ref_pts.T)

        x_min, y_min = ref_pts.min(axis=0)
        x_max, y_max = ref_pts.max(axis=0)
        padx = (x_max - x_min) * 0.08
        pady = (y_max - y_min) * 0.08

        xs = np.linspace(x_min - padx, x_max + padx, 120)
        ys = np.linspace(y_min - pady, y_max + pady, 120)
        X, Y = np.meshgrid(xs, ys)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(positions).T, X.shape)

        ax.contourf(X, Y, Z, levels=15, alpha=0.45)
    except Exception:
        ax.scatter(ref_pts[:, 0], ref_pts[:, 1], s=8, alpha=0.15, color=COLORS["Known MRSA Actives"])

    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")


def plot_figure_3(coord, chem_summary, chem_method, outdir):
    fig = plt.figure(figsize=(14.5, 9.8))
    gs = fig.add_gridspec(2, 3)

    ref = coord[coord["Method"] == "Known MRSA Actives"][["X1", "X2"]].values.astype(float)

    ax = fig.add_subplot(gs[0, 0])
    plot_density_background(ax, ref, chem_method)
    ax.scatter(ref[:, 0], ref[:, 1], s=8, alpha=0.25, color=COLORS["Known MRSA Actives"])
    ax.set_title("A. Known MRSA Actives density")
    ax.grid(True, linestyle="--", alpha=0.20)

    for k, group in enumerate(GROUPS, start=1):
        ax = fig.add_subplot(gs[0 if k < 3 else 1, k if k < 3 else 0])
        plot_density_background(ax, ref, chem_method)
        sub = coord[coord["Method"] == group]
        ax.scatter(sub["X1"], sub["X2"], s=28, alpha=0.78, color=COLORS[group], marker=MARKERS[group])
        ax.set_title(f"{chr(65+k)}. {group}")
        ax.grid(True, linestyle="--", alpha=0.20)

    ax = fig.add_subplot(gs[1, 1:])
    hit_col = f"{chem_method} MRSA high-density hit ratio ↑"
    dist_col = f"{chem_method} mean nearest distance ↓"

    x = np.arange(len(GROUPS))
    width = 0.35

    hit = chem_summary.set_index("Method").loc[GROUPS, hit_col]
    dist = chem_summary.set_index("Method").loc[GROUPS, dist_col]

    ax.bar(x - width / 2, hit.values, width=width, label="High-density hit ratio", color="#8ecae6")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, dist.values, width=width, label="Mean nearest distance", color="#adb5bd")

    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=20, ha="right")
    ax.set_ylabel("Hit ratio, higher is better")
    ax2.set_ylabel("Distance, lower is better")
    ax.set_title("E. Chemical-space proximity statistics")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    fig.suptitle("Figure 3. Chemical-space proximity to MRSA molecules", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, outdir / "figure_3_chemical_space_proximity")
    plt.close(fig)


def plot_figure_4(tanimoto_detail, scaffold_summary, scaffold_freq, args, outdir):
    fig = plt.figure(figsize=(15.2, 13.0))
    gs = fig.add_gridspec(3, 2)

    ax = fig.add_subplot(gs[0, 0])
    thresholds = np.linspace(0, 1, 101)

    for group in GROUPS:
        vals = tanimoto_detail[tanimoto_detail["Method"] == group]["Max Tanimoto"].dropna().values.astype(float)
        survival = [(vals >= t).mean() for t in thresholds]
        ax.plot(thresholds, survival, linewidth=2.2, color=COLORS[group], label=group)

    ax.set_title("A. Nearest-neighbor Tanimoto ECDF")
    ax.set_xlabel("Tanimoto threshold")
    ax.set_ylabel("Ratio above threshold")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for group in GROUPS:
        sub = tanimoto_detail[tanimoto_detail["Method"] == group]
        ax.scatter(
            sub["Max Tanimoto"],
            sub["Activity probability"],
            s=28,
            alpha=0.65,
            color=COLORS[group],
            marker=MARKERS[group],
            label=group,
        )

    ax.axvspan(args.useful_tanimoto_low, args.useful_tanimoto_high, alpha=0.12, color="gray")
    ax.text(args.useful_tanimoto_low + 0.02, 0.05, "Useful analogue region", fontsize=8)
    ax.set_title("B. Similarity-activity relationship")
    ax.set_xlabel("Max Tanimoto to MRSA")
    ax.set_ylabel("Activity probability")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    bins = [
        ("<0.3", -0.001, 0.3),
        ("0.3-0.6", 0.3, 0.6),
        (">=0.6", 0.6, 1.001),
    ]

    bottom = np.zeros(len(GROUPS))
    x = np.arange(len(GROUPS))

    bin_records = []
    for label, low, high in bins:
        vals = []
        for group in GROUPS:
            arr = tanimoto_detail[tanimoto_detail["Method"] == group]["Max Tanimoto"].dropna().values.astype(float)
            ratio = float(((arr >= low) & (arr < high)).mean()) if len(arr) else np.nan
            vals.append(ratio)
            bin_records.append({"Method": group, "Tanimoto bin": label, "Ratio": ratio})
        ax.bar(x, vals, bottom=bottom, label=label)
        bottom += np.nan_to_num(vals)

    pd.DataFrame(bin_records).to_csv(outdir / "figure_4_tanimoto_bin_composition.csv", index=False, encoding="utf-8-sig")

    ax.set_title("C. Tanimoto bin composition")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=20, ha="right")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    metrics = ["Molecule scaffold coverage ↑", "Fragment group coverage ↑"]
    width = 0.35

    for i, metric in enumerate(metrics):
        vals = scaffold_summary.set_index("Method").loc[GROUPS, metric].values.astype(float)
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=metric)

    ax.set_title("D. Scaffold and fragment transfer")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=20, ha="right")
    ax.set_ylabel("Coverage ratio")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, :])
    top_freq = scaffold_freq.copy()
    top_n = min(10, top_freq["Reference scaffold rank"].max())
    top_freq = top_freq[top_freq["Reference scaffold rank"] <= top_n].copy()

    y_positions = np.arange(top_n)
    offsets = {
        "General-Model Generated": -0.22,
        "MRSA-Trained Generated": 0.0,
        "RCFR-Diffusion for MRSA": 0.22,
    }

    for group in GROUPS:
        sub = top_freq[top_freq["Method"] == group].sort_values("Reference scaffold rank")
        y = np.array(sub["Reference scaffold rank"].values, dtype=float) - 1 + offsets[group]
        xvals = sub["Frequency"].values.astype(float)
        ax.hlines(y, 0, xvals, color=COLORS[group], alpha=0.75)
        ax.plot(xvals, y, marker=MARKERS[group], linestyle="", color=COLORS[group], label=group)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Top-{i}" for i in range(1, top_n + 1)])
    ax.invert_yaxis()
    ax.set_xlabel("Frequency in generated molecules")
    ax.set_title("E. Top MRSA scaffold frequency transfer")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Figure 4. Structure-level proximity and novelty", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, outdir / "figure_4_structure_level_proximity")
    plt.close(fig)


def plot_figure_5(tanimoto_detail, args, outdir):
    df = tanimoto_detail[tanimoto_detail["Method"] == "RCFR-Diffusion for MRSA"].copy()
    df = df.dropna(subset=["Max Tanimoto"])

    selected = df[
        (df["Max Tanimoto"] >= args.useful_tanimoto_low)
        & (df["Max Tanimoto"] <= args.case_max_tanimoto)
    ].copy()

    if selected.empty:
        selected = df.copy()

    selected["case_score"] = (
        pd.to_numeric(selected["Activity probability"], errors="coerce").fillna(0)
        - 0.08 * pd.to_numeric(selected["Predicted MIC"], errors="coerce").fillna(selected["Predicted MIC"].median())
        + 0.25 * pd.to_numeric(selected["QED"], errors="coerce").fillna(0)
        - 0.05 * pd.to_numeric(selected["SA"], errors="coerce").fillna(5)
        + 0.2 * pd.to_numeric(selected["Max Tanimoto"], errors="coerce").fillna(0)
    )

    selected = selected.sort_values("case_score", ascending=False).head(args.case_examples).copy()
    selected.to_csv(outdir / "figure_5_representative_candidate_cases.csv", index=False, encoding="utf-8-sig")

    mols = []
    legends = []

    for _, row in selected.iterrows():
        gen_mol = mol_from_smiles(row["smiles"])
        ref_mol = mol_from_smiles(row["nearest_reference_smiles"])

        if gen_mol is None or ref_mol is None:
            continue

        t = row["Max Tanimoto"]
        p = row["Activity probability"]
        mic = row["Predicted MIC"]
        qed = row["QED"]
        sa = row["SA"]

        mols.append(gen_mol)
        legends.append(f"RCFR candidate\nT={t:.3f} P={p:.3f}\nMIC={mic:.3g} QED={qed:.2f} SA={sa:.2f}")

        mols.append(ref_mol)
        legends.append(f"Nearest MRSA reference\nT={t:.3f}")

    if mols:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=2,
            subImgSize=(430, 280),
            legends=legends,
            useSVG=False,
        )
        img.save(str(outdir / "figure_5_representative_generated_candidates.png"))

        svg = Draw.MolsToGridImage(
            mols,
            molsPerRow=2,
            subImgSize=(430, 280),
            legends=legends,
            useSVG=True,
        )
        (outdir / "figure_5_representative_generated_candidates.svg").write_text(svg, encoding="utf-8")


def read_sdf(path: Path, top_n: int):
    mols = []

    if not path.exists():
        return mols

    try:
        suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        for mol in suppl:
            if mol is None:
                continue
            try:
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol)
            except Exception:
                try:
                    Chem.SanitizeMol(
                        mol,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                    )
                except Exception:
                    continue
            mols.append(mol)
            if top_n > 0 and len(mols) >= top_n:
                break
    except Exception:
        return []

    return mols


def uff_metrics(mol: Chem.Mol) -> Dict[str, float]:
    if mol.GetNumConformers() == 0:
        return {
            "Has 3D": 0,
            "UFF success": 0,
            "Energy before": np.nan,
            "Energy after": np.nan,
            "Relaxation RMSD": np.nan,
        }

    try:
        mol_h = Chem.AddHs(mol, addCoords=True)
    except Exception:
        mol_h = Chem.Mol(mol)

    try:
        ff = AllChem.UFFGetMoleculeForceField(mol_h)
        if ff is None:
            raise RuntimeError("UFF unavailable")

        before = float(ff.CalcEnergy())
        opt = Chem.Mol(mol_h)
        code = AllChem.UFFOptimizeMolecule(opt, maxIters=200)
        ff2 = AllChem.UFFGetMoleculeForceField(opt)
        after = float(ff2.CalcEnergy()) if ff2 is not None else np.nan

        try:
            rmsd = float(AllChem.GetBestRMS(mol_h, opt))
        except Exception:
            rmsd = np.nan

        return {
            "Has 3D": 1,
            "UFF success": int(code == 0),
            "Energy before": before,
            "Energy after": after,
            "Relaxation RMSD": rmsd,
        }
    except Exception:
        return {
            "Has 3D": 1,
            "UFF success": 0,
            "Energy before": np.nan,
            "Energy after": np.nan,
            "Relaxation RMSD": np.nan,
        }


def plot_supplementary_3d(args, outdir):
    paths = {
        "General-Model Generated": Path(args.original_sdf),
        "MRSA-Trained Generated": Path(args.scratch_sdf),
        "RCFR-Diffusion for MRSA": Path(args.rcfr_sdf),
    }

    detail_rows = []
    summary_rows = []

    for group, path in paths.items():
        mols = read_sdf(path, args.sdf_top_n)

        if not mols:
            summary_rows.append({
                "Method": group,
                "Status": "Skipped",
                "Reason": f"SDF missing or unreadable: {path}",
            })
            continue

        rows = []
        for i, mol in enumerate(mols):
            m = uff_metrics(mol)
            m["Method"] = group
            m["Index"] = i
            rows.append(m)
            detail_rows.append(m)

        df = pd.DataFrame(rows)

        summary_rows.append({
            "Method": group,
            "Status": "Done",
            "Molecules": len(df),
            "3D conformer ratio ↑": float(df["Has 3D"].mean()),
            "UFF success ratio ↑": float(df["UFF success"].mean()),
            "Energy before mean ↓": float(pd.to_numeric(df["Energy before"], errors="coerce").mean()),
            "Energy after mean ↓": float(pd.to_numeric(df["Energy after"], errors="coerce").mean()),
            "Relaxation RMSD mean ↓": float(pd.to_numeric(df["Relaxation RMSD"], errors="coerce").mean()),
        })

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)

    detail.to_csv(outdir / "supplementary_3d_quality_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "supplementary_3d_quality_summary.csv", index=False, encoding="utf-8-sig")

    done = summary[summary["Status"] == "Done"].copy()

    if done.empty:
        return summary

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    metrics = [
        "UFF success ratio ↑",
        "Energy after mean ↓",
        "Relaxation RMSD mean ↓",
    ]

    for ax, metric in zip(axes, metrics):
        vals = done.set_index("Method").reindex(GROUPS)[metric]
        ax.bar(vals.index, vals.values, color=[COLORS[g] for g in GROUPS])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle("Supplementary Figure S1. 3D conformation plausibility", y=1.02)
    fig.tight_layout()
    save_figure(fig, outdir / "supplementary_figure_s1_3d_conformation_plausibility")
    plt.close(fig)

    return summary



### BILINGUAL_EXPORT_PATCH_BEGIN

def setup_chinese_font():
    try:
        from matplotlib import font_manager, rcParams
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for fp in candidates:
            if Path(fp).exists():
                font_manager.fontManager.addfont(fp)
                prop = font_manager.FontProperties(fname=fp)
                rcParams["font.family"] = prop.get_name()
                rcParams["axes.unicode_minus"] = False
                return prop.get_name()
    except Exception:
        pass
    return None


ZH_TEXT_MAP = {
    "Known MRSA Actives": "原始 MRSA 数据集",
    "General-Model Generated": "原始 Megalodon",
    "MRSA-Trained Generated": "MRSA-Trained Generated",
    "RCFR-Diffusion for MRSA": "RCFR-Diffusion for MRSA",

    "Figure 1. Multi-objective generation performance": "图 1. 多目标生成性能",
    "A. Key generation and drug-likeness metrics": "A. 关键生成质量与药物样性质指标",
    "B. Activity-MIC objective landscape": "B. 活性概率与预测 MIC 分布",
    "C. Top-K MIC enrichment": "C. Top-K 候选分子 MIC 富集",
    "D. Pareto view: QED vs predicted potency": "D. QED 与预测活性的 Pareto 分布",
    "Metric value": "指标值",
    "Activity probability": "活性概率",
    "log10 predicted MIC": "log10 预测 MIC",
    "Top-K candidates": "Top-K 候选分子",
    "Median predicted MIC": "预测 MIC 中位数",
    "-log10 predicted MIC": "-log10 预测 MIC",
    "Ideal region\nhigh probability\nlow MIC": "理想区域\n高活性概率\n低 MIC",
    "1 / SA": "1 / SA",
    "Validity ↑": "有效性 ↑",
    "Diversity ↑": "多样性 ↑",
    "QED ↑": "QED ↑",
    "SA ↓": "SA ↓",

    "Figure 2. Distribution matching to original MRSA data": "图 2. 与原始 MRSA 数据的分布匹配",
    "B. Descriptor distribution distance to MRSA dataset": "B. 相对 MRSA 数据集的描述符分布距离",
    "C. Mean property distance": "C. 平均性质分布距离",
    "Quantile distance, lower is better": "分位数距离，越低越好",
    "Distance, lower is better": "距离，越低越好",
    "Density": "密度",

    "Figure 3. Chemical-space proximity to MRSA molecules": "图 3. 与 MRSA 分子的化学空间接近性",
    "A. Known MRSA Actives density": "A. 原始 MRSA 数据集密度",
    "E. Chemical-space proximity statistics": "E. 化学空间接近性统计",
    "High-density hit ratio": "高密度区域命中率",
    "Mean nearest distance": "平均最近邻距离",
    "Hit ratio, higher is better": "命中率，越高越好",
    "Distance, lower is better": "距离，越低越好",
    "UMAP 1": "UMAP 1",
    "UMAP 2": "UMAP 2",
    "PCA 1": "PCA 1",
    "PCA 2": "PCA 2",

    "Figure 4. Structure-level proximity and novelty": "图 4. 结构层面的接近性与新颖性",
    "A. Nearest-neighbor Tanimoto ECDF": "A. 最近邻 Tanimoto 阈值曲线",
    "B. Similarity-activity relationship": "B. 结构相似性与活性关系",
    "C. Tanimoto bin composition": "C. Tanimoto 相似性分段组成",
    "D. Scaffold and fragment transfer": "D. 骨架与关键片段转移",
    "E. Top MRSA scaffold frequency transfer": "E. MRSA 高频骨架分布转移",
    "Tanimoto threshold": "Tanimoto 阈值",
    "Ratio above threshold": "高于阈值的比例",
    "Max Tanimoto to MRSA": "相对 MRSA 的最大 Tanimoto",
    "Useful analogue region": "有效类似物区间",
    "Frequency in generated molecules": "在生成分子中的频率",
    "Coverage ratio": "覆盖比例",
    "Ratio": "比例",

    "Molecule scaffold coverage ↑": "分子骨架覆盖率 ↑",
    "Fragment group coverage ↑": "关键片段组覆盖率 ↑",

    "Supplementary Figure S1. 3D conformation plausibility": "补充图 S1. 三维构象合理性",
    "UFF success ratio ↑": "UFF 优化成功率 ↑",
    "Energy after mean ↓": "优化后能量均值 ↓",
    "Relaxation RMSD mean ↓": "构象优化 RMSD 均值 ↓",
}


def translate_text_object(text_obj):
    s = text_obj.get_text()
    if not isinstance(s, str) or not s:
        return

    for en, zh in sorted(ZH_TEXT_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        s = s.replace(en, zh)

    text_obj.set_text(s)


def export_english_copies(outdir: Path):
    import shutil

    suffixes = [".png", ".svg", ".pdf", ".mplfig.pkl"]
    prefixes = [
        "figure_1_multi_objective_generation_performance",
        "figure_2_distribution_matching_to_mrsa",
        "figure_3_chemical_space_proximity",
        "figure_4_structure_level_proximity",
        "supplementary_figure_s1_3d_conformation_plausibility",
    ]

    for prefix in prefixes:
        for suffix in suffixes:
            src = outdir / f"{prefix}{suffix}"
            if not src.exists():
                continue
            dst = outdir / f"{prefix}_en{suffix}"
            shutil.copy2(src, dst)

    # Figure 5 is generated by RDKit, usually only PNG/SVG.
    for suffix in [".png", ".svg"]:
        src = outdir / f"figure_5_representative_generated_candidates{suffix}"
        if src.exists():
            dst = outdir / f"figure_5_representative_generated_candidates_en{suffix}"
            shutil.copy2(src, dst)


def export_chinese_from_mplfig(outdir: Path):
    setup_chinese_font()

    prefixes = [
        "figure_1_multi_objective_generation_performance",
        "figure_2_distribution_matching_to_mrsa",
        "figure_3_chemical_space_proximity",
        "figure_4_structure_level_proximity",
        "supplementary_figure_s1_3d_conformation_plausibility",
    ]

    for prefix in prefixes:
        pkl_path = outdir / f"{prefix}.mplfig.pkl"
        if not pkl_path.exists():
            continue

        with open(pkl_path, "rb") as f:
            fig = pickle.load(f)

        for txt in fig.findobj(match=matplotlib.text.Text):
            translate_text_object(txt)

        with open(outdir / f"{prefix}_zh.mplfig.pkl", "wb") as f:
            pickle.dump(fig, f)

        fig.savefig(outdir / f"{prefix}_zh.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / f"{prefix}_zh.svg", bbox_inches="tight")
        fig.savefig(outdir / f"{prefix}_zh.pdf", bbox_inches="tight")
        plt.close(fig)


def export_chinese_figure_5(outdir: Path):
    setup_chinese_font()

    case_csv = outdir / "figure_5_representative_candidate_cases.csv"
    if not case_csv.exists():
        return

    df = pd.read_csv(case_csv)
    if df.empty:
        return

    n = len(df)
    fig_h = max(4.5, n * 2.6)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, fig_h))

    if n == 1:
        axes = np.array([axes])

    for i, (_, row) in enumerate(df.iterrows()):
        gen = mol_from_smiles(row["smiles"])
        ref = mol_from_smiles(row["nearest_reference_smiles"])

        for j, mol in enumerate([gen, ref]):
            ax = axes[i, j]
            ax.axis("off")

            if mol is not None:
                img = Draw.MolToImage(mol, size=(420, 260))
                ax.imshow(img)

            t = row.get("Max Tanimoto", np.nan)

            if j == 0:
                p = row.get("Activity probability", np.nan)
                mic = row.get("Predicted MIC", np.nan)
                qed = row.get("QED", np.nan)
                sa = row.get("SA", np.nan)
                title = f"RCFR候选分子\\nT={t:.3f}  活性={p:.3f}  MIC={mic:.3g}\\nQED={qed:.2f}  SA={sa:.2f}"
            else:
                title = f"最近邻MRSA参考分子\\nT={t:.3f}"

            ax.set_title(title, fontsize=10)

    fig.suptitle("图 5. 代表性 RCFR 候选分子及其最近邻 MRSA 参考分子", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    fig.savefig(outdir / "figure_5_representative_generated_candidates_zh.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "figure_5_representative_generated_candidates_zh.svg", bbox_inches="tight")
    fig.savefig(outdir / "figure_5_representative_generated_candidates_zh.pdf", bbox_inches="tight")

    with open(outdir / "figure_5_representative_generated_candidates_zh.mplfig.pkl", "wb") as f:
        pickle.dump(fig, f)

    plt.close(fig)


def export_bilingual_figures(outdir: Path):
    export_english_copies(outdir)
    export_chinese_from_mplfig(outdir)
    export_chinese_figure_5(outdir)

    bilingual_note = {
        "english_suffix": "_en",
        "chinese_suffix": "_zh",
        "note": "English copies are copied from the original generated figures. Chinese figures translate labels and titles only. All numeric results and CSV data are unchanged.",
    }
    (outdir / "bilingual_figure_manifest.json").write_text(
        json.dumps(bilingual_note, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

### BILINGUAL_EXPORT_PATCH_END


def write_readme(outdir, chem_method):
    text = f"""# Paper-style MRSA molecule generation results

This folder contains paper-style figures and tables.

## Core tables

- table_1_generation_druglikeness_benchmark.csv
- table_2_mrsa_goal_directed_candidate_quality.csv
- table_3_mrsa_space_proximity_benchmark.csv

## Core figures

- figure_1_multi_objective_generation_performance.png
- figure_2_distribution_matching_to_mrsa.png
- figure_3_chemical_space_proximity.png
- figure_4_structure_level_proximity.png
- figure_5_representative_generated_candidates.png
- supplementary_figure_s1_3d_conformation_plausibility.png

## Editable outputs

Each matplotlib figure is exported as PNG, SVG, PDF and MPLFIG.PKL.
SVG/PDF files can be edited in Visio, Illustrator, Inkscape or PowerPoint.
MPLFIG.PKL files can be modified with Python.
CSV files are the source tables for changing real plotted values.

## Chemical space method

The chemical-space figure used: {chem_method}
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference_csv", default="mrsa_scheme_a/processed_drug_mic_dataset_schemeA.csv")

    parser.add_argument("--original_ranked_csv", default="outputs/compare_guided/pretrained_baseline/pretrained_baseline_guided_all_sdf_ranked.csv")
    parser.add_argument("--scratch_ranked_csv", default="outputs/compare_guided/scratch_mrsa/scratch_mrsa_guided_all_sdf_ranked.csv")
    parser.add_argument("--rcfr_ranked_csv", default="outputs/compare_guided/finetune_mrsa/finetune_mrsa_guided_all_sdf_ranked.csv")

    parser.add_argument("--original_all_scored_csv", default="outputs/compare_guided/pretrained_baseline/pretrained_baseline_guided_all_sdf_all_scored.csv")
    parser.add_argument("--scratch_all_scored_csv", default="outputs/compare_guided/scratch_mrsa/scratch_mrsa_guided_all_sdf_all_scored.csv")
    parser.add_argument("--rcfr_all_scored_csv", default="outputs/compare_guided/finetune_mrsa/finetune_mrsa_guided_all_sdf_all_scored.csv")

    parser.add_argument("--original_summary_json", default="outputs/compare_guided/pretrained_baseline/pretrained_baseline_guided_all_sdf_summary.json")
    parser.add_argument("--scratch_summary_json", default="outputs/compare_guided/scratch_mrsa/scratch_mrsa_guided_all_sdf_summary.json")
    parser.add_argument("--rcfr_summary_json", default="outputs/compare_guided/finetune_mrsa/finetune_mrsa_guided_all_sdf_summary.json")

    parser.add_argument("--original_sdf", default="outputs/compare_guided/pretrained_baseline/pretrained_baseline_guided_all_sdf_top2000.sdf")
    parser.add_argument("--scratch_sdf", default="outputs/compare_guided/scratch_mrsa/scratch_mrsa_guided_all_sdf_top2000.sdf")
    parser.add_argument("--rcfr_sdf", default="outputs/compare_guided/finetune_mrsa/finetune_mrsa_guided_all_sdf_top2000.sdf")

    parser.add_argument("--outdir", default="outputs/paper_style_mrsa_figures")

    parser.add_argument("--top_n", type=int, default=200)
    parser.add_argument("--sampled_count", type=int, default=2000)
    parser.add_argument("--activity_threshold", type=float, default=0.5)
    parser.add_argument("--reference_active_only", action="store_true")
    parser.add_argument("--activity_col", default="activity_label")

    parser.add_argument("--qed_min", type=float, default=0.4)
    parser.add_argument("--sa_max", type=float, default=5.0)
    parser.add_argument("--mw_max", type=float, default=600.0)
    parser.add_argument("--logp_max", type=float, default=5.0)
    parser.add_argument("--tpsa_max", type=float, default=160.0)
    parser.add_argument("--hba_max", type=float, default=10.0)
    parser.add_argument("--hbd_max", type=float, default=5.0)
    parser.add_argument("--alert_max", type=float, default=0.0)

    parser.add_argument("--top_fragments", type=int, default=100)
    parser.add_argument("--top_scaffolds", type=int, default=15)
    parser.add_argument("--max_pairs", type=int, default=30000)
    parser.add_argument("--max_reference_plot", type=int, default=3000)
    parser.add_argument("--sdf_top_n", type=int, default=200)

    parser.add_argument("--useful_tanimoto_low", type=float, default=0.3)
    parser.add_argument("--useful_tanimoto_high", type=float, default=0.6)
    parser.add_argument("--case_max_tanimoto", type=float, default=0.7)
    parser.add_argument("--case_examples", type=int, default=8)

    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--high_density_quantile", type=float, default=0.20)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    required = [
        Path(args.reference_csv),
        Path(args.original_ranked_csv),
        Path(args.scratch_ranked_csv),
        Path(args.rcfr_ranked_csv),
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    reference, ranked, all_scored, summary_jsons = load_all(args)

    print("Loaded molecules:")
    print(f"  Known MRSA Actives: {len(reference)}")
    for group in GROUPS:
        print(f"  {group}: top={len(ranked[group])}, all={len(all_scored[group])}")

    table1 = build_table_1(reference, ranked, all_scored, summary_jsons, args, outdir)
    print("Saved Table 1")

    table2 = build_table_2(ranked, args, outdir)
    print("Saved Table 2")

    prop_detail, prop_summary = compute_property_distances(reference, ranked, outdir)
    tan_detail, tan_summary = compute_tanimoto(reference, ranked, args, outdir)
    scaf_summary, scaf_freq = compute_scaffold_fragment(reference, ranked, args, outdir)
    chem_coord, chem_summary, chem_method = chemical_space(reference, ranked, args, outdir)

    table3 = build_table_3(prop_summary, tan_summary, chem_summary, chem_method, scaf_summary, outdir)
    print("Saved Table 3")

    plot_figure_1(table1, table2, ranked, args, outdir)
    print("Saved Figure 1")

    plot_figure_2(reference, ranked, prop_detail, prop_summary, outdir)
    print("Saved Figure 2")

    plot_figure_3(chem_coord, chem_summary, chem_method, outdir)
    print("Saved Figure 3")

    plot_figure_4(tan_detail, scaf_summary, scaf_freq, args, outdir)
    print("Saved Figure 4")

    plot_figure_5(tan_detail, args, outdir)
    print("Saved Figure 5")

    plot_supplementary_3d(args, outdir)
    print("Saved Supplementary Figure S1")

    write_readme(outdir, chem_method)

    export_bilingual_figures(outdir)
    print("Saved bilingual English and Chinese figure versions")

    manifest = {
        "tables": [
            "table_1_generation_druglikeness_benchmark.csv",
            "table_2_mrsa_goal_directed_candidate_quality.csv",
            "table_3_mrsa_space_proximity_benchmark.csv",
        ],
        "figures": [
            "figure_1_multi_objective_generation_performance.png",
            "figure_2_distribution_matching_to_mrsa.png",
            "figure_3_chemical_space_proximity.png",
            "figure_4_structure_level_proximity.png",
            "figure_5_representative_generated_candidates.png",
            "supplementary_figure_s1_3d_conformation_plausibility.png",
        ],
        "chemical_space_method": chem_method,
    }

    (outdir / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Output directory: {outdir.resolve()}")


if __name__ == "__main__":
    main()
