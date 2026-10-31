#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
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


GROUP_ORDER = [
    "Known MRSA Actives",
    "General-Model Generated",
    "MRSA-Trained Generated",
    "RCFR-Diffusion for MRSA",
]

GENERATED_GROUPS = [
    "General-Model Generated",
    "MRSA-Trained Generated",
    "RCFR-Diffusion for MRSA",
]

GROUP_COLORS = {
    "Known MRSA Actives": "#1f77b4",
    "General-Model Generated": "#7f7f7f",
    "MRSA-Trained Generated": "#ff7f0e",
    "RCFR-Diffusion for MRSA": "#2ca02c",
}

DESCRIPTOR_COLS = [
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

PROPERTY_PLOT_COLS = [
    "MW",
    "LogP",
    "TPSA",
    "QED",
    "SA",
    "log10_MIC",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath.with_suffix(".png")), dpi=300, bbox_inches="tight")
    fig.savefig(str(outpath.with_suffix(".svg")), bbox_inches="tight")
    fig.savefig(str(outpath.with_suffix(".pdf")), bbox_inches="tight")


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def safe_ratio(num: float, den: float) -> float:
    if den == 0 or pd.isna(den):
        return float("nan")
    return float(num) / float(den)


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


def mol_from_smiles_raw(smiles: str) -> Optional[Chem.Mol]:
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


def largest_fragment_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return mol

        def frag_score(frag):
            heavy = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() > 1)
            carbon = int(any(a.GetSymbol() == "C" for a in frag.GetAtoms()))
            return carbon, heavy, frag.GetNumAtoms()

        best = sorted(frags, key=frag_score, reverse=True)[0]
        try:
            best.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(best)
        except Exception:
            pass
        smi = Chem.MolToSmiles(best, canonical=True)
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def clean_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    raw = mol_from_smiles_raw(smiles)
    if raw is None:
        return None
    mol = largest_fragment_mol(raw)
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


def calc_log10_mic(row: pd.Series, is_reference: bool) -> float:
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
        candidates = [
            "pred_log10_pmic",
            "pred_pmic_ug_ml",
            "pred_mic_ug_ml",
        ]

    for c in candidates:
        if c not in row.index:
            continue

        val = pd.to_numeric(row[c], errors="coerce")
        if pd.isna(val):
            continue

        val = float(val)

        if c == "pred_log10_pmic":
            return val

        if val > 0:
            return float(np.log10(val))

    return float("nan")


def get_numeric(row: pd.Series, candidates: List[str]) -> float:
    for c in candidates:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v):
                return float(v)
    return float("nan")


def load_molecule_csv(
    csv_path: Path,
    group_name: str,
    top_n: int,
    is_reference: bool,
    reference_active_only: bool,
    activity_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])

    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {csv_path}")

    if is_reference and reference_active_only and activity_col in df.columns:
        df = df[df[activity_col] == 1].copy()

    if (not is_reference) and top_n > 0:
        df = df.head(top_n).copy()

    rows = []

    for idx, raw in df.iterrows():
        smiles = raw[smiles_col]
        raw_mol = mol_from_smiles_raw(smiles)

        if raw_mol is None:
            continue

        try:
            n_frags = len(Chem.GetMolFrags(raw_mol, asMols=False))
        except Exception:
            n_frags = 1

        connected = int(n_frags == 1)
        mol = largest_fragment_mol(raw_mol)

        if mol is None:
            continue

        smi = canonical_smiles(mol)

        if smi is None:
            continue

        row = {
            "Group": group_name,
            "source_index": int(idx),
            "smiles": smi,
            "raw_smiles": str(smiles),
            "mol": mol,
            "connected": connected,
            "log10_MIC": calc_log10_mic(raw, is_reference=is_reference),
            "activity_label_probability": get_numeric(raw, ["activity_label_probability", "activity_probability", "activity_prob"]),
            "pred_pmic_ug_ml": get_numeric(raw, ["pred_pmic_ug_ml", "pred_mic_ug_ml", "pred_MIC_ug_ml"]),
            "pred_log10_pmic": get_numeric(raw, ["pred_log10_pmic"]),
            "mic_score": get_numeric(raw, ["mic_score"]),
            "total_score": get_numeric(raw, ["total_score"]),
            "lipinski_score": get_numeric(raw, ["lipinski_score"]),
        }

        desc = calc_desc(mol)
        row.update(desc)

        if "qed" in raw.index and pd.notna(pd.to_numeric(raw["qed"], errors="coerce")):
            row["QED_csv"] = float(pd.to_numeric(raw["qed"], errors="coerce"))

        if "alert_count" in raw.index and pd.notna(pd.to_numeric(raw["alert_count"], errors="coerce")):
            row["AlertCount"] = float(pd.to_numeric(raw["alert_count"], errors="coerce"))

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError(f"No valid molecules loaded from {csv_path}")

    if is_reference:
        out = out.drop_duplicates("smiles").reset_index(drop=True)

    return out


def mol_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def fp_to_numpy(fp, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def sparse_bits(mol: Chem.Mol, radius: int = 2) -> set:
    try:
        fp = AllChem.GetMorganFingerprint(mol, radius=radius)
        return set(fp.GetNonzeroElements().keys())
    except Exception:
        return set()


def scaffold_smi(mol: Chem.Mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return "NO_SCAFFOLD"
        smi = Chem.MolToSmiles(scaf, canonical=True)
        return smi if smi else "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


def average_pairwise_diversity(smiles_list: List[str], max_pairs: int, seed: int) -> float:
    mols = []
    for smi in sorted(set(smiles_list)):
        mol = clean_mol_from_smiles(smi)
        if mol is not None:
            mols.append(mol)

    if len(mols) < 2:
        return float("nan")

    fps = [mol_fp(m) for m in mols]
    n = len(fps)
    rng = np.random.default_rng(seed)

    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = []
        for _ in range(max_pairs):
            i, j = rng.choice(n, 2, replace=False)
            if i > j:
                i, j = j, i
            pairs.append((int(i), int(j)))

    vals = []
    for i, j in pairs:
        vals.append(1.0 - float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))

    return float(np.mean(vals)) if vals else float("nan")


def druglike_mask(df: pd.DataFrame, args) -> pd.Series:
    qed = pd.to_numeric(df["QED"], errors="coerce")
    sa = pd.to_numeric(df["SA"], errors="coerce")
    mw = pd.to_numeric(df["MW"], errors="coerce")
    logp = pd.to_numeric(df["LogP"], errors="coerce")
    tpsa = pd.to_numeric(df["TPSA"], errors="coerce")
    hba = pd.to_numeric(df["HBA"], errors="coerce")
    hbd = pd.to_numeric(df["HBD"], errors="coerce")
    alert = pd.to_numeric(df["AlertCount"], errors="coerce").fillna(0)

    return (
        (qed >= args.qed_min)
        & (sa <= args.sa_max)
        & (mw <= args.mw_max)
        & (logp <= args.logp_max)
        & (tpsa <= args.tpsa_max)
        & (hba <= args.hba_max)
        & (hbd <= args.hbd_max)
        & (alert <= args.alert_max)
    )


def normalize_series(s: pd.Series, higher_is_better: bool) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    mn = vals.min()
    mx = vals.max()

    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        return pd.Series([0.5] * len(vals), index=s.index)

    out = (vals - mn) / (mx - mn)

    if not higher_is_better:
        out = 1.0 - out

    return out


def quantile_distance(ref: np.ndarray, gen: np.ndarray, n: int = 101) -> float:
    ref = np.asarray(ref, dtype=float)
    gen = np.asarray(gen, dtype=float)
    ref = ref[np.isfinite(ref)]
    gen = gen[np.isfinite(gen)]

    if len(ref) == 0 or len(gen) == 0:
        return float("nan")

    qs = np.linspace(0, 1, n)
    r = np.quantile(ref, qs)
    g = np.quantile(gen, qs)

    scale = np.percentile(ref, 75) - np.percentile(ref, 25)
    if scale <= 1e-8:
        scale = np.std(ref)
    if scale <= 1e-8:
        scale = 1.0

    return float(np.mean(np.abs(r - g)) / scale)


def wasserstein(ref: np.ndarray, gen: np.ndarray) -> float:
    try:
        from scipy.stats import wasserstein_distance
        ref = np.asarray(ref, dtype=float)
        gen = np.asarray(gen, dtype=float)
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
    except Exception:
        return float("nan")


def ks_stat(ref: np.ndarray, gen: np.ndarray) -> float:
    try:
        from scipy.stats import ks_2samp
        ref = np.asarray(ref, dtype=float)
        gen = np.asarray(gen, dtype=float)
        ref = ref[np.isfinite(ref)]
        gen = gen[np.isfinite(gen)]
        if len(ref) == 0 or len(gen) == 0:
            return float("nan")
        return float(ks_2samp(ref, gen).statistic)
    except Exception:
        return float("nan")


def frechet_distance(X: np.ndarray, Y: np.ndarray) -> float:
    try:
        from scipy.linalg import sqrtm
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
    except Exception:
        return float("nan")


def load_inputs(args):
    reference = load_molecule_csv(
        csv_path=Path(args.reference_csv),
        group_name="Known MRSA Actives",
        top_n=0,
        is_reference=True,
        reference_active_only=args.reference_active_only,
        activity_col=args.activity_col,
    )

    ranked_tables = {
        "General-Model Generated": load_molecule_csv(
            Path(args.original_ranked_csv),
            "General-Model Generated",
            args.top_n,
            False,
            False,
            args.activity_col,
        ),
        "MRSA-Trained Generated": load_molecule_csv(
            Path(args.scratch_ranked_csv),
            "MRSA-Trained Generated",
            args.top_n,
            False,
            False,
            args.activity_col,
        ),
        "RCFR-Diffusion for MRSA": load_molecule_csv(
            Path(args.rcfr_ranked_csv),
            "RCFR-Diffusion for MRSA",
            args.top_n,
            False,
            False,
            args.activity_col,
        ),
    }

    all_scored_paths = {
        "General-Model Generated": Path(args.original_all_scored_csv),
        "MRSA-Trained Generated": Path(args.scratch_all_scored_csv),
        "RCFR-Diffusion for MRSA": Path(args.rcfr_all_scored_csv),
    }

    all_tables = {}
    for group, path in all_scored_paths.items():
        if path.exists():
            all_tables[group] = load_molecule_csv(
                path,
                group,
                0,
                False,
                False,
                args.activity_col,
            )
        else:
            all_tables[group] = ranked_tables[group].copy()

    summary_jsons = {
        "General-Model Generated": read_json(Path(args.original_summary_json)),
        "MRSA-Trained Generated": read_json(Path(args.scratch_summary_json)),
        "RCFR-Diffusion for MRSA": read_json(Path(args.rcfr_summary_json)),
    }

    return reference, ranked_tables, all_tables, summary_jsons


def experiment_1_generation_quality(reference, ranked, all_tables, summary_jsons, args, outdir):
    ref_smiles = set(reference["smiles"].tolist())
    rows = []

    for group in GENERATED_GROUPS:
        df = all_tables[group]
        unique_smiles = sorted(set(df["smiles"].dropna().astype(str).tolist()))
        novel_count = sum(1 for s in unique_smiles if s not in ref_smiles)
        sampled = int(summary_jsons.get(group, {}).get("generated_objects", args.sampled_count))
        valid = int(summary_jsons.get(group, {}).get("valid_raw_molecules", len(df)))
        connected = int(pd.to_numeric(df["connected"], errors="coerce").sum())
        scored = int(len(df))
        unique = int(len(unique_smiles))
        diversity = average_pairwise_diversity(unique_smiles, args.max_pairs, args.seed)

        rows.append({
            "Group": group,
            "Sampled molecules": sampled,
            "Valid raw molecules": valid,
            "Connected molecules": connected,
            "Scored molecules": scored,
            "Unique molecules": unique,
            "Novel molecules": novel_count,
            "Validity ↑": safe_ratio(valid, sampled),
            "Connectivity ↑": safe_ratio(connected, valid),
            "Uniqueness ↑": safe_ratio(unique, scored),
            "Novelty ↑": safe_ratio(novel_count, unique),
            "Diversity ↑": diversity,
        })

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "experiment_1_generation_quality.csv", index=False, encoding="utf-8-sig")

    metrics = ["Validity ↑", "Uniqueness ↑", "Novelty ↑", "Diversity ↑"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    for ax, metric in zip(axes, metrics):
        vals = table.set_index("Group").loc[GENERATED_GROUPS, metric]
        ax.bar(vals.index, vals.values, color=[GROUP_COLORS[g] for g in vals.index])
        ax.set_title(metric)
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(1.05, float(np.nanmax(vals.values)) * 1.15))
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(vals.values):
            if np.isfinite(v):
                ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=8)

    fig.tight_layout()
    save_fig(fig, outdir / "experiment_1_generation_quality")
    plt.close(fig)
    return table


def experiment_2_druglikeness(reference, ranked, args, outdir):
    rows = []
    long_rows = []

    for group in GENERATED_GROUPS:
        df = ranked[group].copy()
        mask = druglike_mask(df, args)

        row = {
            "Group": group,
            "Molecules": len(df),
            "Drug-like pass ratio ↑": float(mask.mean()),
        }

        for col in ["QED", "SA", "MW", "LogP", "TPSA", "HBA", "HBD", "RotB", "Rings", "AlertCount"]:
            vals = pd.to_numeric(df[col], errors="coerce")
            row[f"{col} mean"] = float(vals.mean())
            row[f"{col} std"] = float(vals.std())
            row[f"{col} median"] = float(vals.median())

            for v in vals.dropna().tolist():
                long_rows.append({"Group": group, "Property": col, "Value": v})

        rows.append(row)

    table = pd.DataFrame(rows)
    long = pd.DataFrame(long_rows)

    table.to_csv(outdir / "experiment_2_druglikeness_summary.csv", index=False, encoding="utf-8-sig")
    long.to_csv(outdir / "experiment_2_druglikeness_long.csv", index=False, encoding="utf-8-sig")

    props = ["QED", "SA", "MW", "LogP", "TPSA"]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.8))

    for ax, prop in zip(axes, props):
        box_data = [
            long[(long["Group"] == g) & (long["Property"] == prop)]["Value"].dropna().values
            for g in GENERATED_GROUPS
        ]
        ax.boxplot(box_data, labels=GENERATED_GROUPS, showmeans=True)
        ax.set_title(prop)
        ax.set_ylabel(prop)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    save_fig(fig, outdir / "experiment_2_druglikeness_boxplots")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    vals = table.set_index("Group").loc[GENERATED_GROUPS, "Drug-like pass ratio ↑"]
    ax.bar(vals.index, vals.values, color=[GROUP_COLORS[g] for g in vals.index])
    ax.set_title("Drug-like pass ratio")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(vals.values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, outdir / "experiment_2_druglike_pass_ratio")
    plt.close(fig)

    return table


def experiment_3_activity_proxy(ranked, args, outdir):
    rows = []
    long_rows = []

    for group in GENERATED_GROUPS:
        df = ranked[group].copy()

        prob = pd.to_numeric(df["activity_label_probability"], errors="coerce")
        mic = pd.to_numeric(df["pred_pmic_ug_ml"], errors="coerce")
        logmic = pd.to_numeric(df["pred_log10_pmic"], errors="coerce")
        if logmic.isna().all():
            logmic = np.log10(mic.where(mic > 0))

        row = {
            "Group": group,
            "Molecules": len(df),
            "Activity probability mean ↑": float(prob.mean()),
            "Activity probability median ↑": float(prob.median()),
            "Active ratio ↑": float((prob >= args.activity_threshold).mean()),
            "Predicted MIC mean ↓": float(mic.mean()),
            "Predicted MIC median ↓": float(mic.median()),
            "Predicted log10 MIC mean ↓": float(pd.Series(logmic).mean()),
            "Predicted log10 MIC median ↓": float(pd.Series(logmic).median()),
            "MIC score mean ↑": float(pd.to_numeric(df["mic_score"], errors="coerce").mean()),
            "Total score mean ↑": float(pd.to_numeric(df["total_score"], errors="coerce").mean()),
        }
        rows.append(row)

        for v in prob.dropna().tolist():
            long_rows.append({"Group": group, "Metric": "Activity probability", "Value": v})
        for v in pd.Series(logmic).dropna().tolist():
            long_rows.append({"Group": group, "Metric": "Predicted log10 MIC", "Value": v})
        if "total_score" in df.columns:
            for v in pd.to_numeric(df["total_score"], errors="coerce").dropna().tolist():
                long_rows.append({"Group": group, "Metric": "Total score", "Value": v})

    table = pd.DataFrame(rows)
    long = pd.DataFrame(long_rows)

    table.to_csv(outdir / "experiment_3_activity_proxy_summary.csv", index=False, encoding="utf-8-sig")
    long.to_csv(outdir / "experiment_3_activity_proxy_long.csv", index=False, encoding="utf-8-sig")

    metrics = ["Activity probability", "Predicted log10 MIC", "Total score"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, metric in zip(axes, metrics):
        for group in GENERATED_GROUPS:
            vals = long[(long["Group"] == group) & (long["Metric"] == metric)]["Value"].dropna().values
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=30, density=True, histtype="step", linewidth=2, color=GROUP_COLORS[group], label=group)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, outdir / "experiment_3_activity_proxy_distributions")
    plt.close(fig)

    return table


def experiment_4_topk_enrichment(ranked, args, outdir):
    ks = [20, 50, 100, 200]
    rows = []

    for group in GENERATED_GROUPS:
        df = ranked[group].copy()

        for k in ks:
            sub = df.head(min(k, len(df))).copy()
            prob = pd.to_numeric(sub["activity_label_probability"], errors="coerce")
            mic = pd.to_numeric(sub["pred_pmic_ug_ml"], errors="coerce")
            mask = druglike_mask(sub, args)

            rows.append({
                "Group": group,
                "TopK": k,
                "Activity probability mean ↑": float(prob.mean()),
                "Active ratio ↑": float((prob >= args.activity_threshold).mean()),
                "Predicted MIC median ↓": float(mic.median()),
                "QED mean ↑": float(pd.to_numeric(sub["QED"], errors="coerce").mean()),
                "SA mean ↓": float(pd.to_numeric(sub["SA"], errors="coerce").mean()),
                "Drug-like pass ratio ↑": float(mask.mean()),
            })

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "experiment_4_topk_enrichment.csv", index=False, encoding="utf-8-sig")

    metrics = [
        ("Active ratio ↑", "Ratio"),
        ("Predicted MIC median ↓", "MIC, μg/mL"),
        ("Drug-like pass ratio ↑", "Ratio"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, (metric, ylabel) in zip(axes, metrics):
        for group in GENERATED_GROUPS:
            sub = table[table["Group"] == group]
            ax.plot(sub["TopK"], sub[metric], marker="o", linewidth=2, label=group, color=GROUP_COLORS[group])
        ax.set_title(metric)
        ax.set_xlabel("Top-K")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, outdir / "experiment_4_topk_enrichment")
    plt.close(fig)

    return table


def experiment_5_property_closeness(reference, ranked, outdir):
    all_tables = {"Known MRSA Actives": reference, **ranked}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8))
    axes = axes.flatten()

    for ax, col in zip(axes, PROPERTY_PLOT_COLS):
        all_vals = []
        for group in GROUP_ORDER:
            vals = pd.to_numeric(all_tables[group][col], errors="coerce").dropna().values.astype(float)
            all_vals.extend(vals.tolist())

        if len(all_vals) == 0:
            ax.axis("off")
            continue

        low = np.nanpercentile(all_vals, 1)
        high = np.nanpercentile(all_vals, 99)
        if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-8:
            low = np.nanmin(all_vals)
            high = np.nanmax(all_vals)

        bins = np.linspace(low, high, 35)

        for group in GROUP_ORDER:
            vals = pd.to_numeric(all_tables[group][col], errors="coerce").dropna().values.astype(float)
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals >= low) & (vals <= high)]
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=2.1, color=GROUP_COLORS[group], label=group)

        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.87, wspace=0.25, hspace=0.32)
    save_fig(fig, outdir / "experiment_5_property_distribution_overlap")
    plt.close(fig)

    rows = []
    desc_for_distance = DESCRIPTOR_COLS + ["log10_MIC"]

    for group in GENERATED_GROUPS:
        df = ranked[group]

        for col in desc_for_distance:
            ref_vals = pd.to_numeric(reference[col], errors="coerce").dropna().values.astype(float)
            gen_vals = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(float)

            rows.append({
                "Group": group,
                "Descriptor": col,
                "Quantile distance ↓": quantile_distance(ref_vals, gen_vals),
                "Wasserstein distance ↓": wasserstein(ref_vals, gen_vals),
                "KS statistic ↓": ks_stat(ref_vals, gen_vals),
                "Reference mean": float(np.nanmean(ref_vals)) if len(ref_vals) else np.nan,
                "Generated mean": float(np.nanmean(gen_vals)) if len(gen_vals) else np.nan,
            })

    detail = pd.DataFrame(rows)
    detail.to_csv(outdir / "experiment_5_property_distance_detail.csv", index=False, encoding="utf-8-sig")

    summary = detail.groupby("Group").agg({
        "Quantile distance ↓": "mean",
        "Wasserstein distance ↓": "mean",
        "KS statistic ↓": "mean",
    }).reset_index().rename(columns={
        "Quantile distance ↓": "Mean quantile distance ↓",
        "Wasserstein distance ↓": "Mean Wasserstein distance ↓",
        "KS statistic ↓": "Mean KS statistic ↓",
    })

    ref_X = reference[DESCRIPTOR_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    ref_mean = np.nanmean(ref_X, axis=0)
    ref_std = np.nanstd(ref_X, axis=0)
    ref_std[ref_std <= 1e-8] = 1.0
    ref_Z = (ref_X - ref_mean) / ref_std

    fcd_rows = []
    for group in GENERATED_GROUPS:
        gen_X = ranked[group][DESCRIPTOR_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
        gen_Z = (gen_X - ref_mean) / ref_std
        fcd_rows.append({
            "Group": group,
            "Descriptor Frechet distance ↓": frechet_distance(ref_Z, gen_Z),
        })

    fcd_table = pd.DataFrame(fcd_rows)
    summary = summary.merge(fcd_table, on="Group", how="left")
    summary.to_csv(outdir / "experiment_5_property_distance_summary.csv", index=False, encoding="utf-8-sig")

    pivot = detail.pivot(index="Descriptor", columns="Group", values="Quantile distance ↓")
    plot_rows = ["MW", "LogP", "TPSA", "QED", "SA", "HBA", "HBD", "RotB", "Rings", "log10_MIC"]
    pivot = pivot.reindex(index=plot_rows, columns=GENERATED_GROUPS)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(np.arange(len(GENERATED_GROUPS)))
    ax.set_xticklabels(GENERATED_GROUPS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(plot_rows)))
    ax.set_yticklabels(plot_rows)
    ax.set_title("Property distribution distance to original MRSA dataset")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Quantile distance, lower is better")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    save_fig(fig, outdir / "experiment_5_property_distance_heatmap")
    plt.close(fig)

    return detail, summary


def experiment_6_tanimoto(reference, ranked, args, outdir):
    ref_fps = [mol_fp(m) for m in reference["mol"]]
    detail_rows = []
    summary_rows = []

    for group in GENERATED_GROUPS:
        sims = []

        for _, row in ranked[group].iterrows():
            fp = mol_fp(row["mol"])
            vals = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)

            if vals:
                best_idx = int(np.argmax(vals))
                max_sim = float(vals[best_idx])
                nearest = reference.iloc[best_idx]["smiles"]
            else:
                max_sim = float("nan")
                nearest = ""

            sims.append(max_sim)
            detail_rows.append({
                "Group": group,
                "smiles": row["smiles"],
                "max_tanimoto_to_original_mrsa": max_sim,
                "nearest_reference_smiles": nearest,
                "activity_label_probability": row.get("activity_label_probability", np.nan),
                "pred_pmic_ug_ml": row.get("pred_pmic_ug_ml", np.nan),
                "QED": row.get("QED", np.nan),
                "SA": row.get("SA", np.nan),
                "total_score": row.get("total_score", np.nan),
            })

        arr = np.asarray(sims, dtype=float)
        arr = arr[np.isfinite(arr)]

        summary_rows.append({
            "Group": group,
            "Mean max Tanimoto ↑": float(np.mean(arr)) if len(arr) else np.nan,
            "Median max Tanimoto ↑": float(np.median(arr)) if len(arr) else np.nan,
            "Tanimoto >= 0.3 ratio ↑": safe_ratio(float(np.sum(arr >= 0.3)), len(arr)),
            "Tanimoto >= 0.4 ratio ↑": safe_ratio(float(np.sum(arr >= 0.4)), len(arr)),
            "Tanimoto >= 0.5 ratio ↑": safe_ratio(float(np.sum(arr >= 0.5)), len(arr)),
        })

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)

    detail.to_csv(outdir / "experiment_6_tanimoto_nearest_neighbor_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "experiment_6_tanimoto_nearest_neighbor_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    box_data = [
        detail[detail["Group"] == g]["max_tanimoto_to_original_mrsa"].dropna().values
        for g in GENERATED_GROUPS
    ]

    axes[0].boxplot(box_data, labels=GENERATED_GROUPS, showmeans=True)
    axes[0].set_title("Nearest-neighbor Tanimoto similarity")
    axes[0].set_ylabel("Max Tanimoto to original MRSA dataset")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)

    thresholds = np.linspace(0, 1, 101)
    for group in GENERATED_GROUPS:
        vals = detail[detail["Group"] == group]["max_tanimoto_to_original_mrsa"].dropna().values
        if len(vals) == 0:
            continue
        survival = [(vals >= t).mean() for t in thresholds]
        axes[1].plot(thresholds, survival, linewidth=2, color=GROUP_COLORS[group], label=group)

    axes[1].set_title("Similarity enrichment curve")
    axes[1].set_xlabel("Tanimoto threshold")
    axes[1].set_ylabel("Ratio above threshold")
    axes[1].grid(True, linestyle="--", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, outdir / "experiment_6_tanimoto_similarity")
    plt.close(fig)

    return detail, summary


def make_chemical_features(reference, ranked, args):
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError("scikit-learn is required for chemical space visualization") from e

    ref = reference.copy()
    if args.max_reference_plot > 0 and len(ref) > args.max_reference_plot:
        ref = ref.sample(n=args.max_reference_plot, random_state=args.seed).copy()

    records = []

    for _, row in ref.iterrows():
        fp = fp_to_numpy(mol_fp(row["mol"]))
        desc = row[["MW", "LogP", "TPSA", "QED", "SA", "HBA", "HBD", "RotB", "Rings"]].astype(float).values
        records.append((row["Group"], row["smiles"], fp, desc))

    for group in GENERATED_GROUPS:
        for _, row in ranked[group].iterrows():
            fp = fp_to_numpy(mol_fp(row["mol"]))
            desc = row[["MW", "LogP", "TPSA", "QED", "SA", "HBA", "HBD", "RotB", "Rings"]].astype(float).values
            records.append((group, row["smiles"], fp, desc))

    fps = np.vstack([r[2] for r in records])
    descs = np.vstack([r[3] for r in records])
    descs = np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)
    descs = StandardScaler().fit_transform(descs)

    X = np.hstack([fps, descs])
    meta = pd.DataFrame({
        "Group": [r[0] for r in records],
        "smiles": [r[1] for r in records],
    })

    return X, meta


def chemical_space_summary(coord_df: pd.DataFrame, outdir: Path, prefix: str):
    ref = coord_df[coord_df["Group"] == "Known MRSA Actives"][["X1", "X2"]].values.astype(float)
    ref_centroid = ref.mean(axis=0)
    ref_cov = np.cov(ref, rowvar=False)

    try:
        inv_cov = np.linalg.pinv(ref_cov)
        cutoff = 5.991
    except Exception:
        inv_cov = None
        cutoff = None

    rows = []

    for group in GENERATED_GROUPS:
        pts = coord_df[coord_df["Group"] == group][["X1", "X2"]].values.astype(float)
        centroid = pts.mean(axis=0)

        nn_dists = []
        inside = []

        for p in pts:
            nn_dists.append(float(np.min(np.linalg.norm(ref - p, axis=1))))
            if inv_cov is not None:
                diff = p - ref_centroid
                md2 = float(diff.T.dot(inv_cov).dot(diff))
                inside.append(int(md2 <= cutoff))

        rows.append({
            "Group": group,
            f"{prefix} centroid distance to MRSA ↓": float(np.linalg.norm(centroid - ref_centroid)),
            f"{prefix} mean nearest distance to MRSA ↓": float(np.mean(nn_dists)),
            f"{prefix} median nearest distance to MRSA ↓": float(np.median(nn_dists)),
            f"{prefix} MRSA cloud overlap ratio ↑": float(np.mean(inside)) if inside else np.nan,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / f"experiment_7_{prefix.lower()}_chemical_space_distance_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_chemical_space(coord_df: pd.DataFrame, outdir: Path, prefix: str):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True, sharey=True)

    for ax, group in zip(axes, GENERATED_GROUPS):
        ref = coord_df[coord_df["Group"] == "Known MRSA Actives"]
        gen = coord_df[coord_df["Group"] == group]

        ax.scatter(ref["X1"], ref["X2"], s=8, alpha=0.18, color=GROUP_COLORS["Known MRSA Actives"], label="Known MRSA Actives")
        ax.scatter(gen["X1"], gen["X2"], s=28, alpha=0.75, color=GROUP_COLORS[group], label=group)

        ax.set_title(group)
        ax.set_xlabel(f"{prefix} 1")
        ax.set_ylabel(f"{prefix} 2")
        ax.grid(True, linestyle="--", alpha=0.22)
        ax.legend(fontsize=8)

    fig.suptitle(f"{prefix} chemical space proximity to original MRSA dataset", y=1.02)
    fig.tight_layout()
    save_fig(fig, outdir / f"experiment_7_{prefix.lower()}_chemical_space_panels")
    plt.close(fig)


def experiment_7_chemical_space(reference, ranked, args, outdir):
    try:
        from sklearn.decomposition import PCA
    except Exception:
        note = pd.DataFrame([{"Status": "Skipped", "Reason": "scikit-learn is not available"}])
        note.to_csv(outdir / "experiment_7_chemical_space_skipped.csv", index=False)
        return pd.DataFrame()

    X, meta = make_chemical_features(reference, ranked, args)

    pca = PCA(n_components=2, random_state=args.seed)
    coords = pca.fit_transform(X)

    coord_df = meta.copy()
    coord_df["X1"] = coords[:, 0]
    coord_df["X2"] = coords[:, 1]
    coord_df.to_csv(outdir / "experiment_7_pca_chemical_space_coordinates.csv", index=False, encoding="utf-8-sig")

    pca_summary = chemical_space_summary(coord_df, outdir, "PCA")
    plot_chemical_space(coord_df, outdir, "PCA")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    vals = pca_summary.set_index("Group").loc[GENERATED_GROUPS, "PCA mean nearest distance to MRSA ↓"]
    ax.bar(vals.index, vals.values, color=[GROUP_COLORS[g] for g in vals.index])
    ax.set_title("Mean nearest distance in PCA chemical space")
    ax.set_ylabel("Distance, lower is better")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(vals.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, outdir / "experiment_7_pca_distance_bar")
    plt.close(fig)

    summaries = [pca_summary]

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=args.seed, n_neighbors=30, min_dist=0.1)
        ucoords = reducer.fit_transform(X)

        ucoord_df = meta.copy()
        ucoord_df["X1"] = ucoords[:, 0]
        ucoord_df["X2"] = ucoords[:, 1]
        ucoord_df.to_csv(outdir / "experiment_7_umap_chemical_space_coordinates.csv", index=False, encoding="utf-8-sig")

        umap_summary = chemical_space_summary(ucoord_df, outdir, "UMAP")
        plot_chemical_space(ucoord_df, outdir, "UMAP")
        summaries.append(umap_summary)
    except Exception as e:
        pd.DataFrame([{"Status": "Skipped", "Reason": str(e)}]).to_csv(outdir / "experiment_7_umap_skipped.csv", index=False, encoding="utf-8-sig")

    return pca_summary


def experiment_8_scaffold_fragment(reference, ranked, args, outdir):
    ref_scaffolds = set(scaffold_smi(m) for m in reference["mol"])
    ref_scaffolds = set(x for x in ref_scaffolds if x != "NO_SCAFFOLD")

    ref_scaf_counter = Counter(scaffold_smi(m) for m in reference["mol"])
    ref_scaf_counter.pop("NO_SCAFFOLD", None)
    top_ref_scaffolds = [x for x, _ in ref_scaf_counter.most_common(args.top_scaffolds)]

    ref_bit_counter = Counter()
    for m in reference["mol"]:
        ref_bit_counter.update(sparse_bits(m))

    top_bits = set(bit for bit, _ in ref_bit_counter.most_common(args.top_fragments))

    rows = []
    detail_rows = []
    scaf_freq_rows = []

    for group in GENERATED_GROUPS:
        df = ranked[group]

        scaf_list = []
        group_bits = set()
        mol_scaf_hits = []
        mol_frag_cov = []

        for _, row in df.iterrows():
            mol = row["mol"]
            scaf = scaffold_smi(mol)
            bits = sparse_bits(mol)
            group_bits.update(bits)

            scaf_hit = int(scaf in ref_scaffolds)
            frag_cov = safe_ratio(len(bits & top_bits), len(top_bits))

            scaf_list.append(scaf)
            mol_scaf_hits.append(scaf_hit)
            mol_frag_cov.append(frag_cov)

            detail_rows.append({
                "Group": group,
                "smiles": row["smiles"],
                "scaffold": scaf,
                "scaffold_in_original_mrsa": scaf_hit,
                "key_fragment_coverage": frag_cov,
            })

        scaf_set = set(x for x in scaf_list if x != "NO_SCAFFOLD")
        overlap = scaf_set & ref_scaffolds

        rows.append({
            "Group": group,
            "Generated scaffold count": len(scaf_set),
            "Overlapped scaffold count": len(overlap),
            "Scaffold set overlap ratio ↑": safe_ratio(len(overlap), len(scaf_set)),
            "Molecule scaffold coverage ↑": float(np.mean(mol_scaf_hits)) if mol_scaf_hits else np.nan,
            "Key fragment group coverage ↑": safe_ratio(len(group_bits & top_bits), len(top_bits)),
            "Mean key fragment coverage ↑": float(np.mean(mol_frag_cov)) if mol_frag_cov else np.nan,
        })

        cnt = Counter(scaf_list)
        n = max(1, len(scaf_list))
        for i, scaf in enumerate(top_ref_scaffolds, start=1):
            scaf_freq_rows.append({
                "Group": group,
                "Reference scaffold rank": i,
                "Scaffold": scaf,
                "Frequency ratio": cnt.get(scaf, 0) / n,
            })

    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)
    scaf_freq = pd.DataFrame(scaf_freq_rows)

    summary.to_csv(outdir / "experiment_8_scaffold_fragment_summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(outdir / "experiment_8_scaffold_fragment_detail.csv", index=False, encoding="utf-8-sig")
    scaf_freq.to_csv(outdir / "experiment_8_top_reference_scaffold_frequency.csv", index=False, encoding="utf-8-sig")

    metrics = [
        "Scaffold set overlap ratio ↑",
        "Molecule scaffold coverage ↑",
        "Key fragment group coverage ↑",
        "Mean key fragment coverage ↑",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    for ax, metric in zip(axes, metrics):
        vals = summary.set_index("Group").loc[GENERATED_GROUPS, metric]
        ax.bar(vals.index, vals.values, color=[GROUP_COLORS[g] for g in vals.index])
        ax.set_title(metric)
        ax.set_ylabel("Ratio")
        ax.set_ylim(0, min(1.05, max(0.2, float(np.nanmax(vals.values)) * 1.25)))
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(vals.values):
            if np.isfinite(v):
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    save_fig(fig, outdir / "experiment_8_scaffold_fragment_coverage")
    plt.close(fig)

    return summary


def experiment_9_representative_pairs(tanimoto_detail, args, outdir):
    df = tanimoto_detail[tanimoto_detail["Group"] == "RCFR-Diffusion for MRSA"].copy()
    df = df.dropna(subset=["max_tanimoto_to_original_mrsa"])

    if df.empty:
        pd.DataFrame([{"Status": "Skipped", "Reason": "No RCFR nearest-neighbor records"}]).to_csv(outdir / "experiment_9_representative_pairs_skipped.csv", index=False)
        return pd.DataFrame()

    mid = df[(df["max_tanimoto_to_original_mrsa"] >= args.pair_min_tanimoto) & (df["max_tanimoto_to_original_mrsa"] <= args.pair_max_tanimoto)].copy()

    if mid.empty:
        mid = df.copy()

    mid["selection_score"] = (
        pd.to_numeric(mid["activity_label_probability"], errors="coerce").fillna(0)
        - 0.05 * pd.to_numeric(mid["pred_pmic_ug_ml"], errors="coerce").fillna(mid["pred_pmic_ug_ml"].median())
        + 0.2 * pd.to_numeric(mid["QED"], errors="coerce").fillna(0)
        - 0.05 * pd.to_numeric(mid["SA"], errors="coerce").fillna(5)
        + 0.2 * pd.to_numeric(mid["max_tanimoto_to_original_mrsa"], errors="coerce").fillna(0)
    )

    selected = mid.sort_values("selection_score", ascending=False).head(args.pair_examples).copy()
    selected.to_csv(outdir / "experiment_9_representative_rcfr_pairs.csv", index=False, encoding="utf-8-sig")

    mols = []
    legends = []

    for i, row in selected.iterrows():
        gen_mol = clean_mol_from_smiles(row["smiles"])
        ref_mol = clean_mol_from_smiles(row["nearest_reference_smiles"])

        if gen_mol is None or ref_mol is None:
            continue

        sim = row["max_tanimoto_to_original_mrsa"]
        prob = row.get("activity_label_probability", np.nan)
        mic = row.get("pred_pmic_ug_ml", np.nan)

        mols.append(gen_mol)
        legends.append(f"RCFR generated\nT={sim:.3f}, P={prob:.3f}, MIC={mic:.3g}")

        mols.append(ref_mol)
        legends.append(f"Nearest MRSA reference\nT={sim:.3f}")

    if mols:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=2,
            subImgSize=(420, 280),
            legends=legends,
            useSVG=False,
        )
        img.save(str(outdir / "experiment_9_representative_rcfr_nearest_pairs.png"))

    return selected


def experiment_10_close_not_copy(tanimoto_detail, args, outdir):
    df = tanimoto_detail.copy()
    df["Tanimoto bin"] = pd.cut(
        df["max_tanimoto_to_original_mrsa"],
        bins=[-0.001, 0.3, 0.6, 1.0],
        labels=["Distant <0.3", "Moderate 0.3-0.6", "High >=0.6"],
    )

    rows = []

    for group in GENERATED_GROUPS:
        sub = df[df["Group"] == group].copy()
        total = len(sub)

        for b in ["Distant <0.3", "Moderate 0.3-0.6", "High >=0.6"]:
            s = sub[sub["Tanimoto bin"] == b]
            rows.append({
                "Group": group,
                "Tanimoto bin": b,
                "Count": len(s),
                "Ratio": safe_ratio(len(s), total),
                "Activity probability mean ↑": float(pd.to_numeric(s["activity_label_probability"], errors="coerce").mean()) if len(s) else np.nan,
                "Predicted MIC median ↓": float(pd.to_numeric(s["pred_pmic_ug_ml"], errors="coerce").median()) if len(s) else np.nan,
                "QED mean ↑": float(pd.to_numeric(s["QED"], errors="coerce").mean()) if len(s) else np.nan,
                "SA mean ↓": float(pd.to_numeric(s["SA"], errors="coerce").mean()) if len(s) else np.nan,
            })

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "experiment_10_close_not_copy_bins.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    for group in GENERATED_GROUPS:
        sub = df[df["Group"] == group]
        axes[0].scatter(
            sub["max_tanimoto_to_original_mrsa"],
            sub["activity_label_probability"],
            s=25,
            alpha=0.65,
            color=GROUP_COLORS[group],
            label=group,
        )
        axes[1].scatter(
            sub["max_tanimoto_to_original_mrsa"],
            sub["pred_pmic_ug_ml"],
            s=25,
            alpha=0.65,
            color=GROUP_COLORS[group],
            label=group,
        )

    axes[0].set_title("Similarity vs activity probability")
    axes[0].set_xlabel("Max Tanimoto to original MRSA dataset")
    axes[0].set_ylabel("Activity probability")
    axes[0].grid(True, linestyle="--", alpha=0.25)
    axes[0].axvspan(0.3, 0.6, alpha=0.12)

    axes[1].set_title("Similarity vs predicted MIC")
    axes[1].set_xlabel("Max Tanimoto to original MRSA dataset")
    axes[1].set_ylabel("Predicted MIC, μg/mL")
    axes[1].grid(True, linestyle="--", alpha=0.25)
    axes[1].axvspan(0.3, 0.6, alpha=0.12)

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, outdir / "experiment_10_close_not_copy_scatter")
    plt.close(fig)

    pivot = table.pivot(index="Group", columns="Tanimoto bin", values="Ratio").reindex(GENERATED_GROUPS)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bottom = np.zeros(len(GENERATED_GROUPS))

    for b in ["Distant <0.3", "Moderate 0.3-0.6", "High >=0.6"]:
        vals = pivot[b].fillna(0).values
        ax.bar(GENERATED_GROUPS, vals, bottom=bottom, label=b)
        bottom += vals

    ax.set_title("Tanimoto similarity bin composition")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, outdir / "experiment_10_tanimoto_bin_composition")
    plt.close(fig)

    return table


def read_sdf_molecules(path: Path, top_n: int = 0) -> List[Chem.Mol]:
    if not path.exists():
        return []

    mols = []
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


def uff_metrics_for_mol(mol: Chem.Mol) -> Optional[Dict[str, float]]:
    if mol.GetNumConformers() == 0:
        return None

    try:
        mol_h = Chem.AddHs(mol, addCoords=True)
    except Exception:
        mol_h = Chem.Mol(mol)

    if mol_h.GetNumConformers() == 0:
        return None

    try:
        conf = mol_h.GetConformer()
        pos = conf.GetPositions()
        finite = int(np.isfinite(pos).all())
    except Exception:
        finite = 0

    try:
        props = AllChem.UFFGetMoleculeForceField(mol_h)
        if props is None:
            return {
                "has_3d_conformer": 1,
                "finite_coordinates": finite,
                "uff_available": 0,
                "uff_optimization_success": 0,
                "uff_energy_before": np.nan,
                "uff_energy_after": np.nan,
                "relaxation_rmsd": np.nan,
            }

        before = float(props.CalcEnergy())
        opt_mol = Chem.Mol(mol_h)
        result = AllChem.UFFOptimizeMolecule(opt_mol, maxIters=200)
        props_after = AllChem.UFFGetMoleculeForceField(opt_mol)
        after = float(props_after.CalcEnergy()) if props_after is not None else np.nan

        try:
            rmsd = float(AllChem.GetBestRMS(mol_h, opt_mol))
        except Exception:
            rmsd = np.nan

        return {
            "has_3d_conformer": 1,
            "finite_coordinates": finite,
            "uff_available": 1,
            "uff_optimization_success": int(result == 0),
            "uff_energy_before": before,
            "uff_energy_after": after,
            "relaxation_rmsd": rmsd,
        }
    except Exception:
        return {
            "has_3d_conformer": 1,
            "finite_coordinates": finite,
            "uff_available": 0,
            "uff_optimization_success": 0,
            "uff_energy_before": np.nan,
            "uff_energy_after": np.nan,
            "relaxation_rmsd": np.nan,
        }


def experiment_11_3d_quality(args, outdir):
    sdf_paths = {
        "General-Model Generated": Path(args.original_sdf),
        "MRSA-Trained Generated": Path(args.scratch_sdf),
        "RCFR-Diffusion for MRSA": Path(args.rcfr_sdf),
    }

    detail_rows = []
    summary_rows = []

    for group, path in sdf_paths.items():
        if not path.exists():
            summary_rows.append({
                "Group": group,
                "Status": "Skipped",
                "Reason": f"SDF not found: {path}",
            })
            continue

        mols = read_sdf_molecules(path, top_n=args.sdf_top_n)
        rows = []

        for i, mol in enumerate(mols):
            m = uff_metrics_for_mol(mol)
            if m is None:
                m = {
                    "has_3d_conformer": 0,
                    "finite_coordinates": 0,
                    "uff_available": 0,
                    "uff_optimization_success": 0,
                    "uff_energy_before": np.nan,
                    "uff_energy_after": np.nan,
                    "relaxation_rmsd": np.nan,
                }

            m["Group"] = group
            m["Molecule index"] = i
            rows.append(m)
            detail_rows.append(m)

        df = pd.DataFrame(rows)

        if df.empty:
            summary_rows.append({
                "Group": group,
                "Status": "No valid SDF molecules",
                "Reason": str(path),
            })
            continue

        summary_rows.append({
            "Group": group,
            "Status": "Done",
            "SDF molecules": len(df),
            "3D conformer ratio ↑": float(df["has_3d_conformer"].mean()),
            "Finite coordinate ratio ↑": float(df["finite_coordinates"].mean()),
            "UFF available ratio ↑": float(df["uff_available"].mean()),
            "UFF optimization success ratio ↑": float(df["uff_optimization_success"].mean()),
            "UFF energy before mean ↓": float(pd.to_numeric(df["uff_energy_before"], errors="coerce").mean()),
            "UFF energy after mean ↓": float(pd.to_numeric(df["uff_energy_after"], errors="coerce").mean()),
            "Relaxation RMSD mean ↓": float(pd.to_numeric(df["relaxation_rmsd"], errors="coerce").mean()),
        })

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)

    detail.to_csv(outdir / "experiment_11_3d_quality_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "experiment_11_3d_quality_summary.csv", index=False, encoding="utf-8-sig")

    done = summary[summary["Status"] == "Done"].copy()

    if not done.empty:
        metrics = ["UFF optimization success ratio ↑", "UFF energy after mean ↓", "Relaxation RMSD mean ↓"]
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

        for ax, metric in zip(axes, metrics):
            vals = done.set_index("Group").reindex(GENERATED_GROUPS)[metric]
            ax.bar(vals.index, vals.values, color=[GROUP_COLORS[g] for g in vals.index])
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.tick_params(axis="x", rotation=20)

        fig.tight_layout()
        save_fig(fig, outdir / "experiment_11_3d_quality_summary")
        plt.close(fig)

    return summary


def build_final_summary(exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp10, exp11, outdir):
    final = pd.DataFrame({"Group": GENERATED_GROUPS})

    parts = [
        exp1,
        exp2[["Group", "Drug-like pass ratio ↑"]] if not exp2.empty else None,
        exp3[["Group", "Activity probability mean ↑", "Active ratio ↑", "Predicted MIC median ↓", "Total score mean ↑"]] if not exp3.empty else None,
        exp5[["Group", "Mean quantile distance ↓", "Mean Wasserstein distance ↓", "Mean KS statistic ↓", "Descriptor Frechet distance ↓"]] if not exp5.empty else None,
        exp6[["Group", "Mean max Tanimoto ↑", "Median max Tanimoto ↑", "Tanimoto >= 0.4 ratio ↑"]] if not exp6.empty else None,
        exp7 if exp7 is not None and not exp7.empty else None,
        exp8[["Group", "Scaffold set overlap ratio ↑", "Molecule scaffold coverage ↑", "Key fragment group coverage ↑", "Mean key fragment coverage ↑"]] if not exp8.empty else None,
    ]

    for p in parts:
        if p is not None:
            final = final.merge(p, on="Group", how="left")

    final.to_csv(outdir / "final_all_experiments_summary.csv", index=False, encoding="utf-8-sig")

    rank_rules = {
        "Validity ↑": "higher",
        "Diversity ↑": "higher",
        "Drug-like pass ratio ↑": "higher",
        "Activity probability mean ↑": "higher",
        "Predicted MIC median ↓": "lower",
        "Mean quantile distance ↓": "lower",
        "Mean max Tanimoto ↑": "higher",
        "PCA mean nearest distance to MRSA ↓": "lower",
        "Scaffold set overlap ratio ↑": "higher",
        "Key fragment group coverage ↑": "higher",
    }

    rank_df = final[["Group"]].copy()
    rank_cols = []

    for metric, direction in rank_rules.items():
        if metric not in final.columns:
            continue
        vals = pd.to_numeric(final[metric], errors="coerce")
        if vals.notna().sum() < 2:
            continue
        ascending = direction == "lower"
        col = f"Rank - {metric}"
        rank_df[col] = vals.rank(method="min", ascending=ascending, na_option="bottom")
        rank_cols.append(col)

    if rank_cols:
        rank_df["Mean rank ↓"] = rank_df[rank_cols].mean(axis=1)
        rank_df["Total rank ↓"] = rank_df[rank_cols].sum(axis=1)

    rank_df.to_csv(outdir / "final_rank_summary.csv", index=False, encoding="utf-8-sig")

    if "Mean rank ↓" in rank_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        vals = rank_df.set_index("Group").loc[GENERATED_GROUPS, "Mean rank ↓"]
        ax.bar(vals.index, vals.values, color=[GROUP_COLORS[g] for g in vals.index])
        ax.set_title("Overall rank across selected experiments")
        ax.set_ylabel("Mean rank, lower is better")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(vals.values):
            ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
        fig.tight_layout()
        save_fig(fig, outdir / "final_overall_rank")
        plt.close(fig)

    manifest = {
        "experiments": [
            "experiment_1_generation_quality",
            "experiment_2_druglikeness",
            "experiment_3_activity_proxy",
            "experiment_4_topk_enrichment",
            "experiment_5_property_closeness",
            "experiment_6_tanimoto_nearest_neighbor",
            "experiment_7_chemical_space",
            "experiment_8_scaffold_fragment",
            "experiment_9_representative_pairs",
            "experiment_10_close_not_copy",
            "experiment_11_3d_quality",
        ],
        "main_tables": [
            "final_all_experiments_summary.csv",
            "final_rank_summary.csv",
            "experiment_1_generation_quality.csv",
            "experiment_2_druglikeness_summary.csv",
            "experiment_3_activity_proxy_summary.csv",
            "experiment_4_topk_enrichment.csv",
            "experiment_5_property_distance_summary.csv",
            "experiment_6_tanimoto_nearest_neighbor_summary.csv",
            "experiment_7_pca_chemical_space_distance_summary.csv",
            "experiment_8_scaffold_fragment_summary.csv",
            "experiment_10_close_not_copy_bins.csv",
            "experiment_11_3d_quality_summary.csv",
        ],
        "main_figures": [
            "experiment_1_generation_quality.png",
            "experiment_2_druglikeness_boxplots.png",
            "experiment_2_druglike_pass_ratio.png",
            "experiment_3_activity_proxy_distributions.png",
            "experiment_4_topk_enrichment.png",
            "experiment_5_property_distribution_overlap.png",
            "experiment_5_property_distance_heatmap.png",
            "experiment_6_tanimoto_similarity.png",
            "experiment_7_pca_chemical_space_panels.png",
            "experiment_7_pca_distance_bar.png",
            "experiment_8_scaffold_fragment_coverage.png",
            "experiment_9_representative_rcfr_nearest_pairs.png",
            "experiment_10_close_not_copy_scatter.png",
            "experiment_10_tanimoto_bin_composition.png",
            "experiment_11_3d_quality_summary.png",
            "final_overall_rank.png",
        ],
    }

    (outdir / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = """# Full MRSA generation evaluation experiments

This folder contains 11 experiments for evaluating MRSA molecule generation.

## Experiment list

1. Generation validity and diversity
2. Drug-likeness and synthetic accessibility
3. MRSA activity proxy evaluation
4. Top-K enrichment analysis
5. Property distribution closeness to original MRSA dataset
6. Nearest-neighbor Tanimoto similarity
7. Chemical space visualization and distance analysis
8. Scaffold and key fragment coverage
9. Representative RCFR molecule and nearest MRSA reference pairs
10. Close-but-not-copy balance analysis
11. 3D conformation quality analysis

## Main evidence

Use `final_all_experiments_summary.csv` and `final_rank_summary.csv` as the overview tables.

The most important figures for showing that RCFR-Diffusion for MRSA is closer to the original MRSA molecular space are:

- `experiment_5_property_distribution_overlap.png`
- `experiment_5_property_distance_heatmap.png`
- `experiment_6_tanimoto_similarity.png`
- `experiment_7_pca_chemical_space_panels.png`
- `experiment_8_scaffold_fragment_coverage.png`
- `experiment_9_representative_rcfr_nearest_pairs.png`

All figures are also exported as SVG and PDF for further editing.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")

    return final, rank_df


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

    parser.add_argument("--outdir", default="outputs/full_mrsa_generation_experiments")

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
    parser.add_argument("--pair_examples", type=int, default=8)
    parser.add_argument("--pair_min_tanimoto", type=float, default=0.3)
    parser.add_argument("--pair_max_tanimoto", type=float, default=0.7)

    parser.add_argument("--max_reference_plot", type=int, default=3000)
    parser.add_argument("--max_pairs", type=int, default=30000)
    parser.add_argument("--sdf_top_n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    required = [
        args.reference_csv,
        args.original_ranked_csv,
        args.scratch_ranked_csv,
        args.rcfr_ranked_csv,
    ]

    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    reference, ranked, all_tables, summary_jsons = load_inputs(args)

    print("Loaded molecules:")
    print(f"  Known MRSA Actives: {len(reference)}")
    for g in GENERATED_GROUPS:
        print(f"  {g}: ranked={len(ranked[g])}, all_scored={len(all_tables[g])}")

    exp1 = experiment_1_generation_quality(reference, ranked, all_tables, summary_jsons, args, outdir)
    print("Done: experiment 1")

    exp2 = experiment_2_druglikeness(reference, ranked, args, outdir)
    print("Done: experiment 2")

    exp3 = experiment_3_activity_proxy(ranked, args, outdir)
    print("Done: experiment 3")

    exp4 = experiment_4_topk_enrichment(ranked, args, outdir)
    print("Done: experiment 4")

    exp5_detail, exp5_summary = experiment_5_property_closeness(reference, ranked, outdir)
    print("Done: experiment 5")

    exp6_detail, exp6_summary = experiment_6_tanimoto(reference, ranked, args, outdir)
    print("Done: experiment 6")

    exp7_summary = experiment_7_chemical_space(reference, ranked, args, outdir)
    print("Done: experiment 7")

    exp8_summary = experiment_8_scaffold_fragment(reference, ranked, args, outdir)
    print("Done: experiment 8")

    exp9 = experiment_9_representative_pairs(exp6_detail, args, outdir)
    print("Done: experiment 9")

    exp10 = experiment_10_close_not_copy(exp6_detail, args, outdir)
    print("Done: experiment 10")

    exp11 = experiment_11_3d_quality(args, outdir)
    print("Done: experiment 11")

    final, rank = build_final_summary(
        exp1=exp1,
        exp2=exp2,
        exp3=exp3,
        exp4=exp4,
        exp5=exp5_summary,
        exp6=exp6_summary,
        exp7=exp7_summary,
        exp8=exp8_summary,
        exp10=exp10,
        exp11=exp11,
        outdir=outdir,
    )

    print("\nFinal summary:")
    print(final.to_string(index=False))

    print("\nRank summary:")
    print(rank.to_string(index=False))

    print(f"\nSaved all outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
