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

DESCRIPTOR_COLUMNS = [
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

PLOT_DESCRIPTOR_COLUMNS = [
    "MW",
    "LogP",
    "TPSA",
    "QED",
    "SA",
    "log10_MIC",
]


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

    import rdkit
    rdkit_base = Path(rdkit.__file__).resolve().parent
    candidates = [
        rdkit_base / "Contrib" / "SA_Score",
        rdkit_base.parent / "Contrib" / "SA_Score",
        Path(os.environ.get("CONDA_PREFIX", "")) / "share" / "RDKit" / "Contrib" / "SA_Score",
        Path(os.environ.get("CONDA_PREFIX", "")) / "Lib" / "site-packages" / "rdkit" / "Contrib" / "SA_Score",
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            sys.path.insert(0, str(candidate))
            try:
                import sascorer
                return sascorer
            except Exception:
                continue

    return None


SASCORER = load_sascorer()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return float(a) / float(b)


def robust_mol_from_smiles(smiles: str, keep_largest_fragment: bool = True) -> Optional[Chem.Mol]:
    if pd.isna(smiles):
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    except Exception:
        return None

    if mol is None:
        return None

    if keep_largest_fragment:
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if frags:
                def frag_score(frag):
                    atoms = list(frag.GetAtoms())
                    has_carbon = int(any(atom.GetSymbol() == "C" for atom in atoms))
                    heavy_atoms = sum(1 for atom in atoms if atom.GetAtomicNum() > 1)
                    total_atoms = len(atoms)
                    return has_carbon, heavy_atoms, total_atoms
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
        smiles_clean = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)
        mol_clean = Chem.MolFromSmiles(smiles_clean)
        return mol_clean
    except Exception:
        return mol


def canonicalize_smiles(smiles: str, keep_largest_fragment: bool = True) -> Optional[str]:
    mol = robust_mol_from_smiles(smiles, keep_largest_fragment=keep_largest_fragment)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_sa_score(mol: Chem.Mol) -> float:
    if SASCORER is None:
        return float("nan")
    try:
        return float(SASCORER.calculateScore(mol))
    except Exception:
        return float("nan")


def mol_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "QED": float(QED.qed(mol)),
        "SA": get_sa_score(mol),
        "HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "RotB": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "Rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "AromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "HeavyAtoms": float(mol.GetNumHeavyAtoms()),
    }


def mol_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def fp_to_numpy(fp, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def sparse_morgan_bits(mol: Chem.Mol, radius: int = 2) -> set:
    fp = AllChem.GetMorganFingerprint(mol, radius=radius)
    return set(fp.GetNonzeroElements().keys())


def scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return "NO_SCAFFOLD"
        smi = Chem.MolToSmiles(scaf, canonical=True)
        return smi if smi else "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


def average_pairwise_diversity(smiles_list: List[str], max_pairs: int = 30000, seed: int = 42) -> float:
    mols = []
    for smiles in smiles_list:
        mol = robust_mol_from_smiles(smiles, keep_largest_fragment=True)
        if mol is not None:
            mols.append(mol)

    fps = [mol_fp(m) for m in mols]
    n = len(fps)

    if n < 2:
        return float("nan")

    rng = np.random.default_rng(seed)

    if n * (n - 1) // 2 <= max_pairs:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = []
        for _ in range(max_pairs):
            i, j = rng.choice(n, size=2, replace=False)
            if i > j:
                i, j = j, i
            pairs.append((int(i), int(j)))

    distances = []
    for i, j in pairs:
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        distances.append(1.0 - float(sim))

    return float(np.mean(distances)) if distances else float("nan")


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_log10_mic(row: pd.Series, is_reference: bool) -> float:
    if is_reference:
        if "pmic_ug_ml" in row.index:
            val = pd.to_numeric(row["pmic_ug_ml"], errors="coerce")
            if pd.notna(val) and val > 0:
                return float(np.log10(float(val)))
        return float("nan")

    if "pred_log10_pmic" in row.index:
        val = pd.to_numeric(row["pred_log10_pmic"], errors="coerce")
        if pd.notna(val):
            return float(val)

    if "pred_pmic_ug_ml" in row.index:
        val = pd.to_numeric(row["pred_pmic_ug_ml"], errors="coerce")
        if pd.notna(val) and val > 0:
            return float(np.log10(float(val)))

    return float("nan")


def load_molecule_table(
    csv_path: Path,
    group_name: str,
    top_n: int,
    is_reference: bool,
    active_only: bool,
    activity_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if is_reference and active_only and activity_col in df.columns:
        df = df[df[activity_col] == 1].copy()

    if "smiles" not in df.columns:
        raise ValueError(f"Column smiles not found in {csv_path}")

    if top_n > 0 and not is_reference:
        df = df.head(top_n).copy()

    rows = []
    seen = set()

    for idx, raw in df.iterrows():
        smiles = raw["smiles"]
        can = canonicalize_smiles(smiles, keep_largest_fragment=True)

        if can is None or can in seen:
            continue

        seen.add(can)
        mol = robust_mol_from_smiles(can, keep_largest_fragment=False)

        if mol is None:
            continue

        row = {
            "Group": group_name,
            "source_index": int(idx),
            "smiles": can,
            "mol": mol,
            "log10_MIC": extract_log10_mic(raw, is_reference=is_reference),
        }

        row.update(mol_descriptors(mol))

        copy_cols = [
            "activity_label_probability",
            "activity_label",
            "activity_label_binary",
            "pred_pmic_ug_ml",
            "pred_log10_pmic",
            "mic_score",
            "total_score",
            "qed",
            "lipinski_score",
            "alert_score",
            "alert_count",
            "passed_filter",
        ]

        for col in copy_cols:
            if col in raw.index:
                row[col] = raw[col]

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError(f"No valid molecules loaded for {group_name}: {csv_path}")

    return out


def normalize_0_1(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    min_v = vals.min()
    max_v = vals.max()

    if pd.isna(min_v) or pd.isna(max_v) or abs(max_v - min_v) < 1e-12:
        return pd.Series([0.5] * len(vals), index=vals.index)

    norm = (vals - min_v) / (max_v - min_v)
    if not higher_is_better:
        norm = 1.0 - norm
    return norm


def quantile_distribution_distance(ref_vals: np.ndarray, gen_vals: np.ndarray, n_quantiles: int = 101) -> float:
    ref_vals = np.asarray(ref_vals, dtype=float)
    gen_vals = np.asarray(gen_vals, dtype=float)
    ref_vals = ref_vals[np.isfinite(ref_vals)]
    gen_vals = gen_vals[np.isfinite(gen_vals)]

    if len(ref_vals) == 0 or len(gen_vals) == 0:
        return float("nan")

    qs = np.linspace(0, 1, n_quantiles)
    ref_q = np.quantile(ref_vals, qs)
    gen_q = np.quantile(gen_vals, qs)

    iqr = np.percentile(ref_vals, 75) - np.percentile(ref_vals, 25)
    if iqr <= 1e-8:
        iqr = np.std(ref_vals)
    if iqr <= 1e-8:
        iqr = 1.0

    return float(np.mean(np.abs(ref_q - gen_q)) / iqr)


def load_all_inputs(args):
    reference = load_molecule_table(
        csv_path=Path(args.reference_csv),
        group_name="Known MRSA Actives",
        top_n=0,
        is_reference=True,
        active_only=args.reference_active_only,
        activity_col=args.activity_col,
    )

    ranked_tables = {
        "General-Model Generated": load_molecule_table(
            Path(args.original_ranked_csv),
            group_name="General-Model Generated",
            top_n=args.top_n,
            is_reference=False,
            active_only=False,
            activity_col=args.activity_col,
        ),
        "MRSA-Trained Generated": load_molecule_table(
            Path(args.scratch_ranked_csv),
            group_name="MRSA-Trained Generated",
            top_n=args.top_n,
            is_reference=False,
            active_only=False,
            activity_col=args.activity_col,
        ),
        "RCFR-Diffusion for MRSA": load_molecule_table(
            Path(args.rcfr_ranked_csv),
            group_name="RCFR-Diffusion for MRSA",
            top_n=args.top_n,
            is_reference=False,
            active_only=False,
            activity_col=args.activity_col,
        ),
    }

    all_scored_tables = {}

    all_paths = {
        "General-Model Generated": Path(args.original_all_scored_csv),
        "MRSA-Trained Generated": Path(args.scratch_all_scored_csv),
        "RCFR-Diffusion for MRSA": Path(args.rcfr_all_scored_csv),
    }

    for group, path in all_paths.items():
        if path.exists():
            all_scored_tables[group] = load_molecule_table(
                path,
                group_name=group,
                top_n=0,
                is_reference=False,
                active_only=False,
                activity_col=args.activity_col,
            )
        else:
            all_scored_tables[group] = ranked_tables[group].copy()

    summary_jsons = {
        "General-Model Generated": read_json(Path(args.original_summary_json)),
        "MRSA-Trained Generated": read_json(Path(args.scratch_summary_json)),
        "RCFR-Diffusion for MRSA": read_json(Path(args.rcfr_summary_json)),
    }

    return reference, ranked_tables, all_scored_tables, summary_jsons


def experiment_1_generation_quality(reference, ranked_tables, all_scored_tables, summary_jsons, args, outdir: Path):
    ref_smiles = set(reference["smiles"].tolist())
    rows = []

    for group in GENERATED_GROUPS:
        all_df = all_scored_tables[group]
        ranked_df = ranked_tables[group]
        summary = summary_jsons.get(group, {})

        smiles_all = all_df["smiles"].dropna().astype(str).tolist()
        unique_smiles = sorted(set(smiles_all))
        novel_count = sum(1 for s in unique_smiles if s not in ref_smiles)

        generated_objects = summary.get("generated_objects", args.sampled_count)
        valid_raw = summary.get("valid_raw_molecules", len(smiles_all))

        diversity = average_pairwise_diversity(unique_smiles, max_pairs=args.max_pairs, seed=args.seed)

        activity_col = "activity_label_probability"
        pred_mic_col = "pred_pmic_ug_ml"

        activity_mean = float(pd.to_numeric(ranked_df.get(activity_col, pd.Series(dtype=float)), errors="coerce").mean()) if activity_col in ranked_df else np.nan
        active_ratio = float((pd.to_numeric(ranked_df.get(activity_col, pd.Series(dtype=float)), errors="coerce") >= args.activity_threshold).mean()) if activity_col in ranked_df else np.nan
        median_mic = float(pd.to_numeric(ranked_df.get(pred_mic_col, pd.Series(dtype=float)), errors="coerce").median()) if pred_mic_col in ranked_df else np.nan

        rows.append({
            "Group": group,
            "Sampled molecules": int(generated_objects),
            "Valid raw molecules": int(valid_raw),
            "Scored molecules": int(len(smiles_all)),
            "Unique molecules": int(len(unique_smiles)),
            "Novel molecules": int(novel_count),
            "Validity ↑": safe_ratio(valid_raw, generated_objects),
            "Uniqueness ↑": safe_ratio(len(unique_smiles), len(smiles_all)),
            "Novelty ↑": safe_ratio(novel_count, len(unique_smiles)),
            "Diversity ↑": diversity,
            "Top activity probability mean ↑": activity_mean,
            "Top active ratio ↑": active_ratio,
            "Top predicted MIC median ↓": median_mic,
            "Top QED mean ↑": float(ranked_df["QED"].mean()) if "QED" in ranked_df else np.nan,
            "Top SA mean ↓": float(ranked_df["SA"].mean()) if "SA" in ranked_df else np.nan,
            "Top MW mean": float(ranked_df["MW"].mean()) if "MW" in ranked_df else np.nan,
            "Top LogP mean": float(ranked_df["LogP"].mean()) if "LogP" in ranked_df else np.nan,
            "Top TPSA mean": float(ranked_df["TPSA"].mean()) if "TPSA" in ranked_df else np.nan,
        })

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "experiment_1_generation_quality_summary.csv", index=False, encoding="utf-8-sig")

    plot_metrics = [
        ("Validity ↑", "Higher is better"),
        ("Diversity ↑", "Higher is better"),
        ("Top activity probability mean ↑", "Higher is better"),
        ("Top predicted MIC median ↓", "Lower is better"),
        ("Top QED mean ↑", "Higher is better"),
        ("Top SA mean ↓", "Lower is better"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (metric, note) in zip(axes, plot_metrics):
        vals = table.set_index("Group").loc[GENERATED_GROUPS, metric]
        ax.bar(vals.index, vals.values, color=[GROUP_COLORS[g] for g in vals.index])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.text(0.02, 0.93, note, transform=ax.transAxes, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(outdir / "experiment_1_generation_quality_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return table


def experiment_2_topk_enrichment(ranked_tables, args, outdir: Path):
    ks = [20, 50, 100, 200]
    rows = []

    for group in GENERATED_GROUPS:
        df = ranked_tables[group].copy()

        for k in ks:
            sub = df.head(k).copy()

            activity = pd.to_numeric(sub.get("activity_label_probability", pd.Series(dtype=float)), errors="coerce")
            mic = pd.to_numeric(sub.get("pred_pmic_ug_ml", pd.Series(dtype=float)), errors="coerce")
            qed = pd.to_numeric(sub.get("QED", pd.Series(dtype=float)), errors="coerce")
            sa = pd.to_numeric(sub.get("SA", pd.Series(dtype=float)), errors="coerce")
            mw = pd.to_numeric(sub.get("MW", pd.Series(dtype=float)), errors="coerce")
            logp = pd.to_numeric(sub.get("LogP", pd.Series(dtype=float)), errors="coerce")
            tpsa = pd.to_numeric(sub.get("TPSA", pd.Series(dtype=float)), errors="coerce")

            druglike = (
                (qed >= 0.40)
                & (sa <= 5.0)
                & (mw <= 600)
                & (logp <= 5)
                & (tpsa <= 160)
            )

            rows.append({
                "Group": group,
                "TopK": k,
                "Active ratio ↑": float((activity >= args.activity_threshold).mean()) if len(activity) else np.nan,
                "Mean activity probability ↑": float(activity.mean()) if len(activity) else np.nan,
                "Median predicted MIC ↓": float(mic.median()) if len(mic) else np.nan,
                "Drug-like ratio ↑": float(druglike.mean()) if len(druglike) else np.nan,
                "Mean QED ↑": float(qed.mean()) if len(qed) else np.nan,
                "Mean SA ↓": float(sa.mean()) if len(sa) else np.nan,
            })

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "experiment_2_topk_enrichment.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    metrics = [
        ("Active ratio ↑", "Higher is better"),
        ("Drug-like ratio ↑", "Higher is better"),
        ("Median predicted MIC ↓", "Lower is better"),
    ]

    for ax, (metric, note) in zip(axes, metrics):
        for group in GENERATED_GROUPS:
            sub = table[table["Group"] == group]
            ax.plot(sub["TopK"], sub[metric], marker="o", label=group, color=GROUP_COLORS[group])
        ax.set_title(metric)
        ax.set_xlabel("Top-K candidates")
        ax.set_ylabel(metric)
        ax.text(0.02, 0.92, note, transform=ax.transAxes, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "experiment_2_topk_enrichment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return table


def experiment_3_composite_generation_score(exp1_table, exp2_table, outdir: Path):
    top200 = exp2_table[exp2_table["TopK"] == 200].copy()
    df = exp1_table.merge(
        top200[["Group", "Active ratio ↑", "Drug-like ratio ↑", "Median predicted MIC ↓"]],
        on="Group",
        how="left",
        suffixes=("", " Top200"),
    )

    score = pd.DataFrame({"Group": df["Group"]})
    score["Validity score"] = normalize_0_1(df["Validity ↑"], higher_is_better=True)
    score["Diversity score"] = normalize_0_1(df["Diversity ↑"], higher_is_better=True)
    score["Activity score"] = normalize_0_1(df["Active ratio ↑"], higher_is_better=True)
    score["MIC score"] = normalize_0_1(df["Median predicted MIC ↓"], higher_is_better=False)
    score["Drug-like score"] = normalize_0_1(df["Drug-like ratio ↑"], higher_is_better=True)
    score["QED score"] = normalize_0_1(df["Top QED mean ↑"], higher_is_better=True)
    score["SA score"] = normalize_0_1(df["Top SA mean ↓"], higher_is_better=False)

    score["Composite generation quality score ↑"] = score[
        [
            "Validity score",
            "Diversity score",
            "Activity score",
            "MIC score",
            "Drug-like score",
            "QED score",
            "SA score",
        ]
    ].mean(axis=1)

    score = score.sort_values("Composite generation quality score ↑", ascending=False).reset_index(drop=True)
    score.to_csv(outdir / "experiment_3_composite_generation_quality_score.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    vals = score.set_index("Group").loc[GENERATED_GROUPS, "Composite generation quality score ↑"]
    ax.bar(vals.index, vals.values, color=[GROUP_COLORS[g] for g in vals.index])
    ax.set_title("Composite generation quality score")
    ax.set_ylabel("Normalized composite score, higher is better")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    for i, v in enumerate(vals.values):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "experiment_3_composite_generation_quality_score.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return score


def experiment_4_property_distribution(reference, ranked_tables, outdir: Path):
    all_tables = {"Known MRSA Actives": reference, **ranked_tables}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8))
    axes = axes.flatten()

    for ax, col in zip(axes, PLOT_DESCRIPTOR_COLUMNS):
        all_vals = []

        for group in GROUP_ORDER:
            vals = pd.to_numeric(all_tables[group][col], errors="coerce").dropna().to_numpy(dtype=float)
            all_vals.extend(vals.tolist())

        if not all_vals:
            ax.axis("off")
            continue

        min_v = np.nanpercentile(all_vals, 1)
        max_v = np.nanpercentile(all_vals, 99)

        if not np.isfinite(min_v) or not np.isfinite(max_v) or abs(max_v - min_v) < 1e-8:
            min_v = np.nanmin(all_vals)
            max_v = np.nanmax(all_vals)

        bins = np.linspace(min_v, max_v, 35)

        for group in GROUP_ORDER:
            vals = pd.to_numeric(all_tables[group][col], errors="coerce").dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals >= min_v) & (vals <= max_v)]
            if len(vals) == 0:
                continue

            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.1,
                color=GROUP_COLORS[group],
                label=group,
            )

        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=True,
        fontsize=10,
    )

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.87, wspace=0.23, hspace=0.33)
    fig.savefig(outdir / "experiment_4_property_distribution_overlap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def experiment_5_descriptor_distance(reference, ranked_tables, outdir: Path):
    all_tables = {"Known MRSA Actives": reference, **ranked_tables}

    rows = []
    for group in GROUP_ORDER:
        df = all_tables[group]

        for col in DESCRIPTOR_COLUMNS + ["log10_MIC"]:
            ref_vals = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy(dtype=float)
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)

            if len(ref_vals) == 0 or len(vals) == 0:
                continue

            qdist = quantile_distribution_distance(ref_vals, vals)

            ref_mean = float(np.nanmean(ref_vals))
            group_mean = float(np.nanmean(vals))
            ref_std = float(np.nanstd(ref_vals))
            if ref_std <= 1e-8:
                ref_std = 1.0

            mean_z = abs(group_mean - ref_mean) / ref_std

            rows.append({
                "Group": group,
                "Descriptor": col,
                "Quantile distribution distance ↓": qdist,
                "Mean absolute z-distance ↓": mean_z,
                "Reference mean": ref_mean,
                "Group mean": group_mean,
            })

    detail = pd.DataFrame(rows)
    detail.to_csv(outdir / "experiment_5_descriptor_distance_detail.csv", index=False, encoding="utf-8-sig")

    summary = (
        detail.groupby("Group")
        .agg({
            "Quantile distribution distance ↓": "mean",
            "Mean absolute z-distance ↓": "mean",
        })
        .reset_index()
        .rename(columns={
            "Quantile distribution distance ↓": "Mean descriptor distribution distance ↓",
            "Mean absolute z-distance ↓": "Mean descriptor z-distance ↓",
        })
    )

    summary.to_csv(outdir / "experiment_5_descriptor_distance_summary.csv", index=False, encoding="utf-8-sig")

    pivot = detail.pivot(index="Descriptor", columns="Group", values="Quantile distribution distance ↓")
    display_rows = ["MW", "LogP", "TPSA", "QED", "SA", "HBA", "HBD", "RotB", "Rings", "log10_MIC"]
    pivot = pivot.reindex(index=display_rows, columns=GROUP_ORDER)

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    data = pivot.to_numpy(dtype=float)

    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(display_rows)))
    ax.set_yticklabels(display_rows)
    ax.set_title("Descriptor distribution distance to original MRSA dataset")
    ax.set_xlabel("Group")
    ax.set_ylabel("Descriptor")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Quantile distribution distance, lower is better")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "experiment_5_descriptor_distance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return detail, summary


def experiment_6_tanimoto_similarity(reference, ranked_tables, args, outdir: Path):
    ref_fps = [mol_fp(m) for m in reference["mol"]]

    detail_rows = []
    summary_rows = []
    nearest_pairs = {}

    for group in GROUP_ORDER:
        if group == "Known MRSA Actives":
            df = reference.copy()
            if args.max_reference_self > 0 and len(df) > args.max_reference_self:
                df = df.sample(n=args.max_reference_self, random_state=args.seed).copy()
            fps = [mol_fp(m) for m in df["mol"]]

            sims = []
            for i, (smi, fp) in enumerate(zip(df["smiles"], fps)):
                vals = DataStructs.BulkTanimotoSimilarity(fp, fps)
                if len(vals) > i:
                    vals[i] = -1.0
                max_sim = float(max(vals)) if vals else float("nan")
                sims.append(max_sim)
                detail_rows.append({
                    "Group": group,
                    "smiles": smi,
                    "max_tanimoto_to_original_mrsa": max_sim,
                    "nearest_reference_smiles": "",
                })
        else:
            df = ranked_tables[group]
            sims = []
            pairs = []

            for smi, mol in zip(df["smiles"], df["mol"]):
                fp = mol_fp(mol)
                vals = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)

                if vals:
                    best_idx = int(np.argmax(vals))
                    max_sim = float(vals[best_idx])
                    nearest_smi = reference.iloc[best_idx]["smiles"]
                else:
                    max_sim = float("nan")
                    nearest_smi = ""

                sims.append(max_sim)
                pairs.append((smi, nearest_smi, max_sim))

                detail_rows.append({
                    "Group": group,
                    "smiles": smi,
                    "max_tanimoto_to_original_mrsa": max_sim,
                    "nearest_reference_smiles": nearest_smi,
                })

            nearest_pairs[group] = pairs

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

    detail.to_csv(outdir / "experiment_6_tanimoto_similarity_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "experiment_6_tanimoto_similarity_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2))

    box_data = []
    box_labels = []

    for group in GROUP_ORDER:
        vals = detail.loc[detail["Group"] == group, "max_tanimoto_to_original_mrsa"].dropna().to_numpy(dtype=float)
        if len(vals):
            box_data.append(vals)
            box_labels.append(group)

    axes[0].boxplot(box_data, labels=box_labels, showmeans=True)
    axes[0].set_title("Nearest-neighbor Tanimoto similarity")
    axes[0].set_ylabel("Max Tanimoto to original MRSA dataset")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)

    thresholds = np.linspace(0, 1, 101)
    for group in GROUP_ORDER:
        vals = detail.loc[detail["Group"] == group, "max_tanimoto_to_original_mrsa"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        survival = [(vals >= t).mean() for t in thresholds]
        axes[1].plot(thresholds, survival, label=group, color=GROUP_COLORS[group])

    axes[1].set_title("Similarity enrichment curve")
    axes[1].set_xlabel("Tanimoto threshold")
    axes[1].set_ylabel("Ratio with max similarity above threshold")
    axes[1].grid(True, linestyle="--", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "experiment_6_tanimoto_similarity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return detail, summary, nearest_pairs


def experiment_7_chemical_space_pca(reference, ranked_tables, args, outdir: Path):
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception:
        print("sklearn not available, skip PCA experiment")
        return pd.DataFrame()

    ref_plot = reference.copy()
    if args.max_reference_plot > 0 and len(ref_plot) > args.max_reference_plot:
        ref_plot = ref_plot.sample(n=args.max_reference_plot, random_state=args.seed).copy()

    records = []

    for _, row in ref_plot.iterrows():
        fp = fp_to_numpy(mol_fp(row["mol"]))
        desc = np.array([row[c] for c in ["MW", "LogP", "TPSA", "QED", "SA", "HBA", "HBD", "RotB", "Rings"]], dtype=np.float32)
        records.append(("Known MRSA Actives", row["smiles"], fp, desc))

    for group in GENERATED_GROUPS:
        for _, row in ranked_tables[group].iterrows():
            fp = fp_to_numpy(mol_fp(row["mol"]))
            desc = np.array([row[c] for c in ["MW", "LogP", "TPSA", "QED", "SA", "HBA", "HBD", "RotB", "Rings"]], dtype=np.float32)
            records.append((group, row["smiles"], fp, desc))

    fps = np.vstack([r[2] for r in records])
    descs = np.vstack([r[3] for r in records])
    descs = np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)

    descs_scaled = StandardScaler().fit_transform(descs)
    X = np.hstack([fps, descs_scaled])

    coords = PCA(n_components=2, random_state=args.seed).fit_transform(X)

    coord_df = pd.DataFrame({
        "Group": [r[0] for r in records],
        "smiles": [r[1] for r in records],
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
    })
    coord_df.to_csv(outdir / "experiment_7_chemical_space_pca_coordinates.csv", index=False, encoding="utf-8-sig")

    ref_coords = coord_df[coord_df["Group"] == "Known MRSA Actives"][["PC1", "PC2"]].to_numpy(dtype=float)
    ref_centroid = ref_coords.mean(axis=0)

    rows = []
    for group in GROUP_ORDER:
        group_coords = coord_df[coord_df["Group"] == group][["PC1", "PC2"]].to_numpy(dtype=float)
        if len(group_coords) == 0:
            continue

        centroid = group_coords.mean(axis=0)
        centroid_distance = float(np.linalg.norm(centroid - ref_centroid))

        nn_distances = []
        for point in group_coords:
            d = np.linalg.norm(ref_coords - point, axis=1)
            nn_distances.append(float(np.min(d)))

        rows.append({
            "Group": group,
            "PCA centroid distance to MRSA ↓": centroid_distance,
            "Mean nearest PCA distance to MRSA ↓": float(np.mean(nn_distances)),
            "Median nearest PCA distance to MRSA ↓": float(np.median(nn_distances)),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "experiment_7_chemical_space_pca_distance_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True, sharey=True)

    for ax, group in zip(axes, GENERATED_GROUPS):
        ref = coord_df[coord_df["Group"] == "Known MRSA Actives"]
        gen = coord_df[coord_df["Group"] == group]

        ax.scatter(ref["PC1"], ref["PC2"], s=8, alpha=0.18, color=GROUP_COLORS["Known MRSA Actives"], label="Known MRSA Actives")
        ax.scatter(gen["PC1"], gen["PC2"], s=28, alpha=0.78, color=GROUP_COLORS[group], label=group)

        ax.set_title(group)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle="--", alpha=0.22)
        ax.legend(fontsize=8)

    fig.suptitle("Chemical space proximity to original MRSA dataset", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "experiment_7_chemical_space_pca_panels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    vals = summary.set_index("Group").loc[GROUP_ORDER, "Mean nearest PCA distance to MRSA ↓"]
    ax.bar(vals.index, vals.values, color=[GROUP_COLORS[g] for g in vals.index])
    ax.set_title("Mean nearest distance in PCA chemical space")
    ax.set_ylabel("Mean nearest PCA distance, lower is better")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    for i, v in enumerate(vals.values):
        if np.isfinite(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "experiment_7_chemical_space_pca_distance_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return summary


def experiment_8_scaffold_fragment_and_pairs(reference, ranked_tables, nearest_pairs, args, outdir: Path):
    ref_scaffolds = set(scaffold_smiles(m) for m in reference["mol"])
    ref_scaffolds = set(s for s in ref_scaffolds if s != "NO_SCAFFOLD")

    ref_bit_counter = Counter()
    for mol in reference["mol"]:
        ref_bit_counter.update(sparse_morgan_bits(mol))

    top_bits = set(bit for bit, _ in ref_bit_counter.most_common(args.top_fragments))

    rows = []
    detail_rows = []

    for group in GROUP_ORDER:
        if group == "Known MRSA Actives":
            df = reference
        else:
            df = ranked_tables[group]

        scaffold_flags = []
        per_mol_frag_cov = []
        group_union_bits = set()

        for _, row in df.iterrows():
            mol = row["mol"]
            smi = row["smiles"]

            scaf = scaffold_smiles(mol)
            bits = sparse_morgan_bits(mol)
            group_union_bits.update(bits)

            scaffold_hit = int(scaf in ref_scaffolds) if group != "Known MRSA Actives" else 1
            frag_cov = safe_ratio(len(bits & top_bits), len(top_bits))

            scaffold_flags.append(scaffold_hit)
            per_mol_frag_cov.append(frag_cov)

            detail_rows.append({
                "Group": group,
                "smiles": smi,
                "scaffold": scaf,
                "scaffold_in_original_mrsa": scaffold_hit,
                "key_fragment_coverage": frag_cov,
            })

        if group == "Known MRSA Actives":
            scaffold_coverage = 1.0
            group_fragment_coverage = 1.0
            mean_fragment_coverage = 1.0
        else:
            scaffold_coverage = safe_ratio(sum(scaffold_flags), len(scaffold_flags))
            group_fragment_coverage = safe_ratio(len(group_union_bits & top_bits), len(top_bits))
            mean_fragment_coverage = float(np.nanmean(per_mol_frag_cov)) if per_mol_frag_cov else np.nan

        rows.append({
            "Group": group,
            "Scaffold coverage ↑": scaffold_coverage,
            "Key fragment group coverage ↑": group_fragment_coverage,
            "Mean key fragment coverage ↑": mean_fragment_coverage,
        })

    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)

    summary.to_csv(outdir / "experiment_8_scaffold_fragment_summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(outdir / "experiment_8_scaffold_fragment_detail.csv", index=False, encoding="utf-8-sig")

    metrics = [
        "Scaffold coverage ↑",
        "Key fragment group coverage ↑",
        "Mean key fragment coverage ↑",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

    for ax, metric in zip(axes, metrics):
        vals = summary.set_index("Group").loc[GROUP_ORDER, metric]
        ax.bar(vals.index, vals.values, color=[GROUP_COLORS[g] for g in vals.index])
        ax.set_title(metric)
        ax.set_ylabel("Ratio, higher is better")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

        for i, v in enumerate(vals.values):
            if np.isfinite(v):
                ax.text(i, min(v + 0.025, 1.02), f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "experiment_8_scaffold_fragment_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    make_rcfr_nearest_reference_pair_image(nearest_pairs, outdir, max_pairs=args.pair_examples)

    return summary, detail


def make_rcfr_nearest_reference_pair_image(nearest_pairs, outdir: Path, max_pairs: int = 8):
    pairs = nearest_pairs.get("RCFR-Diffusion for MRSA", [])
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:max_pairs]

    mols = []
    legends = []

    for idx, (gen_smi, ref_smi, sim) in enumerate(pairs, start=1):
        gen_mol = robust_mol_from_smiles(gen_smi, keep_largest_fragment=True)
        ref_mol = robust_mol_from_smiles(ref_smi, keep_largest_fragment=True)

        if gen_mol is None or ref_mol is None:
            continue

        mols.append(gen_mol)
        legends.append(f"RCFR #{idx}\nTanimoto={sim:.3f}")

        mols.append(ref_mol)
        legends.append(f"Nearest MRSA reference\nTanimoto={sim:.3f}")

    if not mols:
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=2,
        subImgSize=(380, 260),
        legends=legends,
        useSVG=False,
    )

    img.save(str(outdir / "experiment_8_rcfr_nearest_reference_pairs.png"))


def build_final_ranking(
    exp1_table,
    exp3_table,
    exp5_summary,
    exp6_summary,
    exp7_summary,
    exp8_summary,
    outdir: Path,
):
    df = pd.DataFrame({"Group": GROUP_ORDER})

    for part in [exp1_table, exp3_table, exp5_summary, exp6_summary, exp7_summary, exp8_summary]:
        if part is not None and not part.empty:
            df = df.merge(part, on="Group", how="left")

    metric_rules = {
        "Validity ↑": "higher",
        "Diversity ↑": "higher",
        "Top activity probability mean ↑": "higher",
        "Top predicted MIC median ↓": "lower",
        "Composite generation quality score ↑": "higher",
        "Mean descriptor distribution distance ↓": "lower",
        "Mean descriptor z-distance ↓": "lower",
        "Mean max Tanimoto ↑": "higher",
        "Median max Tanimoto ↑": "higher",
        "Tanimoto >= 0.4 ratio ↑": "higher",
        "PCA centroid distance to MRSA ↓": "lower",
        "Mean nearest PCA distance to MRSA ↓": "lower",
        "Scaffold coverage ↑": "higher",
        "Key fragment group coverage ↑": "higher",
        "Mean key fragment coverage ↑": "higher",
    }

    rank_cols = []

    for metric, direction in metric_rules.items():
        if metric not in df.columns:
            continue

        tmp = df[metric]
        if tmp.notna().sum() <= 1:
            continue

        ascending = direction == "lower"
        rank_col = f"Rank - {metric}"
        df[rank_col] = tmp.rank(method="min", ascending=ascending, na_option="bottom")
        rank_cols.append(rank_col)

    df["Total rank score ↓"] = df[rank_cols].sum(axis=1)
    df["Mean rank ↓"] = df[rank_cols].mean(axis=1)

    generated_df = df[df["Group"].isin(GENERATED_GROUPS)].copy()

    generation_metrics = [
        "Validity ↑",
        "Diversity ↑",
        "Top activity probability mean ↑",
        "Top predicted MIC median ↓",
        "Composite generation quality score ↑",
    ]

    closeness_metrics = [
        "Mean descriptor distribution distance ↓",
        "Mean max Tanimoto ↑",
        "Median max Tanimoto ↑",
        "Tanimoto >= 0.4 ratio ↑",
        "Mean nearest PCA distance to MRSA ↓",
        "Scaffold coverage ↑",
        "Key fragment group coverage ↑",
        "Mean key fragment coverage ↑",
    ]

    support_rows = []

    for category, metrics in [
        ("Generation quality", generation_metrics),
        ("MRSA database closeness", closeness_metrics),
    ]:
        used = 0
        rcfr_best = 0
        best_items = []

        for metric in metrics:
            if metric not in generated_df.columns:
                continue

            vals = generated_df[["Group", metric]].dropna()
            if vals.empty:
                continue

            used += 1
            lower_is_better = "↓" in metric

            if lower_is_better:
                best_group = vals.sort_values(metric, ascending=True).iloc[0]["Group"]
            else:
                best_group = vals.sort_values(metric, ascending=False).iloc[0]["Group"]

            if best_group == "RCFR-Diffusion for MRSA":
                rcfr_best += 1

            best_items.append(f"{metric}: {best_group}")

        support_rows.append({
            "Conclusion category": category,
            "Comparable metrics": used,
            "RCFR best metrics": rcfr_best,
            "RCFR best ratio": safe_ratio(rcfr_best, used),
            "Detail": "; ".join(best_items),
        })

    support_table = pd.DataFrame(support_rows)

    df.to_csv(outdir / "final_all_experiments_summary.csv", index=False, encoding="utf-8-sig")
    support_table.to_csv(outdir / "final_rcfr_support_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    vals = df.set_index("Group").loc[GROUP_ORDER, "Mean rank ↓"]
    ax.bar(vals.index, vals.values, color=[GROUP_COLORS[g] for g in vals.index])
    ax.set_title("Overall ranking across all experiments")
    ax.set_ylabel("Mean rank, lower is better")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    for i, v in enumerate(vals.values):
        if np.isfinite(v):
            ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "final_overall_experiment_ranking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return df, support_table


def write_readme(outdir: Path, support_table: pd.DataFrame):
    text = """# RCFR-Diffusion for MRSA generation and database closeness experiments

This folder contains eight experiments for evaluating generated MRSA candidate molecules.

## Experiment list

1. Generation quality summary
2. Top-K activity and drug-likeness enrichment
3. Composite generation quality score
4. Four-group property distribution overlap
5. Descriptor distribution distance to original MRSA dataset
6. Nearest-neighbor Tanimoto similarity to original MRSA dataset
7. Chemical space PCA proximity to original MRSA dataset
8. Scaffold, key fragment, and nearest reference molecule comparison

## Key interpretation

Use `final_rcfr_support_summary.csv` and `final_all_experiments_summary.csv` as the main evidence tables.

If RCFR-Diffusion for MRSA has lower descriptor distribution distance, higher Tanimoto similarity, lower PCA distance, and higher scaffold/fragment coverage than General-Model Generated and MRSA-Trained Generated, it supports the conclusion that RCFR-Diffusion for MRSA is closer to the original MRSA molecular space.

"""

    text += "\n## Automatic support summary\n\n"
    text += support_table.to_markdown(index=False)

    (outdir / "README_experiment_interpretation.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run RCFR generation quality and MRSA molecular-space closeness experiments")
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

    parser.add_argument("--outdir", default="outputs/rcfr_generation_and_closeness_experiments")
    parser.add_argument("--top_n", type=int, default=200)
    parser.add_argument("--sampled_count", type=int, default=2000)
    parser.add_argument("--activity_threshold", type=float, default=0.5)
    parser.add_argument("--reference_active_only", action="store_true")
    parser.add_argument("--activity_col", default="activity_label")

    parser.add_argument("--top_fragments", type=int, default=100)
    parser.add_argument("--max_reference_self", type=int, default=1000)
    parser.add_argument("--max_reference_plot", type=int, default=3000)
    parser.add_argument("--pair_examples", type=int, default=8)
    parser.add_argument("--max_pairs", type=int, default=30000)
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

    reference, ranked_tables, all_scored_tables, summary_jsons = load_all_inputs(args)

    print("Loaded molecule counts:")
    print(f"  Known MRSA Actives: {len(reference)}")
    for group, table in ranked_tables.items():
        print(f"  {group}: {len(table)} top molecules")
    for group, table in all_scored_tables.items():
        print(f"  {group}: {len(table)} all-scored molecules")

    exp1 = experiment_1_generation_quality(reference, ranked_tables, all_scored_tables, summary_jsons, args, outdir)
    exp2 = experiment_2_topk_enrichment(ranked_tables, args, outdir)
    exp3 = experiment_3_composite_generation_score(exp1, exp2, outdir)

    experiment_4_property_distribution(reference, ranked_tables, outdir)
    exp5_detail, exp5_summary = experiment_5_descriptor_distance(reference, ranked_tables, outdir)
    exp6_detail, exp6_summary, nearest_pairs = experiment_6_tanimoto_similarity(reference, ranked_tables, args, outdir)
    exp7_summary = experiment_7_chemical_space_pca(reference, ranked_tables, args, outdir)
    exp8_summary, exp8_detail = experiment_8_scaffold_fragment_and_pairs(reference, ranked_tables, nearest_pairs, args, outdir)

    final_summary, support_table = build_final_ranking(
        exp1_table=exp1,
        exp3_table=exp3,
        exp5_summary=exp5_summary,
        exp6_summary=exp6_summary,
        exp7_summary=exp7_summary,
        exp8_summary=exp8_summary,
        outdir=outdir,
    )

    write_readme(outdir, support_table)

    evidence = {
        "outdir": str(outdir.resolve()),
        "top_n": args.top_n,
        "reference_active_only": bool(args.reference_active_only),
        "core_figures": [
            "experiment_1_generation_quality_summary.png",
            "experiment_2_topk_enrichment.png",
            "experiment_3_composite_generation_quality_score.png",
            "experiment_4_property_distribution_overlap.png",
            "experiment_5_descriptor_distance_heatmap.png",
            "experiment_6_tanimoto_similarity.png",
            "experiment_7_chemical_space_pca_panels.png",
            "experiment_7_chemical_space_pca_distance_bar.png",
            "experiment_8_scaffold_fragment_coverage.png",
            "experiment_8_rcfr_nearest_reference_pairs.png",
            "final_overall_experiment_ranking.png",
        ],
        "core_tables": [
            "experiment_1_generation_quality_summary.csv",
            "experiment_2_topk_enrichment.csv",
            "experiment_3_composite_generation_quality_score.csv",
            "experiment_5_descriptor_distance_summary.csv",
            "experiment_6_tanimoto_similarity_summary.csv",
            "experiment_7_chemical_space_pca_distance_summary.csv",
            "experiment_8_scaffold_fragment_summary.csv",
            "final_all_experiments_summary.csv",
            "final_rcfr_support_summary.csv",
        ],
    }

    (outdir / "evidence_manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFinal support summary:")
    print(support_table.to_string(index=False))

    print("\nFinal all-experiment summary:")
    print(final_summary.to_string(index=False))

    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
