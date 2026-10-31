#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


GROUPS = [
    "General-Model Generated",
    "MRSA-Trained Generated",
    "RCFR-Diffusion for MRSA",
]

GROUP_COLORS = {
    "Reference MRSA": "#1f77b4",
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
                def frag_score(f):
                    atoms = list(f.GetAtoms())
                    has_carbon = int(any(a.GetSymbol() == "C" for a in atoms))
                    heavy_atoms = sum(1 for a in atoms if a.GetAtomicNum() > 1)
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


def smiles_to_table(df: pd.DataFrame, smiles_col: str, group: str, top_n: int = 0) -> pd.DataFrame:
    if smiles_col not in df.columns:
        raise ValueError(f"Column {smiles_col} not found")

    if top_n > 0:
        df = df.head(top_n).copy()

    rows = []
    seen = set()

    for idx, smiles in enumerate(df[smiles_col].dropna().astype(str)):
        can = canonicalize_smiles(smiles, keep_largest_fragment=True)
        if can is None or can in seen:
            continue
        seen.add(can)

        mol = robust_mol_from_smiles(can, keep_largest_fragment=False)
        if mol is None:
            continue

        row = {
            "group": group,
            "source_index": idx,
            "smiles": can,
            "mol": mol,
        }
        row.update(mol_descriptors(mol))

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No valid molecules loaded for {group}")

    return out


def load_reference(reference_csv: Path, smiles_col: str, active_only: bool, activity_col: str) -> pd.DataFrame:
    df = pd.read_csv(reference_csv)

    if active_only and activity_col in df.columns:
        df = df[df[activity_col] == 1].copy()

    return smiles_to_table(df, smiles_col=smiles_col, group="Reference MRSA", top_n=0)


def load_generated(csv_path: Path, smiles_col: str, group: str, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return smiles_to_table(df, smiles_col=smiles_col, group=group, top_n=top_n)


def mol_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def fp_to_array(fp, n_bits: int = 2048) -> np.ndarray:
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
        s = Chem.MolToSmiles(scaf, canonical=True)
        return s if s else "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


def wasserstein_distance_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return float("nan")

    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        return float(abs(np.nanmean(a) - np.nanmean(b)))


def ks_stat_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return float("nan")

    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).statistic)
    except Exception:
        grid = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 256)
        ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
        cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
        return float(np.max(np.abs(ca - cb)))


def descriptor_closeness(reference: pd.DataFrame, generated_tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for group, gdf in generated_tables.items():
        for col in DESCRIPTOR_COLUMNS:
            ref_vals = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy()
            gen_vals = pd.to_numeric(gdf[col], errors="coerce").dropna().to_numpy()

            if len(ref_vals) == 0 or len(gen_vals) == 0:
                continue

            iqr = np.nanpercentile(ref_vals, 75) - np.nanpercentile(ref_vals, 25)
            std = np.nanstd(ref_vals)
            scale = iqr if iqr > 1e-8 else std
            if scale <= 1e-8:
                scale = 1.0

            wd = wasserstein_distance_safe(ref_vals, gen_vals)
            ks = ks_stat_safe(ref_vals, gen_vals)

            rows.append({
                "Model": group,
                "Descriptor": col,
                "Reference mean": float(np.nanmean(ref_vals)),
                "Generated mean": float(np.nanmean(gen_vals)),
                "Reference median": float(np.nanmedian(ref_vals)),
                "Generated median": float(np.nanmedian(gen_vals)),
                "Wasserstein distance": wd,
                "Normalized Wasserstein distance": float(wd / scale),
                "KS statistic": ks,
            })

    detail = pd.DataFrame(rows)

    summary = (
        detail.groupby("Model")
        .agg({
            "Normalized Wasserstein distance": "mean",
            "KS statistic": "mean",
        })
        .reset_index()
        .rename(columns={
            "Normalized Wasserstein distance": "Descriptor distribution distance ↓",
            "KS statistic": "Descriptor KS statistic ↓",
        })
    )

    return detail, summary


def fingerprint_closeness(reference: pd.DataFrame, generated_tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_fps = [mol_fp(m) for m in reference["mol"]]

    detail_rows = []
    summary_rows = []

    for group, gdf in generated_tables.items():
        sims = []

        for smiles, mol in zip(gdf["smiles"], gdf["mol"]):
            fp = mol_fp(mol)
            vals = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            max_sim = float(max(vals)) if vals else float("nan")
            sims.append(max_sim)
            detail_rows.append({
                "Model": group,
                "smiles": smiles,
                "max_tanimoto_to_reference": max_sim,
            })

        sims_arr = np.asarray(sims, dtype=float)
        sims_arr = sims_arr[np.isfinite(sims_arr)]

        summary_rows.append({
            "Model": group,
            "Mean max Tanimoto ↑": float(np.mean(sims_arr)) if len(sims_arr) else float("nan"),
            "Median max Tanimoto ↑": float(np.median(sims_arr)) if len(sims_arr) else float("nan"),
            "Tanimoto >= 0.3 ratio ↑": safe_ratio(float(np.sum(sims_arr >= 0.3)), len(sims_arr)),
            "Tanimoto >= 0.4 ratio ↑": safe_ratio(float(np.sum(sims_arr >= 0.4)), len(sims_arr)),
            "Tanimoto >= 0.5 ratio ↑": safe_ratio(float(np.sum(sims_arr >= 0.5)), len(sims_arr)),
        })

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def scaffold_and_fragment_closeness(reference: pd.DataFrame, generated_tables: Dict[str, pd.DataFrame], top_fragments: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ref_scaffolds = set(scaffold_smiles(m) for m in reference["mol"])
    ref_scaffolds_no_empty = set(x for x in ref_scaffolds if x != "NO_SCAFFOLD")

    ref_bit_counter = Counter()
    for mol in reference["mol"]:
        ref_bit_counter.update(sparse_morgan_bits(mol))

    top_bits = set(bit for bit, _ in ref_bit_counter.most_common(top_fragments))

    summary_rows = []
    scaffold_rows = []
    fragment_rows = []

    for group, gdf in generated_tables.items():
        gen_scaffolds = [scaffold_smiles(m) for m in gdf["mol"]]
        gen_scaffolds_no_empty = [s for s in gen_scaffolds if s != "NO_SCAFFOLD"]

        matched_scaffold = [
            int(s in ref_scaffolds_no_empty)
            for s in gen_scaffolds_no_empty
        ]

        scaffold_coverage = safe_ratio(sum(matched_scaffold), len(matched_scaffold))
        no_scaffold_ratio = safe_ratio(gen_scaffolds.count("NO_SCAFFOLD"), len(gen_scaffolds))

        gen_scaffold_set = set(gen_scaffolds_no_empty)
        scaffold_set_overlap = safe_ratio(
            len(gen_scaffold_set & ref_scaffolds_no_empty),
            len(gen_scaffold_set)
        )

        union_bits = set()
        per_mol_cov = []

        for smiles, mol, scaf in zip(gdf["smiles"], gdf["mol"], gen_scaffolds):
            bits = sparse_morgan_bits(mol)
            union_bits.update(bits)
            cov = safe_ratio(len(bits & top_bits), len(top_bits))
            per_mol_cov.append(cov)

            scaffold_rows.append({
                "Model": group,
                "smiles": smiles,
                "scaffold": scaf,
                "scaffold_in_reference": int(scaf in ref_scaffolds_no_empty),
            })

            fragment_rows.append({
                "Model": group,
                "smiles": smiles,
                "key_fragment_coverage_per_molecule": cov,
            })

        group_fragment_coverage = safe_ratio(len(union_bits & top_bits), len(top_bits))

        summary_rows.append({
            "Model": group,
            "Scaffold coverage ↑": scaffold_coverage,
            "Generated scaffold overlap ↑": scaffold_set_overlap,
            "No scaffold ratio ↓": no_scaffold_ratio,
            "Key fragment group coverage ↑": group_fragment_coverage,
            "Mean key fragment coverage ↑": float(np.nanmean(per_mol_cov)) if per_mol_cov else float("nan"),
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(scaffold_rows), pd.DataFrame(fragment_rows)


def mic_activity_closeness(reference_csv: Path, generated_csvs: Dict[str, Path], top_n: int, active_only: bool, activity_col: str) -> pd.DataFrame:
    ref = pd.read_csv(reference_csv)

    if active_only and activity_col in ref.columns:
        ref = ref[ref[activity_col] == 1].copy()

    if "pmic_ug_ml" not in ref.columns:
        return pd.DataFrame()

    ref_vals = pd.to_numeric(ref["pmic_ug_ml"], errors="coerce").dropna()
    ref_vals = ref_vals[ref_vals > 0]
    if len(ref_vals) == 0:
        return pd.DataFrame()

    ref_log = np.log10(ref_vals.to_numpy(dtype=float))

    rows = []
    for group, csv_path in generated_csvs.items():
        df = pd.read_csv(csv_path)
        if top_n > 0:
            df = df.head(top_n).copy()

        if "pred_log10_pmic" in df.columns:
            gen_log = pd.to_numeric(df["pred_log10_pmic"], errors="coerce").dropna().to_numpy(dtype=float)
        elif "pred_pmic_ug_ml" in df.columns:
            vals = pd.to_numeric(df["pred_pmic_ug_ml"], errors="coerce").dropna()
            vals = vals[vals > 0]
            gen_log = np.log10(vals.to_numpy(dtype=float))
        else:
            continue

        if len(gen_log) == 0:
            continue

        iqr = np.nanpercentile(ref_log, 75) - np.nanpercentile(ref_log, 25)
        scale = iqr if iqr > 1e-8 else np.nanstd(ref_log)
        if scale <= 1e-8:
            scale = 1.0

        wd = wasserstein_distance_safe(ref_log, gen_log)
        ks = ks_stat_safe(ref_log, gen_log)

        rows.append({
            "Model": group,
            "MIC log10 Wasserstein distance ↓": wd,
            "MIC normalized distance ↓": float(wd / scale),
            "MIC KS statistic ↓": ks,
            "Reference log10 MIC mean": float(np.nanmean(ref_log)),
            "Generated log10 MIC mean": float(np.nanmean(gen_log)),
        })

    return pd.DataFrame(rows)


def make_rank_summary(
    descriptor_summary: pd.DataFrame,
    fp_summary: pd.DataFrame,
    scaffold_summary: pd.DataFrame,
    mic_summary: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame({"Model": GROUPS})

    for part in [descriptor_summary, fp_summary, scaffold_summary, mic_summary]:
        if part is not None and not part.empty:
            df = df.merge(part, on="Model", how="left")

    metric_directions = {
        "Descriptor distribution distance ↓": "lower",
        "Descriptor KS statistic ↓": "lower",
        "Mean max Tanimoto ↑": "higher",
        "Median max Tanimoto ↑": "higher",
        "Tanimoto >= 0.4 ratio ↑": "higher",
        "Scaffold coverage ↑": "higher",
        "Generated scaffold overlap ↑": "higher",
        "Key fragment group coverage ↑": "higher",
        "Mean key fragment coverage ↑": "higher",
        "MIC normalized distance ↓": "lower",
        "MIC KS statistic ↓": "lower",
    }

    rank_cols = []
    for metric, direction in metric_directions.items():
        if metric not in df.columns:
            continue
        ascending = direction == "lower"
        rank_col = f"Rank - {metric}"
        df[rank_col] = df[metric].rank(method="min", ascending=ascending, na_option="bottom")
        rank_cols.append(rank_col)

    df["Total rank score ↓"] = df[rank_cols].sum(axis=1)
    df["Mean rank ↓"] = df[rank_cols].mean(axis=1)
    df = df.sort_values("Total rank score ↓", ascending=True).reset_index(drop=True)

    return df


def plot_descriptor_heatmap(detail: pd.DataFrame, outdir: Path) -> None:
    pivot = detail.pivot(index="Descriptor", columns="Model", values="Normalized Wasserstein distance")
    pivot = pivot.reindex(DESCRIPTOR_COLUMNS)
    pivot = pivot[GROUPS]

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")

    ax.set_xticks(np.arange(len(GROUPS)))
    ax.set_xticklabels(GROUPS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Descriptor distribution distance to reference")
    ax.set_xlabel("Generated model")
    ax.set_ylabel("Descriptor")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized Wasserstein distance, lower is better")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "descriptor_distance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bar(rank_summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    vals = rank_summary.set_index("Model").loc[GROUPS, "Mean rank ↓"]
    ax.bar(vals.index, vals.values)
    ax.set_ylabel("Mean rank, lower is better")
    ax.set_title("Overall database closeness ranking")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "overall_database_closeness_rank.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tanimoto_boxplot(fp_detail: pd.DataFrame, outdir: Path) -> None:
    data = []
    labels = []

    for group in GROUPS:
        vals = fp_detail.loc[fp_detail["Model"] == group, "max_tanimoto_to_reference"].dropna().to_numpy(dtype=float)
        data.append(vals)
        labels.append(group)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Maximum Tanimoto similarity to reference")
    ax.set_title("Nearest-neighbor structural similarity to MRSA database")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "max_tanimoto_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scaffold_fragment_bar(scaffold_summary: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        "Scaffold coverage ↑",
        "Key fragment group coverage ↑",
        "Mean key fragment coverage ↑",
    ]

    x = np.arange(len(GROUPS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    for i, metric in enumerate(metrics):
        vals = scaffold_summary.set_index("Model").loc[GROUPS, metric].to_numpy(dtype=float)
        ax.bar(x + (i - 1) * width, vals, width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ratio, higher is better")
    ax.set_title("Scaffold and key fragment consistency with MRSA database")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "scaffold_fragment_consistency_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_space(reference: pd.DataFrame, generated_tables: Dict[str, pd.DataFrame], outdir: Path, max_reference_plot: int, seed: int) -> None:
    try:
        from sklearn.decomposition import PCA
    except Exception:
        print("sklearn not found, skip PCA plot")
        return

    rng = np.random.default_rng(seed)

    ref_plot = reference.copy()
    if max_reference_plot > 0 and len(ref_plot) > max_reference_plot:
        ref_plot = ref_plot.sample(n=max_reference_plot, random_state=seed).copy()

    rows = []

    for _, row in ref_plot.iterrows():
        fp = mol_fp(row["mol"])
        arr = fp_to_array(fp)
        rows.append(("Reference MRSA", row["smiles"], arr))

    for group, gdf in generated_tables.items():
        for _, row in gdf.iterrows():
            fp = mol_fp(row["mol"])
            arr = fp_to_array(fp)
            rows.append((group, row["smiles"], arr))

    X = np.vstack([r[2] for r in rows])
    groups = [r[0] for r in rows]
    smiles = [r[1] for r in rows]

    coords = PCA(n_components=2, random_state=seed).fit_transform(X)

    coord_df = pd.DataFrame({
        "group": groups,
        "smiles": smiles,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
    })
    coord_df.to_csv(outdir / "chemical_space_pca_coordinates.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.6, 6.2))

    for group in ["Reference MRSA"] + GROUPS:
        sub = coord_df[coord_df["group"] == group]
        if group == "Reference MRSA":
            ax.scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.22, label=group, color=GROUP_COLORS[group])
        else:
            ax.scatter(sub["PC1"], sub["PC2"], s=24, alpha=0.75, label=group, color=GROUP_COLORS[group])

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Chemical space overlap with MRSA database")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "chemical_space_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mic_distribution(reference_csv: Path, generated_csvs: Dict[str, Path], outdir: Path, top_n: int, active_only: bool, activity_col: str) -> None:
    ref = pd.read_csv(reference_csv)

    if active_only and activity_col in ref.columns:
        ref = ref[ref[activity_col] == 1].copy()

    if "pmic_ug_ml" not in ref.columns:
        return

    ref_vals = pd.to_numeric(ref["pmic_ug_ml"], errors="coerce").dropna()
    ref_vals = ref_vals[ref_vals > 0]
    if len(ref_vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ref_log = np.log10(ref_vals.to_numpy(dtype=float))
    ax.hist(ref_log, bins=40, density=True, alpha=0.30, label="Reference MRSA")

    for group, csv_path in generated_csvs.items():
        df = pd.read_csv(csv_path)
        if top_n > 0:
            df = df.head(top_n).copy()

        if "pred_log10_pmic" in df.columns:
            vals = pd.to_numeric(df["pred_log10_pmic"], errors="coerce").dropna().to_numpy(dtype=float)
        elif "pred_pmic_ug_ml" in df.columns:
            vals_raw = pd.to_numeric(df["pred_pmic_ug_ml"], errors="coerce").dropna()
            vals_raw = vals_raw[vals_raw > 0]
            vals = np.log10(vals_raw.to_numpy(dtype=float))
        else:
            continue

        if len(vals) == 0:
            continue

        ax.hist(vals, bins=40, density=True, alpha=0.25, label=group)

    ax.set_xlabel("log10 MIC or predicted MIC")
    ax.set_ylabel("Density")
    ax.set_title("MIC distribution consistency")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "mic_distribution_consistency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze whether RCFR-Diffusion for MRSA molecules are closer to MRSA database")
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--original_csv", required=True)
    parser.add_argument("--scratch_csv", required=True)
    parser.add_argument("--finetune_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--reference_smiles_col", default="smiles")
    parser.add_argument("--generated_smiles_col", default="smiles")
    parser.add_argument("--top_n", type=int, default=200)
    parser.add_argument("--reference_active_only", action="store_true")
    parser.add_argument("--activity_col", default="activity_label")
    parser.add_argument("--top_fragments", type=int, default=100)
    parser.add_argument("--max_reference_plot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reference_csv = Path(args.reference_csv)
    generated_csvs = {
        "General-Model Generated": Path(args.original_csv),
        "MRSA-Trained Generated": Path(args.scratch_csv),
        "RCFR-Diffusion for MRSA": Path(args.finetune_csv),
    }

    reference = load_reference(
        reference_csv=reference_csv,
        smiles_col=args.reference_smiles_col,
        active_only=args.reference_active_only,
        activity_col=args.activity_col,
    )

    generated_tables = {
        group: load_generated(csv_path, args.generated_smiles_col, group, args.top_n)
        for group, csv_path in generated_csvs.items()
    }

    print(f"Reference valid molecules: {len(reference)}")
    for group, table in generated_tables.items():
        print(f"{group} valid molecules: {len(table)}")

    reference.drop(columns=["mol"]).to_csv(outdir / "reference_descriptor_table.csv", index=False, encoding="utf-8-sig")
    for group, table in generated_tables.items():
        safe_name = group.lower().replace(" ", "_").replace("-", "_")
        table.drop(columns=["mol"]).to_csv(outdir / f"{safe_name}_descriptor_table.csv", index=False, encoding="utf-8-sig")

    descriptor_detail, descriptor_summary = descriptor_closeness(reference, generated_tables)
    fp_detail, fp_summary = fingerprint_closeness(reference, generated_tables)
    scaffold_summary, scaffold_detail, fragment_detail = scaffold_and_fragment_closeness(
        reference,
        generated_tables,
        top_fragments=args.top_fragments,
    )
    mic_summary = mic_activity_closeness(
        reference_csv=reference_csv,
        generated_csvs=generated_csvs,
        top_n=args.top_n,
        active_only=args.reference_active_only,
        activity_col=args.activity_col,
    )

    rank_summary = make_rank_summary(
        descriptor_summary=descriptor_summary,
        fp_summary=fp_summary,
        scaffold_summary=scaffold_summary,
        mic_summary=mic_summary,
    )

    descriptor_detail.to_csv(outdir / "descriptor_closeness_detail.csv", index=False, encoding="utf-8-sig")
    descriptor_summary.to_csv(outdir / "descriptor_closeness_summary.csv", index=False, encoding="utf-8-sig")
    fp_detail.to_csv(outdir / "fingerprint_similarity_detail.csv", index=False, encoding="utf-8-sig")
    fp_summary.to_csv(outdir / "fingerprint_similarity_summary.csv", index=False, encoding="utf-8-sig")
    scaffold_summary.to_csv(outdir / "scaffold_fragment_summary.csv", index=False, encoding="utf-8-sig")
    scaffold_detail.to_csv(outdir / "scaffold_detail.csv", index=False, encoding="utf-8-sig")
    fragment_detail.to_csv(outdir / "fragment_detail.csv", index=False, encoding="utf-8-sig")
    if not mic_summary.empty:
        mic_summary.to_csv(outdir / "mic_activity_closeness_summary.csv", index=False, encoding="utf-8-sig")

    rank_summary.to_csv(outdir / "database_closeness_rank_summary.csv", index=False, encoding="utf-8-sig")

    all_summary = rank_summary.copy()
    all_summary.to_csv(outdir / "database_closeness_final_table.csv", index=False, encoding="utf-8-sig")

    plot_descriptor_heatmap(descriptor_detail, outdir)
    plot_summary_bar(rank_summary, outdir)
    plot_tanimoto_boxplot(fp_detail, outdir)
    plot_scaffold_fragment_bar(scaffold_summary, outdir)
    plot_pca_space(
        reference=reference,
        generated_tables=generated_tables,
        outdir=outdir,
        max_reference_plot=args.max_reference_plot,
        seed=args.seed,
    )
    plot_mic_distribution(
        reference_csv=reference_csv,
        generated_csvs=generated_csvs,
        outdir=outdir,
        top_n=args.top_n,
        active_only=args.reference_active_only,
        activity_col=args.activity_col,
    )

    evidence = {
        "reference_molecules": int(len(reference)),
        "generated_molecules": {group: int(len(table)) for group, table in generated_tables.items()},
        "best_by_total_rank": rank_summary.iloc[0]["Model"],
        "top_n_generated": args.top_n,
        "reference_active_only": bool(args.reference_active_only),
        "outputs": {
            "final_table": str((outdir / "database_closeness_final_table.csv").resolve()),
            "rank_summary": str((outdir / "database_closeness_rank_summary.csv").resolve()),
            "descriptor_heatmap": str((outdir / "descriptor_distance_heatmap.png").resolve()),
            "tanimoto_boxplot": str((outdir / "max_tanimoto_boxplot.png").resolve()),
            "scaffold_fragment_bar": str((outdir / "scaffold_fragment_consistency_bar.png").resolve()),
            "pca": str((outdir / "chemical_space_pca.png").resolve()),
            "mic_distribution": str((outdir / "mic_distribution_consistency.png").resolve()),
        },
    }

    (outdir / "evidence_summary.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nFinal rank summary:")
    print(rank_summary.to_string(index=False))
    print(f"\nSaved all outputs to: {outdir}")


if __name__ == "__main__":
    main()
