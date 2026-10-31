#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
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
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolDescriptors
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


def extract_log10_mic(row: pd.Series, is_reference: bool) -> float:
    if is_reference:
        if "pmic_ug_ml" in row.index:
            val = pd.to_numeric(row["pmic_ug_ml"], errors="coerce")
            if pd.notna(val) and val > 0:
                return float(np.log10(val))
        return float("nan")

    if "pred_log10_pmic" in row.index:
        val = pd.to_numeric(row["pred_log10_pmic"], errors="coerce")
        if pd.notna(val):
            return float(val)

    if "pred_pmic_ug_ml" in row.index:
        val = pd.to_numeric(row["pred_pmic_ug_ml"], errors="coerce")
        if pd.notna(val) and val > 0:
            return float(np.log10(val))

    return float("nan")


def load_molecule_table(
    csv_path: Path,
    smiles_col: str,
    group_name: str,
    top_n: int,
    is_reference: bool,
    active_only: bool,
    activity_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if is_reference and active_only and activity_col in df.columns:
        df = df[df[activity_col] == 1].copy()

    if smiles_col not in df.columns:
        raise ValueError(f"Column {smiles_col} not found in {csv_path}")

    if top_n > 0 and not is_reference:
        df = df.head(top_n).copy()

    rows = []
    seen = set()

    for idx, row_raw in df.iterrows():
        smiles = row_raw[smiles_col]
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
            "log10_MIC": extract_log10_mic(row_raw, is_reference=is_reference),
        }
        row.update(mol_descriptors(mol))
        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError(f"No valid molecules loaded for {group_name} from {csv_path}")

    return out


def mol_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


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


def compute_descriptor_stats(all_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for group, df in all_tables.items():
        row = {
            "Group": group,
            "Molecule count": int(len(df)),
        }
        for col in DESCRIPTOR_COLUMNS + ["log10_MIC"]:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            row[f"{col} mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col} median"] = float(vals.median()) if len(vals) else float("nan")
            row[f"{col} std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def compute_descriptor_distance(reference: pd.DataFrame, all_tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []

    for group, df in all_tables.items():
        for col in DESCRIPTOR_COLUMNS + ["log10_MIC"]:
            ref_vals = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy(dtype=float)
            gen_vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)

            if len(ref_vals) == 0 or len(gen_vals) == 0:
                continue

            ref_mean = float(np.nanmean(ref_vals))
            gen_mean = float(np.nanmean(gen_vals))
            ref_std = float(np.nanstd(ref_vals))

            if ref_std <= 1e-8:
                ref_std = 1.0

            mean_z_distance = abs(gen_mean - ref_mean) / ref_std

            detail_rows.append({
                "Group": group,
                "Descriptor": col,
                "Reference mean": ref_mean,
                "Group mean": gen_mean,
                "Mean absolute z-distance ↓": float(mean_z_distance),
            })

    detail = pd.DataFrame(detail_rows)

    summary = (
        detail.groupby("Group")["Mean absolute z-distance ↓"]
        .mean()
        .reset_index()
        .rename(columns={"Mean absolute z-distance ↓": "Mean descriptor distance ↓"})
    )

    return detail, summary


def compute_tanimoto_similarity(reference: pd.DataFrame, all_tables: Dict[str, pd.DataFrame], max_reference_self: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reference_fps = [mol_fp(mol) for mol in reference["mol"]]

    detail_rows = []
    summary_rows = []

    for group, df in all_tables.items():
        sims = []

        if group == "Known MRSA Actives":
            ref_small = reference.copy()
            if max_reference_self > 0 and len(ref_small) > max_reference_self:
                ref_small = ref_small.sample(n=max_reference_self, random_state=seed).copy()

            small_fps = [mol_fp(mol) for mol in ref_small["mol"]]

            for i, (smiles, fp) in enumerate(zip(ref_small["smiles"], small_fps)):
                vals = DataStructs.BulkTanimotoSimilarity(fp, small_fps)
                if len(vals) > i:
                    vals[i] = -1.0
                max_sim = float(max(vals)) if vals else float("nan")
                sims.append(max_sim)
                detail_rows.append({
                    "Group": group,
                    "smiles": smiles,
                    "max_tanimoto_to_mrsa_database": max_sim,
                })
        else:
            for smiles, mol in zip(df["smiles"], df["mol"]):
                fp = mol_fp(mol)
                vals = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
                max_sim = float(max(vals)) if vals else float("nan")
                sims.append(max_sim)
                detail_rows.append({
                    "Group": group,
                    "smiles": smiles,
                    "max_tanimoto_to_mrsa_database": max_sim,
                })

        arr = np.asarray(sims, dtype=float)
        arr = arr[np.isfinite(arr)]

        summary_rows.append({
            "Group": group,
            "Mean max Tanimoto ↑": float(np.mean(arr)) if len(arr) else float("nan"),
            "Median max Tanimoto ↑": float(np.median(arr)) if len(arr) else float("nan"),
            "Tanimoto >= 0.3 ratio ↑": safe_ratio(float(np.sum(arr >= 0.3)), len(arr)),
            "Tanimoto >= 0.4 ratio ↑": safe_ratio(float(np.sum(arr >= 0.4)), len(arr)),
            "Tanimoto >= 0.5 ratio ↑": safe_ratio(float(np.sum(arr >= 0.5)), len(arr)),
        })

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def compute_scaffold_fragment(reference: pd.DataFrame, all_tables: Dict[str, pd.DataFrame], top_fragments: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_scaffolds = set(scaffold_smiles(mol) for mol in reference["mol"])
    ref_scaffolds = set(x for x in ref_scaffolds if x != "NO_SCAFFOLD")

    ref_bits = Counter()
    for mol in reference["mol"]:
        ref_bits.update(sparse_morgan_bits(mol))

    top_bits = set(bit for bit, _ in ref_bits.most_common(top_fragments))

    summary_rows = []
    detail_rows = []

    for group, df in all_tables.items():
        scaffold_flags = []
        fragment_coverages = []
        union_bits = set()

        for smiles, mol in zip(df["smiles"], df["mol"]):
            scaf = scaffold_smiles(mol)
            bits = sparse_morgan_bits(mol)
            union_bits.update(bits)

            scaffold_in_ref = int(scaf in ref_scaffolds)
            fragment_cov = safe_ratio(len(bits & top_bits), len(top_bits))

            scaffold_flags.append(scaffold_in_ref)
            fragment_coverages.append(fragment_cov)

            detail_rows.append({
                "Group": group,
                "smiles": smiles,
                "scaffold": scaf,
                "scaffold_in_mrsa_database": scaffold_in_ref,
                "key_fragment_coverage": fragment_cov,
            })

        if group == "Known MRSA Actives":
            scaffold_coverage = 1.0
            group_fragment_coverage = 1.0
            mean_fragment_coverage = 1.0
        else:
            scaffold_coverage = safe_ratio(sum(scaffold_flags), len(scaffold_flags))
            group_fragment_coverage = safe_ratio(len(union_bits & top_bits), len(top_bits))
            mean_fragment_coverage = float(np.nanmean(fragment_coverages)) if fragment_coverages else float("nan")

        summary_rows.append({
            "Group": group,
            "Scaffold coverage ↑": scaffold_coverage,
            "Key fragment group coverage ↑": group_fragment_coverage,
            "Mean key fragment coverage ↑": mean_fragment_coverage,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def build_final_summary(
    descriptor_summary: pd.DataFrame,
    tanimoto_summary: pd.DataFrame,
    scaffold_summary: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame({"Group": GROUP_ORDER})

    for part in [descriptor_summary, tanimoto_summary, scaffold_summary]:
        df = df.merge(part, on="Group", how="left")

    rank_rules = {
        "Mean descriptor distance ↓": "lower",
        "Mean max Tanimoto ↑": "higher",
        "Median max Tanimoto ↑": "higher",
        "Tanimoto >= 0.4 ratio ↑": "higher",
        "Scaffold coverage ↑": "higher",
        "Key fragment group coverage ↑": "higher",
        "Mean key fragment coverage ↑": "higher",
    }

    rank_cols = []
    for metric, direction in rank_rules.items():
        if metric not in df.columns:
            continue

        if df.loc[df["Group"] != "Known MRSA Actives", metric].notna().sum() == 0:
            continue

        rank_col = f"Rank - {metric}"
        ascending = direction == "lower"
        df[rank_col] = df[metric].rank(method="min", ascending=ascending, na_option="bottom")
        rank_cols.append(rank_col)

    df["Total rank score ↓"] = df[rank_cols].sum(axis=1)
    df["Mean rank ↓"] = df[rank_cols].mean(axis=1)

    return df


def plot_property_distributions(all_tables: Dict[str, pd.DataFrame], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.8))
    axes = axes.flatten()

    for ax, col in zip(axes, PLOT_DESCRIPTOR_COLUMNS):
        values_by_group = {}
        all_vals = []

        for group in GROUP_ORDER:
            vals = pd.to_numeric(all_tables[group][col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                values_by_group[group] = vals
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
            vals = values_by_group.get(group)
            if vals is None or len(vals) == 0:
                continue
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals >= min_v) & (vals <= max_v)]
            if len(vals) == 0:
                continue

            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                label=group,
                color=GROUP_COLORS[group],
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
    fig.savefig(outdir / "property_distribution_four_groups.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_descriptor_distance_heatmap(descriptor_detail: pd.DataFrame, outdir: Path) -> None:
    pivot = descriptor_detail.pivot(index="Descriptor", columns="Group", values="Mean absolute z-distance ↓")
    display_rows = ["MW", "LogP", "TPSA", "QED", "SA", "HBA", "HBD", "RotB", "Rings", "log10_MIC"]
    display_cols = GROUP_ORDER

    pivot = pivot.reindex(index=display_rows, columns=display_cols)

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    data = pivot.to_numpy(dtype=float)

    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(np.arange(len(display_cols)))
    ax.set_xticklabels(display_cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(display_rows)))
    ax.set_yticklabels(display_rows)

    ax.set_title("Descriptor mean distance to MRSA database")
    ax.set_xlabel("Group")
    ax.set_ylabel("Descriptor")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean absolute z-distance, lower is better")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "descriptor_distance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tanimoto_boxplot(tanimoto_detail: pd.DataFrame, outdir: Path) -> None:
    data = []
    labels = []

    for group in GROUP_ORDER:
        vals = pd.to_numeric(
            tanimoto_detail.loc[tanimoto_detail["Group"] == group, "max_tanimoto_to_mrsa_database"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        if len(vals) > 0:
            data.append(vals)
            labels.append(group)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Maximum Tanimoto similarity to MRSA database")
    ax.set_title("Nearest-neighbor structural similarity")
    ax.grid(axis="y", linestyle="--", alpha=0.28)

    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(outdir / "tanimoto_similarity_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_closeness_summary(summary: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        ("Mean descriptor distance ↓", "lower"),
        ("Mean max Tanimoto ↑", "higher"),
        ("Scaffold coverage ↑", "higher"),
        ("Key fragment group coverage ↑", "higher"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    axes = axes.flatten()

    for ax, (metric, direction) in zip(axes, metrics):
        vals = summary.set_index("Group").reindex(GROUP_ORDER)[metric]

        ax.bar(
            vals.index,
            vals.values,
            color=[GROUP_COLORS[g] for g in vals.index],
        )

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.tick_params(axis="x", rotation=20)

        if direction == "higher":
            ax.text(0.02, 0.94, "Higher is better", transform=ax.transAxes, fontsize=9)
        else:
            ax.text(0.02, 0.94, "Lower is better", transform=ax.transAxes, fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "database_closeness_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Four-group MRSA database closeness comparison")
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--original_csv", required=True)
    parser.add_argument("--scratch_csv", required=True)
    parser.add_argument("--rcfr_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--reference_smiles_col", default="smiles")
    parser.add_argument("--generated_smiles_col", default="smiles")
    parser.add_argument("--top_n", type=int, default=200)
    parser.add_argument("--reference_active_only", action="store_true")
    parser.add_argument("--activity_col", default="activity_label")
    parser.add_argument("--top_fragments", type=int, default=100)
    parser.add_argument("--max_reference_self", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reference = load_molecule_table(
        csv_path=Path(args.reference_csv),
        smiles_col=args.reference_smiles_col,
        group_name="Known MRSA Actives",
        top_n=0,
        is_reference=True,
        active_only=args.reference_active_only,
        activity_col=args.activity_col,
    )

    original = load_molecule_table(
        csv_path=Path(args.original_csv),
        smiles_col=args.generated_smiles_col,
        group_name="General-Model Generated",
        top_n=args.top_n,
        is_reference=False,
        active_only=False,
        activity_col=args.activity_col,
    )

    scratch = load_molecule_table(
        csv_path=Path(args.scratch_csv),
        smiles_col=args.generated_smiles_col,
        group_name="MRSA-Trained Generated",
        top_n=args.top_n,
        is_reference=False,
        active_only=False,
        activity_col=args.activity_col,
    )

    rcfr = load_molecule_table(
        csv_path=Path(args.rcfr_csv),
        smiles_col=args.generated_smiles_col,
        group_name="RCFR-Diffusion for MRSA",
        top_n=args.top_n,
        is_reference=False,
        active_only=False,
        activity_col=args.activity_col,
    )

    all_tables = {
        "Known MRSA Actives": reference,
        "General-Model Generated": original,
        "MRSA-Trained Generated": scratch,
        "RCFR-Diffusion for MRSA": rcfr,
    }

    for group, table in all_tables.items():
        print(f"{group}: {len(table)} valid molecules")

    descriptor_stats = compute_descriptor_stats(all_tables)
    descriptor_detail, descriptor_summary = compute_descriptor_distance(reference, all_tables)
    tanimoto_detail, tanimoto_summary = compute_tanimoto_similarity(
        reference=reference,
        all_tables=all_tables,
        max_reference_self=args.max_reference_self,
        seed=args.seed,
    )
    scaffold_summary, scaffold_detail = compute_scaffold_fragment(
        reference=reference,
        all_tables=all_tables,
        top_fragments=args.top_fragments,
    )

    final_summary = build_final_summary(
        descriptor_summary=descriptor_summary,
        tanimoto_summary=tanimoto_summary,
        scaffold_summary=scaffold_summary,
    )

    descriptor_stats.to_csv(outdir / "descriptor_statistics_four_groups.csv", index=False, encoding="utf-8-sig")
    descriptor_detail.to_csv(outdir / "descriptor_distance_detail.csv", index=False, encoding="utf-8-sig")
    descriptor_summary.to_csv(outdir / "descriptor_distance_summary.csv", index=False, encoding="utf-8-sig")
    tanimoto_detail.to_csv(outdir / "tanimoto_similarity_detail.csv", index=False, encoding="utf-8-sig")
    tanimoto_summary.to_csv(outdir / "tanimoto_similarity_summary.csv", index=False, encoding="utf-8-sig")
    scaffold_summary.to_csv(outdir / "scaffold_fragment_summary.csv", index=False, encoding="utf-8-sig")
    scaffold_detail.to_csv(outdir / "scaffold_fragment_detail.csv", index=False, encoding="utf-8-sig")
    final_summary.to_csv(outdir / "database_closeness_final_summary.csv", index=False, encoding="utf-8-sig")

    for group, table in all_tables.items():
        name = group.lower().replace(" ", "_").replace("-", "_")
        table.drop(columns=["mol"]).to_csv(outdir / f"{name}_molecule_descriptor_table.csv", index=False, encoding="utf-8-sig")

    plot_property_distributions(all_tables, outdir)
    plot_descriptor_distance_heatmap(descriptor_detail, outdir)
    plot_tanimoto_boxplot(tanimoto_detail, outdir)
    plot_closeness_summary(final_summary, outdir)

    evidence = {
        "groups": GROUP_ORDER,
        "top_n_generated": args.top_n,
        "reference_active_only": bool(args.reference_active_only),
        "outputs": {
            "final_summary": str((outdir / "database_closeness_final_summary.csv").resolve()),
            "property_distribution": str((outdir / "property_distribution_four_groups.png").resolve()),
            "descriptor_heatmap": str((outdir / "descriptor_distance_heatmap.png").resolve()),
            "tanimoto_boxplot": str((outdir / "tanimoto_similarity_boxplot.png").resolve()),
            "summary_bar": str((outdir / "database_closeness_summary_bar.png").resolve()),
        },
    }

    (outdir / "evidence_summary.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFinal summary:")
    print(final_summary.to_string(index=False))
    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
