#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import pickle
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
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
    "Original MRSA Dataset",
    "Original Megalodon",
    "Scratch-MRSA",
    "RCFR-Diffusion for MRSA",
]

GENERATED_GROUPS = [
    "Original Megalodon",
    "Scratch-MRSA",
    "RCFR-Diffusion for MRSA",
]

GROUP_COLORS = {
    "Original MRSA Dataset": "#1f77b4",
    "Original Megalodon": "#7f7f7f",
    "Scratch-MRSA": "#ff7f0e",
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


def resolve_path(project_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


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

        for extra_col in [
            "pred_pmic_ug_ml",
            "pred_log10_pmic",
            "activity_label_probability",
            "activity_label",
            "total_score",
            "mic_score",
            "lipinski_score",
            "alert_score",
            "alert_count",
        ]:
            if extra_col in row_raw.index:
                row[extra_col] = row_raw[extra_col]

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


def read_summary_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def average_pairwise_diversity(smiles_list: List[str], max_pairs: int = 20000, seed: int = 42) -> float:
    mols = []
    for smi in smiles_list:
        mol = robust_mol_from_smiles(smi, keep_largest_fragment=True)
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
            pairs.append((i, j))

    distances = []
    for i, j in pairs:
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        distances.append(1.0 - float(sim))

    return float(np.mean(distances)) if distances else float("nan")


def write_raw_data_material(raw_csv: Path, reference_csv: Path, outdir: Path) -> None:
    source_csv = raw_csv if raw_csv.exists() else reference_csv
    df = pd.read_csv(source_csv)

    likely_cols = [
        "smiles",
        "SMILES",
        "canonical_smiles",
        "activity_type",
        "standard_type",
        "type",
        "relation",
        "activity_value",
        "standard_value",
        "value",
        "concentration",
        "activity_unit",
        "standard_units",
        "unit",
        "pmic_ug_ml",
        "source",
        "data_source",
        "reference",
        "assay_description",
        "target",
        "organism",
    ]

    cols = [c for c in likely_cols if c in df.columns]
    if not cols:
        cols = list(df.columns[:12])

    out = df[cols].copy()
    out.to_csv(outdir / "01_raw_mrsa_activity_molecule_data_table.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([
        {
            "Item": "Source CSV",
            "Value": str(source_csv),
        },
        {
            "Item": "Rows",
            "Value": len(df),
        },
        {
            "Item": "Columns kept in material table",
            "Value": ", ".join(cols),
        },
        {
            "Item": "Note",
            "Value": "This table is generated from the available raw or processed MRSA activity CSV. It is not fabricated.",
        },
    ])
    summary.to_csv(outdir / "01_raw_data_source_summary.csv", index=False, encoding="utf-8-sig")


def write_cleaning_material(reference_csv: Path, reference_table: pd.DataFrame, outdir: Path) -> None:
    ref_raw = pd.read_csv(reference_csv)

    cleaning_result = reference_table.drop(columns=["mol"]).copy()
    cleaning_result.to_csv(outdir / "02_cleaned_standardized_mrsa_dataset_table.csv", index=False, encoding="utf-8-sig")

    rows = [
        {
            "Metric": "Rows in standardized CSV",
            "Value": len(ref_raw),
        },
        {
            "Metric": "Valid unique molecules parsed by RDKit",
            "Value": len(reference_table),
        },
        {
            "Metric": "Unique canonical SMILES",
            "Value": reference_table["smiles"].nunique(),
        },
    ]

    if "activity_label" in ref_raw.columns:
        counts = ref_raw["activity_label"].value_counts(dropna=False).to_dict()
        for k, v in counts.items():
            rows.append({
                "Metric": f"activity_label={k}",
                "Value": v,
            })

    if "pmic_ug_ml" in ref_raw.columns:
        vals = pd.to_numeric(ref_raw["pmic_ug_ml"], errors="coerce").dropna()
        rows.extend([
            {"Metric": "MIC count", "Value": len(vals)},
            {"Metric": "MIC mean ug/mL", "Value": float(vals.mean()) if len(vals) else np.nan},
            {"Metric": "MIC median ug/mL", "Value": float(vals.median()) if len(vals) else np.nan},
            {"Metric": "MIC min ug/mL", "Value": float(vals.min()) if len(vals) else np.nan},
            {"Metric": "MIC max ug/mL", "Value": float(vals.max()) if len(vals) else np.nan},
        ])

    pd.DataFrame(rows).to_csv(outdir / "02_cleaning_standardization_summary.csv", index=False, encoding="utf-8-sig")


def safe_object_count(path: Path) -> Tuple[Optional[int], str]:
    if not path.exists():
        return None, "missing"

    suffix = path.suffix.lower()

    try:
        if suffix == ".pickle" or suffix == ".pkl":
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if hasattr(obj, "__len__"):
                return len(obj), "pickle_len"
            return None, "pickle_loaded_but_no_len"
    except Exception as e:
        return None, f"pickle_failed: {e}"

    try:
        if suffix == ".pt":
            try:
                import torch
                obj = torch.load(path, map_location="cpu")
                if hasattr(obj, "__len__"):
                    return len(obj), "torch_len"
                if isinstance(obj, dict):
                    for key, val in obj.items():
                        try:
                            if hasattr(val, "__len__"):
                                return len(val), f"torch_dict_key_len:{key}"
                        except Exception:
                            continue
                return None, "torch_loaded_but_count_unknown"
            except Exception as e:
                return None, f"torch_failed: {e}"
    except Exception as e:
        return None, f"load_failed: {e}"

    return None, "count_not_supported"


def write_conformer_split_material(processed_dir: Path, outdir: Path) -> None:
    rows = []
    for split in ["train", "val", "test"]:
        candidates = [
            processed_dir / f"{split}_h.pt",
            processed_dir / f"{split}_n_h.pickle",
            processed_dir / f"{split}_atom_types_h.npy",
            processed_dir / f"{split}_bond_types_h.npy",
            processed_dir / f"{split}_charges_prior_h.npy",
        ]

        for file in candidates:
            count, method = safe_object_count(file)
            rows.append({
                "Split": split,
                "File": str(file),
                "Exists": file.exists(),
                "Size MB": round(file.stat().st_size / 1024 / 1024, 3) if file.exists() else np.nan,
                "Parsed count": count if count is not None else "N/A",
                "Count method": method,
            })

    pd.DataFrame(rows).to_csv(outdir / "03_conformer_generation_and_dataset_split_statistics.csv", index=False, encoding="utf-8-sig")

    log_candidates = []
    for pattern in ["*conformer*", "*process*", "*geom*", "*failed*", "*skip*", "*.log"]:
        log_candidates.extend(processed_dir.parent.glob(pattern))
        log_candidates.extend(processed_dir.glob(pattern))

    log_rows = []
    seen = set()
    for p in log_candidates:
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            log_rows.append({
                "File": str(p),
                "Size MB": round(p.stat().st_size / 1024 / 1024, 3),
                "Modified time": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })

    pd.DataFrame(log_rows).to_csv(outdir / "03_conformer_related_log_files.csv", index=False, encoding="utf-8-sig")


def write_model_experiment_material(project_root: Path, args, outdir: Path) -> None:
    candidate_paths = [
        project_root / "scripts/conf/drugs/megalodon_diffusion.yaml",
        project_root / "scripts/conf/drugs/mrsa_diffusion_scratch_scheme_a.yaml",
        project_root / "scripts/conf/drugs/mrsa_diffusion_finetune_scheme_a.yaml",
        project_root / "scripts/rerun_three_line_sampling_robust_table_weights.sh",
        project_root / "scripts/sample_domain_adapted_robust.py",
        project_root / "outputs/compare_guided/logs/pretrained_baseline_sampling.log",
        project_root / "outputs/compare_guided/logs/scratch_mrsa_sampling.log",
        project_root / "outputs/compare_guided/logs/finetune_mrsa_sampling.log",
        project_root / "results/mrsa_strict_scratch_scheme_a/mrsa_strict_scratch_scheme_a/checkpoints/last.ckpt",
        project_root / "results/mrsa_strict_finetune_scheme_a/mrsa_strict_finetune_scheme_a/checkpoints/last.ckpt",
        resolve_path(project_root, args.original_ranked_csv),
        resolve_path(project_root, args.scratch_ranked_csv),
        resolve_path(project_root, args.rcfr_ranked_csv),
    ]

    rows = []
    for p in candidate_paths:
        rows.append({
            "Material type": classify_material_path(p),
            "Path": str(p),
            "Exists": p.exists(),
            "Size MB": round(p.stat().st_size / 1024 / 1024, 3) if p.exists() else np.nan,
            "Modified time": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if p.exists() else "",
        })

    pd.DataFrame(rows).to_csv(outdir / "04_pretrained_3d_diffusion_model_adaptation_records.csv", index=False, encoding="utf-8-sig")


def classify_material_path(path: Path) -> str:
    s = str(path).lower()
    if s.endswith(".yaml") or s.endswith(".yml"):
        return "Model configuration file"
    if s.endswith(".ckpt"):
        return "Model checkpoint"
    if s.endswith(".log"):
        return "Training or sampling log"
    if s.endswith(".sh"):
        return "Run command script"
    if s.endswith(".py"):
        return "Python script"
    if s.endswith(".csv"):
        return "Generated result table"
    return "Other material"


def write_candidate_evaluation_material(
    generated_tables: Dict[str, pd.DataFrame],
    all_scored_paths: Dict[str, Path],
    ranked_paths: Dict[str, Path],
    summary_paths: Dict[str, Path],
    reference_smiles: set,
    outdir: Path,
    sampled_count: int,
) -> None:
    top_rows = []
    for group, df in generated_tables.items():
        tmp = df.drop(columns=["mol"]).copy()
        tmp.insert(0, "Generated group", group)
        top_rows.append(tmp)

    pd.concat(top_rows, ignore_index=True).to_csv(outdir / "05_generated_candidate_evaluation_top200_table.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for group in GENERATED_GROUPS:
        ranked_path = ranked_paths[group]
        all_path = all_scored_paths[group]
        summary_json = read_summary_json(summary_paths[group])

        csv_for_quality = all_path if all_path.exists() else ranked_path
        qdf = pd.read_csv(csv_for_quality)

        smiles_col = "smiles"
        smiles = []
        valid_mols = []
        for smi in qdf[smiles_col].dropna().astype(str):
            can = canonicalize_smiles(smi, keep_largest_fragment=True)
            if can is not None:
                smiles.append(can)
                mol = robust_mol_from_smiles(can, keep_largest_fragment=False)
                if mol is not None:
                    valid_mols.append(mol)

        valid_count = len(smiles)
        unique_smiles = sorted(set(smiles))
        unique_count = len(unique_smiles)
        novel_count = sum(1 for s in unique_smiles if s not in reference_smiles)
        diversity = average_pairwise_diversity(unique_smiles, max_pairs=20000)

        generated_objects = summary_json.get("generated_objects", sampled_count)
        valid_raw_molecules = summary_json.get("valid_raw_molecules", valid_count)

        desc_rows = []
        for smi in unique_smiles:
            mol = robust_mol_from_smiles(smi, keep_largest_fragment=False)
            if mol is not None:
                desc_rows.append(mol_descriptors(mol))
        desc_df = pd.DataFrame(desc_rows)

        summary_rows.append({
            "Group": group,
            "Sampled molecules": generated_objects,
            "Valid raw molecules": valid_raw_molecules,
            "Scored molecules": valid_count,
            "Unique molecules": unique_count,
            "Novel molecules": novel_count,
            "Validity ↑": safe_ratio(valid_raw_molecules, generated_objects),
            "Uniqueness ↑": safe_ratio(unique_count, valid_count),
            "Novelty ↑": safe_ratio(novel_count, unique_count),
            "Diversity ↑": diversity,
            "QED mean ↑": float(desc_df["QED"].mean()) if "QED" in desc_df else np.nan,
            "SA mean ↓": float(desc_df["SA"].mean()) if "SA" in desc_df else np.nan,
            "LogP mean": float(desc_df["LogP"].mean()) if "LogP" in desc_df else np.nan,
            "TPSA mean": float(desc_df["TPSA"].mean()) if "TPSA" in desc_df else np.nan,
            "MW mean": float(desc_df["MW"].mean()) if "MW" in desc_df else np.nan,
            "HBD mean": float(desc_df["HBD"].mean()) if "HBD" in desc_df else np.nan,
            "HBA mean": float(desc_df["HBA"].mean()) if "HBA" in desc_df else np.nan,
        })

    pd.DataFrame(summary_rows).to_csv(outdir / "05_generated_candidate_quality_summary.csv", index=False, encoding="utf-8-sig")


def compute_descriptor_distance(reference: pd.DataFrame, all_tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []

    for group, df in all_tables.items():
        for col in DESCRIPTOR_COLUMNS + ["log10_MIC"]:
            ref_vals = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy(dtype=float)
            group_vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)

            if len(ref_vals) == 0 or len(group_vals) == 0:
                continue

            ref_mean = float(np.nanmean(ref_vals))
            group_mean = float(np.nanmean(group_vals))
            ref_std = float(np.nanstd(ref_vals))

            if ref_std <= 1e-8:
                ref_std = 1.0

            z_distance = abs(group_mean - ref_mean) / ref_std

            detail_rows.append({
                "Group": group,
                "Descriptor": col,
                "Reference mean": ref_mean,
                "Group mean": group_mean,
                "Mean absolute z-distance ↓": float(z_distance),
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

        if group == "Original MRSA Dataset":
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

        if group == "Original MRSA Dataset":
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

        ascending = direction == "lower"
        rank_col = f"Rank - {metric}"
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
        ("Mean descriptor distance ↓", "Lower is better"),
        ("Mean max Tanimoto ↑", "Higher is better"),
        ("Scaffold coverage ↑", "Higher is better"),
        ("Key fragment group coverage ↑", "Higher is better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    axes = axes.flatten()

    for ax, (metric, note) in zip(axes, metrics):
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
        ax.text(0.02, 0.94, note, transform=ax.transAxes, fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "database_closeness_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rcfr_representative_candidates(rcfr_table: pd.DataFrame, outdir: Path, top_n: int = 12) -> None:
    mols = []
    legends = []

    df = rcfr_table.head(top_n).copy()

    for i, row in df.iterrows():
        mol = robust_mol_from_smiles(row["smiles"], keep_largest_fragment=False)
        if mol is None:
            continue

        mols.append(mol)

        score = row.get("total_score", np.nan)
        pred_mic = row.get("pred_pmic_ug_ml", np.nan)
        qed = row.get("QED", np.nan)
        logp = row.get("LogP", np.nan)

        legends.append(
            f"Rank {len(mols)}\n"
            f"Score={float(score):.3f}" if pd.notna(score) else f"Rank {len(mols)}"
        )

        if pd.notna(pred_mic) or pd.notna(qed) or pd.notna(logp):
            legends[-1] += f"\nMIC={float(pred_mic):.2g}, QED={float(qed):.2f}, LogP={float(logp):.2f}"

    if not mols:
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(340, 260),
        legends=legends,
        useSVG=False,
    )
    img.save(str(outdir / "representative_rcfr_candidates_top12.png"))


def write_markdown_materials(
    outdir: Path,
    final_summary: pd.DataFrame,
    support_statement: str,
) -> None:
    material_list = f"""# 附件一：成果材料清单

1. 抗 MRSA 活性分子原始数据整理表  
   对应文件：`01_raw_mrsa_activity_molecule_data_table.csv`、`01_raw_data_source_summary.csv`

2. 抗 MRSA 分子数据清洗与标准化处理结果表  
   对应文件：`02_cleaned_standardized_mrsa_dataset_table.csv`、`02_cleaning_standardization_summary.csv`

3. 三维构象生成与数据集划分统计表  
   对应文件：`03_conformer_generation_and_dataset_split_statistics.csv`、`03_conformer_related_log_files.csv`

4. 预训练三维扩散模型适配实验记录  
   对应文件：`04_pretrained_3d_diffusion_model_adaptation_records.csv`

5. 生成候选分子评价结果表  
   对应文件：`05_generated_candidate_evaluation_top200_table.csv`、`05_generated_candidate_quality_summary.csv`

6. 代表性候选分子结构展示图  
   对应文件：`representative_rcfr_candidates_top12.png`

7. 专业实践总结报告  
   对应文件：由正文报告提交。本目录中的图表和统计结果可作为报告支撑材料。
"""

    paper_note = f"""# 附件二：纸质证明材料说明

本次专业实践拟提供纸质证明材料 1 份，材料名称为《抗 MRSA 药物分子生成专业实践成果证明材料》。

建议纸质材料包括以下内容：

1. 抗 MRSA 分子数据处理流程图；
2. 数据集预处理统计表；
3. 模型方法框架图；
4. 生成分子评价结果表；
5. 代表性候选分子展示图；
6. 四组分子数据库接近性对比图表。

本次补充实验从理化性质分布、二维结构相似性、骨架覆盖率和关键片段覆盖率四个方面比较了 Original MRSA Dataset、Original Megalodon、Scratch-MRSA 和 RCFR-Diffusion for MRSA。  
自动分析结论如下：

{support_statement}

注意：本脚本不伪造实验数据。若 RCFR-Diffusion for MRSA 未在真实计算指标中取得最优，报告中应如实表述为“部分指标支持”或“未完全支持”。
"""

    outdir.joinpath("附件一_成果材料清单.md").write_text(material_list, encoding="utf-8")
    outdir.joinpath("附件二_纸质证明材料说明.md").write_text(paper_note, encoding="utf-8")


def determine_support(final_summary: pd.DataFrame) -> str:
    generated_df = final_summary[final_summary["Group"].isin(GENERATED_GROUPS)].copy()

    if generated_df.empty:
        return "未能形成有效判断，因为生成分子组结果为空。"

    metrics = {
        "Mean descriptor distance ↓": "lower",
        "Mean max Tanimoto ↑": "higher",
        "Median max Tanimoto ↑": "higher",
        "Scaffold coverage ↑": "higher",
        "Key fragment group coverage ↑": "higher",
        "Mean key fragment coverage ↑": "higher",
        "Mean rank ↓": "lower",
    }

    rcfr_best_count = 0
    used_count = 0
    details = []

    for metric, direction in metrics.items():
        if metric not in generated_df.columns:
            continue

        tmp = generated_df[["Group", metric]].dropna()
        if tmp.empty:
            continue

        used_count += 1

        if direction == "lower":
            best_group = tmp.sort_values(metric, ascending=True).iloc[0]["Group"]
        else:
            best_group = tmp.sort_values(metric, ascending=False).iloc[0]["Group"]

        if best_group == "RCFR-Diffusion for MRSA":
            rcfr_best_count += 1

        details.append(f"{metric}: best={best_group}")

    if used_count == 0:
        return "未能形成有效判断，因为没有可用于比较的接近性指标。"

    if rcfr_best_count >= math.ceil(used_count * 0.6):
        conclusion = (
            f"真实计算结果支持 RCFR-Diffusion for MRSA 在多数接近性指标上优于其他生成线。"
            f"在 {used_count} 个可比较指标中，RCFR-Diffusion for MRSA 有 {rcfr_best_count} 个指标排名最优。"
        )
    else:
        conclusion = (
            f"真实计算结果未完全支持 RCFR-Diffusion for MRSA 在多数接近性指标上最优。"
            f"在 {used_count} 个可比较指标中，RCFR-Diffusion for MRSA 有 {rcfr_best_count} 个指标排名最优。"
            f"建议在报告中如实表述为部分指标支持，而不是伪造结论。"
        )

    return conclusion + "\n\n" + "\n".join(details)


def main():
    parser = argparse.ArgumentParser(description="Build MRSA professional practice attachment materials")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--raw_csv", default="processed_drug_mic_dataset.csv")
    parser.add_argument("--reference_csv", default="mrsa_scheme_a/processed_drug_mic_dataset_schemeA.csv")
    parser.add_argument("--processed_dir", default=None)

    parser.add_argument("--original_ranked_csv", default="outputs/compare_guided/pretrained_baseline/pretrained_baseline_guided_all_sdf_ranked.csv")
    parser.add_argument("--scratch_ranked_csv", default="outputs/compare_guided/scratch_mrsa/scratch_mrsa_guided_all_sdf_ranked.csv")
    parser.add_argument("--rcfr_ranked_csv", default="outputs/compare_guided/finetune_mrsa/finetune_mrsa_guided_all_sdf_ranked.csv")

    parser.add_argument("--original_all_scored_csv", default="outputs/compare_guided/pretrained_baseline/pretrained_baseline_guided_all_sdf_all_scored.csv")
    parser.add_argument("--scratch_all_scored_csv", default="outputs/compare_guided/scratch_mrsa/scratch_mrsa_guided_all_sdf_all_scored.csv")
    parser.add_argument("--rcfr_all_scored_csv", default="outputs/compare_guided/finetune_mrsa/finetune_mrsa_guided_all_sdf_all_scored.csv")

    parser.add_argument("--outdir", default="outputs/practice_attachment_materials")
    parser.add_argument("--top_n", type=int, default=200)
    parser.add_argument("--sampled_count", type=int, default=2000)
    parser.add_argument("--reference_active_only", action="store_true")
    parser.add_argument("--activity_col", default="activity_label")
    parser.add_argument("--top_fragments", type=int, default=100)
    parser.add_argument("--max_reference_self", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    outdir = resolve_path(project_root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_csv = resolve_path(project_root, args.raw_csv)
    reference_csv = resolve_path(project_root, args.reference_csv)
    processed_dir = Path(args.processed_dir)

    ranked_paths = {
        "Original Megalodon": resolve_path(project_root, args.original_ranked_csv),
        "Scratch-MRSA": resolve_path(project_root, args.scratch_ranked_csv),
        "RCFR-Diffusion for MRSA": resolve_path(project_root, args.rcfr_ranked_csv),
    }

    all_scored_paths = {
        "Original Megalodon": resolve_path(project_root, args.original_all_scored_csv),
        "Scratch-MRSA": resolve_path(project_root, args.scratch_all_scored_csv),
        "RCFR-Diffusion for MRSA": resolve_path(project_root, args.rcfr_all_scored_csv),
    }

    summary_paths = {
        "Original Megalodon": ranked_paths["Original Megalodon"].with_name("pretrained_baseline_guided_all_sdf_summary.json"),
        "Scratch-MRSA": ranked_paths["Scratch-MRSA"].with_name("scratch_mrsa_guided_all_sdf_summary.json"),
        "RCFR-Diffusion for MRSA": ranked_paths["RCFR-Diffusion for MRSA"].with_name("finetune_mrsa_guided_all_sdf_summary.json"),
    }

    required = [reference_csv] + list(ranked_paths.values())
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    reference = load_molecule_table(
        csv_path=reference_csv,
        smiles_col="smiles",
        group_name="Original MRSA Dataset",
        top_n=0,
        is_reference=True,
        active_only=args.reference_active_only,
        activity_col=args.activity_col,
    )

    generated_tables = {
        group: load_molecule_table(
            csv_path=path,
            smiles_col="smiles",
            group_name=group,
            top_n=args.top_n,
            is_reference=False,
            active_only=False,
            activity_col=args.activity_col,
        )
        for group, path in ranked_paths.items()
    }

    all_tables = {
        "Original MRSA Dataset": reference,
        **generated_tables,
    }

    print("Loaded molecule counts:")
    for group, table in all_tables.items():
        print(f"  {group}: {len(table)}")

    write_raw_data_material(raw_csv, reference_csv, outdir)
    write_cleaning_material(reference_csv, reference, outdir)
    write_conformer_split_material(processed_dir, outdir)
    write_model_experiment_material(project_root, args, outdir)

    reference_smiles = set(reference["smiles"].tolist())
    write_candidate_evaluation_material(
        generated_tables=generated_tables,
        all_scored_paths=all_scored_paths,
        ranked_paths=ranked_paths,
        summary_paths=summary_paths,
        reference_smiles=reference_smiles,
        outdir=outdir,
        sampled_count=args.sampled_count,
    )

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

    final_summary = build_final_summary(descriptor_summary, tanimoto_summary, scaffold_summary)

    descriptor_detail.to_csv(outdir / "supplement_descriptor_distance_detail.csv", index=False, encoding="utf-8-sig")
    descriptor_summary.to_csv(outdir / "supplement_descriptor_distance_summary.csv", index=False, encoding="utf-8-sig")
    tanimoto_detail.to_csv(outdir / "supplement_tanimoto_similarity_detail.csv", index=False, encoding="utf-8-sig")
    tanimoto_summary.to_csv(outdir / "supplement_tanimoto_similarity_summary.csv", index=False, encoding="utf-8-sig")
    scaffold_summary.to_csv(outdir / "supplement_scaffold_fragment_summary.csv", index=False, encoding="utf-8-sig")
    scaffold_detail.to_csv(outdir / "supplement_scaffold_fragment_detail.csv", index=False, encoding="utf-8-sig")
    final_summary.to_csv(outdir / "supplement_database_closeness_final_summary.csv", index=False, encoding="utf-8-sig")

    for group, table in all_tables.items():
        safe_name = group.lower().replace(" ", "_").replace("-", "_")
        table.drop(columns=["mol"]).to_csv(outdir / f"{safe_name}_descriptor_table.csv", index=False, encoding="utf-8-sig")

    plot_property_distributions(all_tables, outdir)
    plot_descriptor_distance_heatmap(descriptor_detail, outdir)
    plot_tanimoto_boxplot(tanimoto_detail, outdir)
    plot_closeness_summary(final_summary, outdir)
    plot_rcfr_representative_candidates(generated_tables["RCFR-Diffusion for MRSA"], outdir, top_n=12)

    support_statement = determine_support(final_summary)
    write_markdown_materials(outdir, final_summary, support_statement)

    evidence = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(project_root),
        "outdir": str(outdir),
        "reference_csv": str(reference_csv),
        "top_n_generated": args.top_n,
        "reference_active_only": bool(args.reference_active_only),
        "support_statement": support_statement,
        "core_outputs": {
            "final_summary": str((outdir / "supplement_database_closeness_final_summary.csv").resolve()),
            "property_distribution": str((outdir / "property_distribution_four_groups.png").resolve()),
            "descriptor_heatmap": str((outdir / "descriptor_distance_heatmap.png").resolve()),
            "tanimoto_boxplot": str((outdir / "tanimoto_similarity_boxplot.png").resolve()),
            "summary_bar": str((outdir / "database_closeness_summary_bar.png").resolve()),
            "representative_candidates": str((outdir / "representative_rcfr_candidates_top12.png").resolve()),
        },
    }

    (outdir / "evidence_summary.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSupport statement:")
    print(support_statement)

    print("\nFinal summary:")
    print(final_summary.to_string(index=False))

    print(f"\nSaved all attachment materials to: {outdir}")


if __name__ == "__main__":
    main()
