#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Table 4-2 from generated CSV only.

The script calculates basic molecule quality metrics from:
1. Generated molecule CSV with a SMILES column
2. Training/reference CSV with a SMILES column

Outputs:
1. English table CSV
2. Raw metric JSON
3. Optional valid molecule detail CSV
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def format_value(value, is_ratio: bool = False) -> str:
    if value is None:
        return "N/A"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "N/A"
    except Exception:
        pass

    if is_ratio:
        return f"{float(value):.4f}"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def remove_isolated_hydrogen_atoms(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None

    try:
        rw_mol = Chem.RWMol(mol)
        remove_ids = []
        for atom in rw_mol.GetAtoms():
            if atom.GetSymbol() == "H" and atom.GetDegree() == 0:
                remove_ids.append(atom.GetIdx())

        for idx in sorted(remove_ids, reverse=True):
            rw_mol.RemoveAtom(idx)

        clean = rw_mol.GetMol()
        if clean.GetNumAtoms() == 0:
            return None
        return clean
    except Exception:
        return None


def choose_largest_organic_fragment(mol: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return None

    if not fragments:
        return None

    def fragment_score(fragment: Chem.Mol):
        atoms = list(fragment.GetAtoms())
        has_carbon = int(any(atom.GetSymbol() == "C" for atom in atoms))
        heavy_atoms = sum(1 for atom in atoms if atom.GetAtomicNum() > 1)
        total_atoms = len(atoms)
        return has_carbon, heavy_atoms, total_atoms

    return sorted(fragments, key=fragment_score, reverse=True)[0]


def mol_from_smiles_robust(smiles: str, keep_largest_fragment: bool = False) -> Optional[Chem.Mol]:
    if pd.isna(smiles):
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    except Exception:
        return None

    if mol is None:
        return None

    mol = remove_isolated_hydrogen_atoms(mol)
    if mol is None:
        return None

    if keep_largest_fragment:
        mol = choose_largest_organic_fragment(mol)
        if mol is None:
            return None
        mol = remove_isolated_hydrogen_atoms(mol)
        if mol is None:
            return None

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


def canonicalize_generated_smiles(smiles: str) -> Optional[str]:
    mol = mol_from_smiles_robust(smiles, keep_largest_fragment=False)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonicalize_training_smiles(smiles: str) -> Optional[str]:
    mol = mol_from_smiles_robust(smiles, keep_largest_fragment=True)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def is_connected_molecule(mol: Chem.Mol) -> bool:
    try:
        fragments = Chem.GetMolFrags(mol)
        return len(fragments) == 1
    except Exception:
        return False


def load_training_smiles(training_csv: Path, smiles_col: str) -> set:
    df = pd.read_csv(training_csv)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' was not found in {training_csv}")

    training_smiles = set()
    for smiles in df[smiles_col].dropna().astype(str):
        canonical = canonicalize_training_smiles(smiles)
        if canonical:
            training_smiles.add(canonical)

    return training_smiles


def load_generated_smiles(generated_csv: Path, smiles_col: str, top_n: int = 0) -> Tuple[pd.DataFrame, List[Chem.Mol]]:
    df = pd.read_csv(generated_csv)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' was not found in {generated_csv}")

    if "rank" in df.columns:
        df = df.sort_values("rank").copy()

    if top_n and top_n > 0:
        df = df.head(top_n).copy()

    rows = []
    valid_mols = []

    for source_row_index, row in df.iterrows():
        raw_smiles = row[smiles_col]
        mol = mol_from_smiles_robust(raw_smiles, keep_largest_fragment=False)

        if mol is None:
            rows.append({
                "source_row_index": int(source_row_index),
                "raw_smiles": raw_smiles,
                "canonical_smiles": "",
                "is_valid": 0,
                "is_connected": 0,
            })
            continue

        try:
            canonical = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            rows.append({
                "source_row_index": int(source_row_index),
                "raw_smiles": raw_smiles,
                "canonical_smiles": "",
                "is_valid": 0,
                "is_connected": 0,
            })
            continue

        connected = int(is_connected_molecule(mol))
        valid_mols.append(mol)

        rows.append({
            "source_row_index": int(source_row_index),
            "raw_smiles": raw_smiles,
            "canonical_smiles": canonical,
            "is_valid": 1,
            "is_connected": connected,
        })

    return pd.DataFrame(rows), valid_mols


def mean_pairwise_diversity_from_smiles(unique_smiles: List[str], max_pairs: int = 0, seed: int = 42) -> float:
    mols = [Chem.MolFromSmiles(smiles) for smiles in unique_smiles]
    mols = [mol for mol in mols if mol is not None]

    if len(mols) <= 1:
        return float("nan")

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        for mol in mols
    ]

    n = len(fps)
    total_pairs = n * (n - 1) // 2

    if max_pairs and total_pairs > max_pairs:
        rng = np.random.default_rng(seed)
        similarities = []
        for _ in range(max_pairs):
            i, j = rng.choice(n, size=2, replace=False)
            similarities.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        mean_similarity = float(np.mean(similarities))
        return 1.0 - mean_similarity

    similarities = []
    for i in range(1, n):
        similarities.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))

    if not similarities:
        return float("nan")

    mean_similarity = float(np.mean(similarities))
    return 1.0 - mean_similarity


def compute_table(
    generated_csv: Path,
    training_csv: Path,
    generated_smiles_col: str,
    training_smiles_col: str,
    sampled_count: int,
    top_n: int,
    max_diversity_pairs: int,
    seed: int,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    generated_detail, _ = load_generated_smiles(
        generated_csv=generated_csv,
        smiles_col=generated_smiles_col,
        top_n=top_n,
    )

    if sampled_count <= 0:
        sampled_count = int(len(generated_detail))

    valid_detail = generated_detail[generated_detail["is_valid"] == 1].copy()
    valid_count = int(len(valid_detail))

    connected_count = int(valid_detail["is_connected"].sum()) if valid_count > 0 else 0

    unique_smiles = sorted(set(valid_detail["canonical_smiles"].dropna().astype(str)))
    unique_smiles = [smiles for smiles in unique_smiles if smiles]
    unique_count = int(len(unique_smiles))

    training_smiles = load_training_smiles(training_csv, training_smiles_col)

    novel_smiles = [smiles for smiles in unique_smiles if smiles not in training_smiles]
    novel_count = int(len(novel_smiles))

    diversity = mean_pairwise_diversity_from_smiles(
        unique_smiles,
        max_pairs=max_diversity_pairs,
        seed=seed,
    )

    metrics = {
        "sampled_count": int(sampled_count),
        "generated_csv_rows_used": int(len(generated_detail)),
        "valid_count": valid_count,
        "validity": safe_ratio(valid_count, sampled_count),
        "connected_count": connected_count,
        "connected_ratio": safe_ratio(connected_count, valid_count),
        "unique_count": unique_count,
        "uniqueness": safe_ratio(unique_count, valid_count),
        "novel_count": novel_count,
        "novelty": safe_ratio(novel_count, valid_count),
        "diversity": float(diversity),
        "training_unique_smiles": int(len(training_smiles)),
    }

    table_rows = [
        {
            "Metric": "Sampled molecules",
            "Value": format_value(metrics["sampled_count"]),
            "Description": "Number of molecules sampled from the generative model.",
        },
        {
            "Metric": "Valid molecules",
            "Value": format_value(metrics["valid_count"]),
            "Description": "Number of generated molecules that can be legally parsed and sanitized by RDKit.",
        },
        {
            "Metric": "Validity ↑",
            "Value": format_value(metrics["validity"], is_ratio=True),
            "Description": "Ratio of valid molecules to sampled molecules.",
        },
        {
            "Metric": "Connected ↑",
            "Value": format_value(metrics["connected_ratio"], is_ratio=True),
            "Description": "Ratio of single-component molecules to valid molecules.",
        },
        {
            "Metric": "Unique molecules",
            "Value": format_value(metrics["unique_count"]),
            "Description": "Number of unique valid molecules after canonical SMILES deduplication.",
        },
        {
            "Metric": "Uniqueness ↑",
            "Value": format_value(metrics["uniqueness"], is_ratio=True),
            "Description": "Ratio of unique molecules to valid molecules.",
        },
        {
            "Metric": "Novel molecules",
            "Value": format_value(metrics["novel_count"]),
            "Description": "Number of unique generated molecules not found in the training set.",
        },
        {
            "Metric": "Novelty ↑",
            "Value": format_value(metrics["novelty"], is_ratio=True),
            "Description": "Ratio of novel molecules to valid molecules.",
        },
        {
            "Metric": "Diversity ↑",
            "Value": format_value(metrics["diversity"], is_ratio=True),
            "Description": "Average pairwise structural dissimilarity among unique generated molecules based on Morgan fingerprints.",
        },
    ]

    return pd.DataFrame(table_rows), metrics, generated_detail


def main():
    parser = argparse.ArgumentParser(description="Generate Table 4-2 from generated CSV only.")
    parser.add_argument("--generated_csv", required=True, help="Path to generated ranked CSV.")
    parser.add_argument("--training_csv", required=True, help="Path to training/reference CSV.")
    parser.add_argument("--out_csv", required=True, help="Output CSV path for Table 4-2.")
    parser.add_argument("--generated_smiles_col", default="smiles")
    parser.add_argument("--training_smiles_col", default="smiles")
    parser.add_argument("--sampled_count", type=int, default=0, help="Raw sampling count. If 0, use generated CSV row count.")
    parser.add_argument("--top_n", type=int, default=0, help="Use top N rows from generated CSV after rank sorting. 0 means all rows.")
    parser.add_argument("--max_diversity_pairs", type=int, default=0, help="Random pair cap for diversity. 0 means all pairs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", default="", help="Optional output JSON path for raw metrics.")
    parser.add_argument("--out_detail_csv", default="", help="Optional output CSV path for valid parsing details.")
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    table_df, metrics, generated_detail = compute_table(
        generated_csv=Path(args.generated_csv),
        training_csv=Path(args.training_csv),
        generated_smiles_col=args.generated_smiles_col,
        training_smiles_col=args.training_smiles_col,
        sampled_count=args.sampled_count,
        top_n=args.top_n,
        max_diversity_pairs=args.max_diversity_pairs,
        seed=args.seed,
    )

    table_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if args.out_detail_csv:
        generated_detail.to_csv(args.out_detail_csv, index=False, encoding="utf-8-sig")

    print(f"Saved table CSV: {out_csv}")
    print(f"Saved metric JSON: {out_json}")
    if args.out_detail_csv:
        print(f"Saved detail CSV: {args.out_detail_csv}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
