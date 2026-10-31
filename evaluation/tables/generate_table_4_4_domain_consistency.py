#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Table 4-4: Anti-MRSA domain consistency statistics.

Inputs:
1. Generated ranked CSV with a SMILES column
2. Real anti-MRSA reference CSV with a SMILES column

Outputs:
1. Table 4-4 CSV
2. Raw metric JSON
3. Generated molecule detail CSV
4. Key fragment coverage CSV

Recommended use:
python generate_table_4_4_domain_consistency.py \
  --generated_csv /autodl-fs/data/mrsa_generated_epoch29_b200/mrsa_epoch29_b200_ranked.csv \
  --reference_csv /autodl-fs/data/processed_drug_mic_dataset.csv \
  --out_csv /autodl-fs/data/table_4_4_domain_consistency.csv \
  --generated_smiles_col smiles \
  --reference_smiles_col smiles \
  --top_n 200
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

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


def mol_from_smiles_robust(smiles: str, keep_largest_fragment: bool = True) -> Optional[Chem.Mol]:
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


def canonicalize_smiles(smiles: str, keep_largest_fragment: bool = True) -> Optional[str]:
    mol = mol_from_smiles_robust(smiles, keep_largest_fragment=keep_largest_fragment)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return "NO_SCAFFOLD"
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
        return scaffold_smiles if scaffold_smiles else "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


def mol_to_fp(mol: Chem.Mol, n_bits: int):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)


def mol_to_sparse_fp_bits(mol: Chem.Mol) -> set:
    fp = AllChem.GetMorganFingerprint(mol, radius=2)
    return set(fp.GetNonzeroElements().keys())


def qed_safe(mol: Chem.Mol) -> float:
    try:
        return float(QED.qed(mol))
    except Exception:
        return float("nan")


def load_reference_table(
    reference_csv: Path,
    smiles_col: str,
    active_only: bool,
    activity_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(reference_csv)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' was not found in {reference_csv}")

    if active_only and activity_col in df.columns:
        df = df[df[activity_col] == 1].copy()

    rows = []
    seen = set()
    for _, row in df.iterrows():
        raw_smiles = row[smiles_col]
        mol = mol_from_smiles_robust(raw_smiles, keep_largest_fragment=True)
        if mol is None:
            continue
        try:
            canonical = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)

        rows.append({
            "smiles": canonical,
            "mol": mol,
            "scaffold": get_scaffold_smiles(mol),
            "qed": qed_safe(mol),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No valid reference anti-MRSA molecules were loaded")
    return out


def load_generated_table(
    generated_csv: Path,
    smiles_col: str,
    top_n: int,
) -> pd.DataFrame:
    df = pd.read_csv(generated_csv)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' was not found in {generated_csv}")

    if "rank" in df.columns:
        df = df.sort_values("rank").copy()

    if top_n and top_n > 0:
        df = df.head(top_n).copy()

    rows = []
    seen = set()
    for source_index, row in df.iterrows():
        raw_smiles = row[smiles_col]
        mol = mol_from_smiles_robust(raw_smiles, keep_largest_fragment=True)
        if mol is None:
            continue
        try:
            canonical = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)

        record = {
            "source_index": int(source_index),
            "rank": row.get("rank", np.nan),
            "smiles": canonical,
            "mol": mol,
            "scaffold": get_scaffold_smiles(mol),
            "qed": row.get("qed", qed_safe(mol)),
        }

        for col in [
            "activity_label_probability",
            "activity_label",
            "mic_score",
            "total_score",
            "pred_pmic_ug_ml",
            "pred_log10_pmic",
        ]:
            if col in row.index:
                record[col] = row.get(col, np.nan)

        rows.append(record)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No valid generated molecules were loaded")
    return out


def compute_max_tanimoto(generated: pd.DataFrame, reference: pd.DataFrame, n_bits: int) -> pd.DataFrame:
    ref_fps = [mol_to_fp(mol, n_bits=n_bits) for mol in reference["mol"]]
    ref_smiles = reference["smiles"].tolist()

    max_scores = []
    nearest_smiles = []

    for mol in generated["mol"]:
        fp = mol_to_fp(mol, n_bits=n_bits)
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        if not sims:
            max_scores.append(float("nan"))
            nearest_smiles.append("")
            continue
        idx = int(np.argmax(sims))
        max_scores.append(float(sims[idx]))
        nearest_smiles.append(ref_smiles[idx])

    out = generated.copy()
    out["max_tanimoto"] = max_scores
    out["nearest_reference_smiles"] = nearest_smiles
    return out


def build_reference_fragments(reference: pd.DataFrame, top_fragments: int) -> Tuple[List[int], pd.DataFrame]:
    bit_counter = Counter()
    reference_bit_sets = []

    for mol in reference["mol"]:
        bits = mol_to_sparse_fp_bits(mol)
        reference_bit_sets.append(bits)
        bit_counter.update(bits)

    top_bits = [bit for bit, _ in bit_counter.most_common(top_fragments)]

    rows = []
    denom = max(len(reference_bit_sets), 1)
    for bit in top_bits:
        coverage = sum(1 for bits in reference_bit_sets if bit in bits) / denom
        rows.append({
            "fragment_bit": int(bit),
            "reference_coverage": float(coverage),
            "reference_count": int(sum(1 for bits in reference_bit_sets if bit in bits)),
        })

    return top_bits, pd.DataFrame(rows)


def compute_fragment_coverage(generated: pd.DataFrame, top_bits: List[int], fragment_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    generated = generated.copy()
    denominator = max(len(top_bits), 1)

    molecule_scores = []
    generated_bit_sets = []

    for mol in generated["mol"]:
        bits = mol_to_sparse_fp_bits(mol)
        generated_bit_sets.append(bits)
        molecule_scores.append(sum(1 for bit in top_bits if bit in bits) / denominator)

    generated["key_fragment_coverage_score"] = molecule_scores

    rows = []
    n_gen = max(len(generated_bit_sets), 1)
    for _, row in fragment_table.iterrows():
        bit = int(row["fragment_bit"])
        generated_count = int(sum(1 for bits in generated_bit_sets if bit in bits))
        generated_coverage = generated_count / n_gen
        rows.append({
            "fragment_bit": bit,
            "reference_coverage": float(row["reference_coverage"]),
            "generated_coverage": float(generated_coverage),
            "generated_count": generated_count,
        })

    fragment_out = pd.DataFrame(rows)
    return generated, fragment_out


def normalize_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    vmin = values.min()
    vmax = values.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return pd.Series(np.ones(len(series)) * 0.5, index=series.index, dtype=float)
    return ((values - vmin) / (vmax - vmin)).clip(0, 1).fillna(0)


def add_activity_proxy_score(generated: pd.DataFrame, active_proxy_col: str) -> Tuple[pd.DataFrame, str]:
    out = generated.copy()

    if active_proxy_col and active_proxy_col in out.columns:
        values = pd.to_numeric(out[active_proxy_col], errors="coerce")
        out["activity_proxy_score"] = normalize_series(values)
        return out, active_proxy_col

    if "total_score" in out.columns:
        values = pd.to_numeric(out["total_score"], errors="coerce")
        out["activity_proxy_score"] = normalize_series(values)
        return out, "total_score"

    if "activity_label_probability" in out.columns:
        values = pd.to_numeric(out["activity_label_probability"], errors="coerce")
        out["activity_proxy_score"] = normalize_series(values)
        return out, "activity_label_probability"

    if "activity_label" in out.columns:
        values = pd.to_numeric(out["activity_label"], errors="coerce")
        out["activity_proxy_score"] = normalize_series(values)
        return out, "activity_label"

    if "mic_score" in out.columns:
        values = pd.to_numeric(out["mic_score"], errors="coerce")
        out["activity_proxy_score"] = normalize_series(values)
        return out, "mic_score"

    # fallback only when predictor score is absent
    qed_part = normalize_series(out["qed"]) if "qed" in out.columns else pd.Series(np.zeros(len(out)), index=out.index)
    sim_part = normalize_series(out["max_tanimoto"]) if "max_tanimoto" in out.columns else pd.Series(np.zeros(len(out)), index=out.index)
    frag_part = normalize_series(out["key_fragment_coverage_score"]) if "key_fragment_coverage_score" in out.columns else pd.Series(np.zeros(len(out)), index=out.index)
    out["activity_proxy_score"] = (0.40 * sim_part + 0.30 * frag_part + 0.30 * qed_part).clip(0, 1)
    return out, "fallback_similarity_fragment_qed"


def compute_domain_table(
    generated_csv: Path,
    reference_csv: Path,
    generated_smiles_col: str,
    reference_smiles_col: str,
    top_n: int,
    active_only: bool,
    activity_col: str,
    n_bits: int,
    top_fragments: int,
    high_score_threshold: float,
    active_proxy_col: str,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame]:
    reference = load_reference_table(
        reference_csv=reference_csv,
        smiles_col=reference_smiles_col,
        active_only=active_only,
        activity_col=activity_col,
    )
    generated = load_generated_table(
        generated_csv=generated_csv,
        smiles_col=generated_smiles_col,
        top_n=top_n,
    )

    generated = compute_max_tanimoto(generated, reference, n_bits=n_bits)

    reference_scaffolds = set(reference["scaffold"].dropna().astype(str))
    generated["is_known_scaffold"] = generated["scaffold"].map(lambda x: int(x in reference_scaffolds))

    scaffold_coverage = safe_ratio(int(generated["is_known_scaffold"].sum()), len(generated))
    new_scaffold_ratio = safe_ratio(int((generated["is_known_scaffold"] == 0).sum()), len(generated))

    top_bits, fragment_reference = build_reference_fragments(reference, top_fragments=top_fragments)
    generated, fragment_table = compute_fragment_coverage(generated, top_bits, fragment_reference)

    key_fragment_coverage = float(generated["key_fragment_coverage_score"].mean())

    generated, proxy_source = add_activity_proxy_score(generated, active_proxy_col=active_proxy_col)
    high_score_ratio = safe_ratio(
        int((generated["activity_proxy_score"] >= high_score_threshold).sum()),
        len(generated),
    )

    metrics = {
        "generated_molecules": int(len(generated)),
        "reference_molecules": int(len(reference)),
        "mean_max_tanimoto": float(generated["max_tanimoto"].mean()),
        "median_max_tanimoto": float(generated["max_tanimoto"].median()),
        "scaffold_coverage": float(scaffold_coverage),
        "new_scaffold_ratio": float(new_scaffold_ratio),
        "key_fragment_coverage": float(key_fragment_coverage),
        "mean_activity_proxy_score": float(generated["activity_proxy_score"].mean()),
        "high_score_candidate_ratio": float(high_score_ratio),
        "high_score_threshold": float(high_score_threshold),
        "activity_proxy_source": proxy_source,
        "top_fragments": int(top_fragments),
    }

    table_rows = [
        {
            "Metric": "Mean maximum Tanimoto",
            "Value": format_value(metrics["mean_max_tanimoto"], is_ratio=True),
            "Description": "Mean of the maximum Tanimoto similarity between each generated molecule and the anti-MRSA reference library.",
        },
        {
            "Metric": "Median maximum Tanimoto",
            "Value": format_value(metrics["median_max_tanimoto"], is_ratio=True),
            "Description": "Median of the maximum structural similarity between generated molecules and the anti-MRSA reference library.",
        },
        {
            "Metric": "Scaffold coverage",
            "Value": format_value(metrics["scaffold_coverage"], is_ratio=True),
            "Description": "Ratio of generated molecules whose Bemis-Murcko scaffolds appear in the anti-MRSA reference library.",
        },
        {
            "Metric": "New scaffold ratio",
            "Value": format_value(metrics["new_scaffold_ratio"], is_ratio=True),
            "Description": "Ratio of generated molecules with scaffolds not observed in the anti-MRSA reference library.",
        },
        {
            "Metric": "Key fragment coverage",
            "Value": format_value(metrics["key_fragment_coverage"], is_ratio=True),
            "Description": "Average coverage of top anti-MRSA reference fragments in generated molecules.",
        },
        {
            "Metric": "Mean activity proxy score",
            "Value": format_value(metrics["mean_activity_proxy_score"], is_ratio=True),
            "Description": f"Mean activity proxy score calculated from {proxy_source}.",
        },
        {
            "Metric": "High-score candidate ratio",
            "Value": format_value(metrics["high_score_candidate_ratio"], is_ratio=True),
            "Description": f"Ratio of generated molecules with activity proxy score above {high_score_threshold:.2f}.",
        },
    ]

    table_df = pd.DataFrame(table_rows)
    return table_df, metrics, generated, fragment_table


def main():
    parser = argparse.ArgumentParser(description="Generate Table 4-4 anti-MRSA domain consistency statistics.")
    parser.add_argument("--generated_csv", required=True, help="Path to generated ranked CSV.")
    parser.add_argument("--reference_csv", required=True, help="Path to real anti-MRSA reference CSV.")
    parser.add_argument("--out_csv", required=True, help="Output CSV path for Table 4-4.")
    parser.add_argument("--generated_smiles_col", default="smiles")
    parser.add_argument("--reference_smiles_col", default="smiles")
    parser.add_argument("--top_n", type=int, default=200)
    parser.add_argument("--include_inactive_reference", action="store_true")
    parser.add_argument("--activity_col", default="activity_label")
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument("--top_fragments", type=int, default=30)
    parser.add_argument("--high_score_threshold", type=float, default=0.70)
    parser.add_argument("--active_proxy_col", default="", help="Optional generated CSV column used as activity proxy.")
    parser.add_argument("--out_json", default="")
    parser.add_argument("--out_detail_csv", default="")
    parser.add_argument("--out_fragment_csv", default="")
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    table_df, metrics, generated_detail, fragment_table = compute_domain_table(
        generated_csv=Path(args.generated_csv),
        reference_csv=Path(args.reference_csv),
        generated_smiles_col=args.generated_smiles_col,
        reference_smiles_col=args.reference_smiles_col,
        top_n=args.top_n,
        active_only=not args.include_inactive_reference,
        activity_col=args.activity_col,
        n_bits=args.n_bits,
        top_fragments=args.top_fragments,
        high_score_threshold=args.high_score_threshold,
        active_proxy_col=args.active_proxy_col,
    )

    table_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if args.out_detail_csv:
        detail = generated_detail.drop(columns=["mol"], errors="ignore")
        detail.to_csv(args.out_detail_csv, index=False, encoding="utf-8-sig")

    if args.out_fragment_csv:
        fragment_table.to_csv(args.out_fragment_csv, index=False, encoding="utf-8-sig")

    print(f"Saved table CSV: {out_csv}")
    print(f"Saved metric JSON: {out_json}")
    if args.out_detail_csv:
        print(f"Saved detail CSV: {args.out_detail_csv}")
    if args.out_fragment_csv:
        print(f"Saved fragment CSV: {args.out_fragment_csv}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
