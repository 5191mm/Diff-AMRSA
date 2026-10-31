#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


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
        mol2 = Chem.MolFromSmiles(smi)
        return mol2
    except Exception:
        return mol


def canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)
    except Exception:
        return None


def morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


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


def load_reference(path: Path, active_only: bool, activity_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])

    if smiles_col is None:
        raise ValueError(f"No SMILES column found in reference file: {path}")

    if active_only and activity_col in df.columns:
        df = df[df[activity_col] == 1].copy()

    rows = []
    seen = set()

    for idx, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])
        if mol is None:
            continue

        smi = canonical_smiles(mol)
        if not smi or smi in seen:
            continue

        seen.add(smi)

        rows.append({
            "source_index": int(idx),
            "smiles": smi,
            "mol": mol,
            "scaffold": scaffold_smiles(mol),
            "bits": sparse_bits(mol),
        })

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError(f"No valid reference molecules loaded from {path}")

    return out


def load_generated(path: Path, method: str, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles_col = detect_col(df, ["smiles", "SMILES", "canonical_smiles", "mol_smiles"])

    if smiles_col is None:
        raise ValueError(f"No SMILES column found in generated file: {path}")

    if top_n > 0:
        df = df.head(top_n).copy()

    rows = []

    for idx, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])
        if mol is None:
            continue

        smi = canonical_smiles(mol)
        if not smi:
            continue

        total_score = np.nan
        if "total_score" in row.index:
            total_score = pd.to_numeric(row["total_score"], errors="coerce")

        rows.append({
            "Method": method,
            "source_index": int(idx),
            "smiles": smi,
            "mol": mol,
            "scaffold": scaffold_smiles(mol),
            "bits": sparse_bits(mol),
            "total_score": float(total_score) if pd.notna(total_score) else np.nan,
        })

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError(f"No valid generated molecules loaded from {path}")

    return out


def build_reference_fragment_set(reference: pd.DataFrame, top_fragments: int) -> set:
    counter = Counter()

    for bits in reference["bits"]:
        counter.update(bits)

    if top_fragments <= 0:
        return set(counter.keys())

    return set(bit for bit, _ in counter.most_common(top_fragments))


def calc_method_metrics(
    gen: pd.DataFrame,
    reference: pd.DataFrame,
    top_ref_bits: set,
    high_score_threshold: float,
) -> Dict[str, float]:
    ref_fps = [morgan_fp(mol) for mol in reference["mol"]]
    ref_scaffolds = set(reference["scaffold"])
    ref_scaffolds.discard("NO_SCAFFOLD")

    max_sims = []
    scaffold_hits = []
    key_fragment_coverages = []

    detail_rows = []

    for _, row in gen.iterrows():
        mol = row["mol"]
        fp = morgan_fp(mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)

        if sims:
            max_sim = float(np.max(sims))
        else:
            max_sim = np.nan

        scaf = row["scaffold"]
        scaffold_hit = int(scaf in ref_scaffolds and scaf != "NO_SCAFFOLD")

        bits = row["bits"]
        if len(bits) == 0:
            key_cov = np.nan
        else:
            # 这里的定义是：每个生成分子的 Morgan 片段中，有多少比例属于参考库高频关键片段；
            # 再对所有生成分子取均值。这个值通常更接近论文中“Average coverage”的含义。
            key_cov = len(bits & top_ref_bits) / len(bits)

        max_sims.append(max_sim)
        scaffold_hits.append(scaffold_hit)
        key_fragment_coverages.append(key_cov)

        detail_rows.append({
            "Method": row["Method"],
            "smiles": row["smiles"],
            "max_tanimoto": max_sim,
            "scaffold": scaf,
            "scaffold_in_reference": scaffold_hit,
            "key_fragment_coverage_per_molecule": key_cov,
            "total_score": row["total_score"],
        })

    max_sims_arr = np.array(max_sims, dtype=float)
    max_sims_arr = max_sims_arr[np.isfinite(max_sims_arr)]

    total_score = pd.to_numeric(gen["total_score"], errors="coerce")

    known_scaffold_ratio = float(np.mean(scaffold_hits)) if len(scaffold_hits) else np.nan
    new_scaffold_ratio = 1.0 - known_scaffold_ratio if np.isfinite(known_scaffold_ratio) else np.nan

    return {
        "Mean maximum Tanimoto": float(np.mean(max_sims_arr)) if len(max_sims_arr) else np.nan,
        "Median maximum Tanimoto": float(np.median(max_sims_arr)) if len(max_sims_arr) else np.nan,
        "Scaffold coverage": known_scaffold_ratio,
        "New scaffold ratio": new_scaffold_ratio,
        "Key fragment coverage": float(np.nanmean(key_fragment_coverages)),
        "Mean activity proxy score": float(total_score.mean()) if total_score.notna().any() else np.nan,
        "High-score candidate ratio": float((total_score >= high_score_threshold).mean()) if total_score.notna().any() else np.nan,
        "_detail": pd.DataFrame(detail_rows),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--original_csv", required=True)
    parser.add_argument("--rcfr_csv", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--top_n", type=int, default=200)
    parser.add_argument("--top_fragments", type=int, default=100)
    parser.add_argument("--high_score_threshold", type=float, default=0.70)

    parser.add_argument("--reference_active_only", action="store_true")
    parser.add_argument("--activity_col", default="activity_label")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(
        Path(args.reference_csv),
        active_only=args.reference_active_only,
        activity_col=args.activity_col,
    )

    original = load_generated(
        Path(args.original_csv),
        method="Original Megalodon",
        top_n=args.top_n,
    )

    rcfr = load_generated(
        Path(args.rcfr_csv),
        method="RCFR-Diffusion for MRSA",
        top_n=args.top_n,
    )

    top_ref_bits = build_reference_fragment_set(reference, args.top_fragments)

    original_metrics = calc_method_metrics(
        original,
        reference,
        top_ref_bits,
        high_score_threshold=args.high_score_threshold,
    )

    rcfr_metrics = calc_method_metrics(
        rcfr,
        reference,
        top_ref_bits,
        high_score_threshold=args.high_score_threshold,
    )

    metric_descriptions = {
        "Mean maximum Tanimoto": "Mean of the maximum Tanimoto similarity between each generated molecule and the anti-MRSA reference library.",
        "Median maximum Tanimoto": "Median of the maximum structural similarity between generated molecules and the anti-MRSA reference library.",
        "Scaffold coverage": "Ratio of generated molecules whose Bemis-Murcko scaffolds appear in the anti-MRSA reference library.",
        "New scaffold ratio": "Ratio of generated molecules with scaffolds not observed in the anti-MRSA reference library.",
        "Key fragment coverage": "Average ratio of generated-molecule Morgan fragments covered by top anti-MRSA reference fragments.",
        "Mean activity proxy score": "Mean activity proxy score calculated from total_score.",
        "High-score candidate ratio": f"Ratio of generated molecules with activity proxy score above {args.high_score_threshold}.",
    }

    metric_cn = {
        "Mean maximum Tanimoto": "平均最大 Tanimoto 相似度 ↑",
        "Median maximum Tanimoto": "中位最大 Tanimoto 相似度 ↑",
        "Scaffold coverage": "骨架覆盖率 ↑",
        "New scaffold ratio": "新骨架比例 ↑",
        "Key fragment coverage": "关键片段覆盖率 ↑",
        "Mean activity proxy score": "平均活性代理评分 ↑",
        "High-score candidate ratio": "高评分候选比例 ↑",
    }

    rows_en = []
    rows_cn = []

    for metric in metric_descriptions:
        rows_en.append({
            "Metric": metric,
            "Original Megalodon": original_metrics[metric],
            "RCFR-Diffusion for MRSA": rcfr_metrics[metric],
            "Description": metric_descriptions[metric],
        })

        rows_cn.append({
            "指标": metric_cn[metric],
            "Original Megalodon": original_metrics[metric],
            "RCFR-Diffusion for MRSA": rcfr_metrics[metric],
            "指标说明": metric_descriptions[metric],
        })

    table_en = pd.DataFrame(rows_en)
    table_cn = pd.DataFrame(rows_cn)

    for col in ["Original Megalodon", "RCFR-Diffusion for MRSA"]:
        table_en[col] = table_en[col].astype(float).round(4)
        table_cn[col] = table_cn[col].astype(float).round(4)

    table_en.to_csv(outdir / "table_4_4_domain_consistency_compare_en.csv", index=False, encoding="utf-8-sig")
    table_cn.to_csv(outdir / "table_4_4_domain_consistency_compare_cn.csv", index=False, encoding="utf-8-sig")

    original_metrics["_detail"].to_csv(outdir / "original_megalodon_domain_consistency_detail.csv", index=False, encoding="utf-8-sig")
    rcfr_metrics["_detail"].to_csv(outdir / "rcfr_domain_consistency_detail.csv", index=False, encoding="utf-8-sig")

    print("\nEnglish table:")
    print(table_en.to_string(index=False))

    print("\nChinese table:")
    print(table_cn.to_string(index=False))

    print("\nSaved:")
    print(outdir / "table_4_4_domain_consistency_compare_en.csv")
    print(outdir / "table_4_4_domain_consistency_compare_cn.csv")
    print(outdir / "original_megalodon_domain_consistency_detail.csv")
    print(outdir / "rcfr_domain_consistency_detail.csv")


if __name__ == "__main__":
    main()
