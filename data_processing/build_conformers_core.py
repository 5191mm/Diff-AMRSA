#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Diff-AMRSA compatible MRSA geometry data from Scheme A CSV with a fixed unchanged training set.

This script follows the overall role of data_processing/process_geom.py but starts from a SMILES table.
It writes raw pickle splits and PyTorch Geometric processed files used by Diff-AMRSA. Each valid molecule is used once only.

Important compatibility rule
The drug checkpoint expects the original drug atom vocabulary.
Unsupported metals and salts are removed as counterions or skipped if they are part of the core.
No sample repeat, no sample weight, no activity-based filtering is applied in this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

# Match the pretrained drug checkpoint atom classes.
# Do not add new elements unless you retrain or resize checkpoint heads.
PRETRAIN_ATOM_ENCODER = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Al": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Br": 12,
    "I": 13,
    "Hg": 14,
    "Bi": 15,
}
ALLOWED_CHARGES = {-2, -1, 0, 1, 2, 3}


@dataclass
class BuildResult:
    ok: bool
    smiles_input: str
    smiles_core: Optional[str] = None
    mols: Optional[List[Chem.Mol]] = None
    reason: str = ""
    dropped_fragments: str = ""
    unsupported_symbols: str = ""
    scaffold: str = ""
    activity_label: Optional[int] = None
    pmic_ug_ml: Optional[float] = None
    pmic_train: Optional[float] = None
    pmic_regression_mask: Optional[int] = None
    generation_repeat: int = 1
    generation_repeat_ignored: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Diff-AMRSA MRSA Scheme A geometry dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Scheme A CSV file")
    parser.add_argument("--output_root", type=str, default="data/custom_mrsa_schemeA")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--pmic_col", type=str, default="pmic_ug_ml")
    parser.add_argument("--pmic_train_col", type=str, default="pmic_train")
    parser.add_argument("--label_col", type=str, default="activity_label")
    parser.add_argument("--mask_col", type=str, default="pmic_regression_mask")
    parser.add_argument("--repeat_col", type=str, default="generation_repeat")
    parser.add_argument("--num_candidates", type=int, default=30)
    parser.add_argument("--select_conformers", type=int, default=10)
    parser.add_argument("--max_atoms", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--neutralize_core", action="store_true")
    parser.add_argument("--keep_supported_counterions", action="store_true")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of outer Python worker processes for RDKit conformer generation. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--rdkit_threads",
        type=int,
        default=1,
        help="RDKit threads per molecule. Use 1 when num_workers > 1 to avoid CPU oversubscription. Use 0 only for single-process mode.",
    )
    parser.add_argument("--chunk_size", type=int, default=8, help="Task chunk size for multiprocessing")
    parser.add_argument("--raw_only", action="store_true", help="Only build raw pickle files and manifest. Skip PyG .pt conversion.")
    parser.add_argument("--processed_only", action="store_true", help="Only convert existing raw pickle files under output_root/raw to processed .pt files.")
    return parser.parse_args()


def import_utils_data() -> Tuple[object, object, object, object]:
    here = Path(__file__).resolve()
    candidates = [
        here.parent,
        here.parent.parent / "data_processing",
        Path.cwd() / "data_processing",
        Path.cwd(),
    ]
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
    try:
        from utils_data import mol_to_torch_geometric, compute_all_statistics, save_statistics, save_pyg_dataset
        return mol_to_torch_geometric, compute_all_statistics, save_statistics, save_pyg_dataset
    except ModuleNotFoundError as exc:
        if exc.name == "utils_data":
            from data_processing.utils_data import mol_to_torch_geometric, compute_all_statistics, save_statistics, save_pyg_dataset
            return mol_to_torch_geometric, compute_all_statistics, save_statistics, save_pyg_dataset
        raise RuntimeError(
            "Missing dependency while importing Megalodon data_processing/utils_data.py. "
            "Install project requirements including torch and torch_geometric before running this step."
        ) from exc


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)


def symbols_in_mol(mol: Chem.Mol) -> Counter:
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


def unsupported_symbols(mol: Chem.Mol) -> List[str]:
    syms = sorted(set(symbols_in_mol(mol)) - set(PRETRAIN_ATOM_ENCODER))
    return syms


def bad_charges(mol: Chem.Mol) -> List[int]:
    return sorted({atom.GetFormalCharge() for atom in mol.GetAtoms() if atom.GetFormalCharge() not in ALLOWED_CHARGES})


def has_carbon(mol: Chem.Mol) -> bool:
    return any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())


def largest_supported_core(mol: Chem.Mol, keep_supported_counterions: bool) -> Tuple[Optional[Chem.Mol], List[str], List[str]]:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None, [], []

    supported = []
    dropped = []
    unsupported = []
    for frag in frags:
        us = unsupported_symbols(frag)
        bc = bad_charges(frag)
        smi = Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True)
        if us or bc:
            dropped.append(smi)
            unsupported.extend(us + [f"charge_{c}" for c in bc])
            continue
        supported.append(frag)

    carbon_supported = [f for f in supported if has_carbon(f)]
    if not carbon_supported:
        return None, dropped, sorted(set(unsupported))

    core = max(carbon_supported, key=lambda m: m.GetNumHeavyAtoms())
    keep = [core]
    if keep_supported_counterions:
        core_smi = Chem.MolToSmiles(core, canonical=True, isomericSmiles=True)
        for frag in supported:
            smi = Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True)
            if smi == core_smi:
                continue
            if frag.GetNumHeavyAtoms() <= 3 and not has_carbon(frag):
                keep.append(frag)
            else:
                dropped.append(smi)
    else:
        for frag in supported:
            smi = Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True)
            if Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True) != Chem.MolToSmiles(core, canonical=True, isomericSmiles=True):
                dropped.append(smi)

    if len(keep) == 1:
        return core, dropped, sorted(set(unsupported))
    combo = keep[0]
    for frag in keep[1:]:
        combo = Chem.CombineMols(combo, frag)
    return combo, dropped, sorted(set(unsupported))


def neutralize_atoms(mol: Chem.Mol) -> Chem.Mol:
    # Adapted common RDKit neutralizer pattern
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    rw = Chem.RWMol(mol)
    matches = rw.GetMol().GetSubstructMatches(pattern)
    for (idx,) in matches:
        atom = rw.GetAtomWithIdx(idx)
        charge = atom.GetFormalCharge()
        h_count = atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(max(0, h_count - charge))
        atom.UpdatePropertyCache(False)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return mol
    return out


def embed_and_optimize(mol: Chem.Mol, num_candidates: int, select_conformers: int, seed: int, rdkit_threads: int = 1) -> Tuple[List[Chem.Mol], str]:
    mol_h = Chem.AddHs(mol, addCoords=False)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = int(rdkit_threads)
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.pruneRmsThresh = 0.25

    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=int(num_candidates), params=params))
    if not conf_ids:
        return [], "embed_failed"

    energies: List[Tuple[float, int, str]] = []
    mmff_props = None
    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
    except Exception:
        mmff_props = None

    if mmff_props is not None:
        for cid in conf_ids:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props, confId=int(cid))
                if ff is None:
                    raise ValueError("MMFF force field is None")
                ff.Minimize(maxIts=500)
                energies.append((float(ff.CalcEnergy()), int(cid), "MMFF94s"))
            except Exception:
                pass

    if not energies:
        for cid in conf_ids:
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=int(cid))
                if ff is None:
                    raise ValueError("UFF force field is None")
                ff.Minimize(maxIts=500)
                energies.append((float(ff.CalcEnergy()), int(cid), "UFF"))
            except Exception:
                pass

    if not energies:
        return [], "optimize_failed"

    energies.sort(key=lambda x: x[0])
    selected = energies[: max(1, int(select_conformers))]
    out_mols: List[Chem.Mol] = []
    for _, cid, method in selected:
        single = Chem.Mol(mol_h)
        conf = mol_h.GetConformer(int(cid))
        single.RemoveAllConformers()
        single.AddConformer(conf, assignId=True)
        single.SetProp("_OptimizationMethod", method)
        out_mols.append(single)
    return out_mols, "ok"


def safe_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def build_one(row: pd.Series, args: argparse.Namespace, row_seed: int) -> BuildResult:
    smiles = str(row[args.smiles_col]).strip()
    result = BuildResult(ok=False, smiles_input=smiles)
    result.activity_label = int(row.get(args.label_col, -1)) if not pd.isna(row.get(args.label_col, np.nan)) else None
    result.pmic_ug_ml = float(row.get(args.pmic_col, np.nan)) if not pd.isna(row.get(args.pmic_col, np.nan)) else None
    result.pmic_train = float(row.get(args.pmic_train_col, np.nan)) if not pd.isna(row.get(args.pmic_train_col, np.nan)) else None
    result.pmic_regression_mask = int(row.get(args.mask_col, 0)) if not pd.isna(row.get(args.mask_col, np.nan)) else 0
    result.generation_repeat = int(row.get(args.repeat_col, 1)) if not pd.isna(row.get(args.repeat_col, np.nan)) else 1

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result.reason = "invalid_smiles"
        return result

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        result.reason = "sanitize_failed"
        return result

    core, dropped, unsupported = largest_supported_core(mol, args.keep_supported_counterions)
    result.dropped_fragments = ";".join(dropped)
    result.unsupported_symbols = ";".join(unsupported)
    if core is None:
        result.reason = "no_supported_carbon_core"
        return result

    if args.neutralize_core:
        core = neutralize_atoms(core)

    us = unsupported_symbols(core)
    bc = bad_charges(core)
    if us:
        result.reason = "core_has_unsupported_atoms"
        result.unsupported_symbols = ";".join(us)
        return result
    if bc:
        result.reason = "core_has_unsupported_charges"
        result.unsupported_symbols = ";".join([f"charge_{c}" for c in bc])
        return result
    if core.GetNumAtoms() > args.max_atoms:
        result.reason = "too_many_atoms_before_h"
        return result

    result.smiles_core = canonical_smiles(core)
    result.scaffold = safe_scaffold(result.smiles_core)

    mols, status = embed_and_optimize(core, args.num_candidates, args.select_conformers, row_seed, args.rdkit_threads)
    if status != "ok" or not mols:
        result.reason = status
        return result

    for m in mols:
        if m.GetNumAtoms() > args.max_atoms:
            result.reason = "too_many_atoms_with_h"
            return result

    result.ok = True
    result.mols = mols
    result.reason = "ok"
    return result


def build_one_worker(payload: Tuple[int, Dict, Dict, int]) -> Tuple[int, BuildResult]:
    """Windows-safe multiprocessing worker."""
    idx, row_dict, args_dict, row_seed = payload
    args = argparse.Namespace(**args_dict)
    row = pd.Series(row_dict)
    return idx, build_one(row, args, row_seed)


def iter_build_tasks(df: pd.DataFrame, args: argparse.Namespace) -> Iterable[Tuple[int, Dict, Dict, int]]:
    args_dict = vars(args).copy()
    for local_i, (idx, row) in enumerate(df.iterrows()):
        yield int(idx), row.to_dict(), args_dict, int(args.seed + local_i)


def split_indices(df: pd.DataFrame, label_col: str, train_frac: float, val_frac: float, seed: int) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    groups = defaultdict(list)
    for idx, label in zip(df.index.tolist(), df[label_col].tolist()):
        groups[int(label)].append(idx)
    for _, idxs in groups.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        splits["train"].extend(idxs[:n_train])
        splits["val"].extend(idxs[n_train:n_train + n_val])
        splits["test"].extend(idxs[n_train + n_val:])
    for key in splits:
        rng.shuffle(splits[key])
    return splits


def load_raw_pickle(raw_dir: Path) -> Dict[str, List[Tuple[str, List[Chem.Mol]]]]:
    split_items: Dict[str, List[Tuple[str, List[Chem.Mol]]]] = {}
    for split in ["train", "val", "test"]:
        path = raw_dir / f"{split}_data.pickle"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw pickle: {path}")
        with open(path, "rb") as f:
            split_items[split] = pickle.load(f)
    return split_items


def save_raw_pickle(split_items: Dict[str, List[Tuple[str, List[Chem.Mol]]]], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split, items in split_items.items():
        with open(raw_dir / f"{split}_data.pickle", "wb") as f:
            pickle.dump(items, f)


def save_processed(split_items: Dict[str, List[Tuple[str, List[Chem.Mol]]]], processed_dir: Path) -> None:
    mol_to_torch_geometric, compute_all_statistics, save_statistics, save_pyg_dataset = import_utils_data()
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split, items in split_items.items():
        pyg = []
        for smiles, conformers in tqdm(items, desc=f"Convert {split} to PyG"):
            for mol in conformers:
                try:
                    data = mol_to_torch_geometric(mol, smiles, atom_encoder=PRETRAIN_ATOM_ENCODER)
                    pyg.append(data)
                except Exception as exc:
                    print(f"Warning PyG conversion failed for {smiles}: {exc}")
        if not pyg:
            raise RuntimeError(f"No PyG molecules for split {split}")
        save_pyg_dataset(pyg, processed_dir / f"{split}_h.pt")
        stats = compute_all_statistics(pyg, PRETRAIN_ATOM_ENCODER, additional_features=True)
        save_statistics(stats, processed_dir, split)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_path = Path(args.input)
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    processed_dir = output_root / "processed"
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.processed_only:
        split_items = load_raw_pickle(raw_dir)
        save_processed(split_items, processed_dir)
        print("Processed PyG files built from existing raw pickle files")
        print(f"Output root: {output_root.resolve()}")
        return

    df = pd.read_csv(input_path)
    required = [args.smiles_col, args.pmic_col, args.label_col, args.pmic_train_col, args.mask_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Run prepare_mrsa_scheme_a_table.py first")
    if args.repeat_col not in df.columns:
        df[args.repeat_col] = 1

    df = df[df[args.label_col].isin([0, 1])].copy()
    df = df[df[args.smiles_col].notna()].copy()
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()
    if args.test_mode:
        df = df.head(200).copy()

    splits_idx = split_indices(df, args.label_col, args.train_frac, args.val_frac, args.seed)
    idx_to_split = {}
    for split, idxs in splits_idx.items():
        for idx in idxs:
            idx_to_split[idx] = split

    split_items: Dict[str, List[Tuple[str, List[Chem.Mol]]]] = {"train": [], "val": [], "test": []}
    manifest_rows = []
    skipped = Counter()

    if args.num_workers > 1 and args.rdkit_threads == 0:
        print("Warning: --num_workers > 1 with --rdkit_threads 0 can oversubscribe CPU. Reset rdkit_threads to 1.")
        args.rdkit_threads = 1

    def handle_result(idx: int, res: BuildResult) -> None:
        split = idx_to_split.get(idx, "train")
        rep = 1

        if res.ok and res.mols is not None and res.smiles_core is not None:
            split_items[split].append((res.smiles_core, res.mols))
        else:
            skipped[res.reason] += 1

        manifest_rows.append({
            "input_index": int(idx),
            "split": split,
            "status": "kept" if res.ok else "skipped",
            "reason": res.reason,
            "input_smiles": res.smiles_input,
            "core_smiles": res.smiles_core,
            "activity_label": res.activity_label,
            "pmic_ug_ml": res.pmic_ug_ml,
            "pmic_train": res.pmic_train,
            "pmic_regression_mask": res.pmic_regression_mask,
            "generation_repeat_table": res.generation_repeat,
            "generation_repeat_applied": 1 if res.ok else 0,
            "training_set_policy": "fixed_once_no_weight_no_repeat",
            "dropped_fragments": res.dropped_fragments,
            "unsupported_symbols": res.unsupported_symbols,
            "scaffold": res.scaffold,
        })

    print("Training set policy: fixed once, no repeat, no sample weight, no activity filtering")
    print(f"RDKit conformer workers: {args.num_workers}")
    print(f"RDKit threads per worker: {args.rdkit_threads}")
    if args.num_workers <= 1:
        for idx, res in tqdm(
            (build_one_worker(task) for task in iter_build_tasks(df, args)),
            total=len(df),
            desc="Build conformers",
        ):
            handle_result(idx, res)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as executor:
            iterator = executor.map(
                build_one_worker,
                iter_build_tasks(df, args),
                chunksize=max(1, int(args.chunk_size)),
            )
            for idx, res in tqdm(iterator, total=len(df), desc="Build conformers"):
                handle_result(idx, res)

    if not split_items["train"] or not split_items["val"] or not split_items["test"]:
        raise RuntimeError({k: len(v) for k, v in split_items.items()})

    save_raw_pickle(split_items, raw_dir)
    if not args.raw_only:
        save_processed(split_items, processed_dir)
    else:
        print("Raw-only mode enabled. Skipped PyG .pt conversion.")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_root / "mrsa_scheme_a_manifest.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_csv": str(input_path.resolve()),
        "output_root": str(output_root.resolve()),
        "atom_encoder": PRETRAIN_ATOM_ENCODER,
        "allowed_charges": sorted(ALLOWED_CHARGES),
        "num_candidates": args.num_candidates,
        "select_conformers": args.select_conformers,
        "training_set_policy": "fixed_once_no_weight_no_repeat",
        "use_train_repeat": False,
        "sample_weight_used": False,
        "activity_filter_used": False,
        "num_workers": int(args.num_workers),
        "rdkit_threads": int(args.rdkit_threads),
        "chunk_size": int(args.chunk_size),
        "raw_only": bool(args.raw_only),
        "split_items": {k: len(v) for k, v in split_items.items()},
        "split_conformers": {k: int(sum(len(c) for _, c in v)) for k, v in split_items.items()},
        "skipped_reasons": dict(skipped),
    }
    (output_root / "mrsa_scheme_a_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("MRSA Scheme A geometry dataset built")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
