#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild Megalodon processed .pt files from existing raw pickle files and attach adaptive sample weights.

This script does not regenerate conformers and does not change the training set.
It only rewrites processed/train_h.pt val_h.pt test_h.pt with metadata fields:
sample_weight pmic_bin_id activity_label pmic_train pmic_regression_mask.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data.collate import collate
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

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

BIN_TO_ID = {
    "inactive": 0,
    "mic_0_1": 1,
    "mic_1_10": 2,
    "mic_10_100": 3,
    "unknown": 4,
}

DEFAULT_WEIGHTS = {
    "inactive": 1.0,
    "mic_0_1": 1.0,
    "mic_1_10": 1.0,
    "mic_10_100": 1.0,
    "unknown": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--scheme_csv", type=str, required=False, default=None)
    parser.add_argument("--manifest", type=str, required=False, default=None)
    parser.add_argument("--bin_weights", type=str, default=None)
    parser.add_argument("--processed_folder", type=str, default="processed")
    parser.add_argument("--backup_existing", action="store_true")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--label_col", type=str, default="activity_label")
    parser.add_argument("--pmic_train_col", type=str, default="pmic_train")
    parser.add_argument("--mask_col", type=str, default="pmic_regression_mask")
    parser.add_argument("--strict_manifest", action="store_true")
    return parser.parse_args()


def import_utils_data():
    from utils_data import (
        mol_to_torch_geometric, compute_all_statistics, save_statistics
    )
    return mol_to_torch_geometric, compute_all_statistics, save_statistics


def save_pyg_dataset(pyg_batch: List[Any], save_path: Path) -> None:
    collated_data = collate(pyg_batch[0].__class__, pyg_batch, increment=False, add_batch=False)
    torch.save(collated_data[:2], save_path)


def load_raw_pickle(raw_dir: Path) -> Dict[str, List[Tuple[str, List[Chem.Mol]]]]:
    out = {}
    for split in ["train", "val", "test"]:
        path = raw_dir / f"{split}_data.pickle"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw pickle: {path}")
        with open(path, "rb") as f:
            out[split] = pickle.load(f)
    return out


def normalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles).strip()
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def bin_from_values(label: int, pmic_train: float, mask: int) -> str:
    if int(label) == 0 or int(mask) == 0 or pmic_train is None or (isinstance(pmic_train, float) and math.isnan(pmic_train)):
        return "inactive"
    x = float(pmic_train)
    if x <= 1.0:
        return "mic_0_1"
    if x <= 10.0:
        return "mic_1_10"
    if x < 100.0:
        return "mic_10_100"
    return "inactive"


def load_weights(path: str | None) -> Dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if path is None:
        return weights
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    obj = json.loads(p.read_text(encoding="utf-8"))
    bins = obj.get("bins", obj)
    for key in weights:
        val = bins.get(key, weights[key])
        if isinstance(val, dict):
            val = val.get("weight", weights[key])
        weights[key] = float(val)
    return weights


def choose_better(old: Dict[str, Any] | None, new: Dict[str, Any]) -> Dict[str, Any]:
    if old is None:
        return new
    old_label = int(old.get("activity_label", 0))
    new_label = int(new.get("activity_label", 0))
    if new_label > old_label:
        return new
    if new_label < old_label:
        return old
    old_p = old.get("pmic_train", np.nan)
    new_p = new.get("pmic_train", np.nan)
    old_is_nan = pd.isna(old_p)
    new_is_nan = pd.isna(new_p)
    if old_is_nan and not new_is_nan:
        return new
    if new_is_nan and not old_is_nan:
        return old
    if not old_is_nan and not new_is_nan and float(new_p) < float(old_p):
        return new
    return old


def build_metadata_from_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    df = pd.read_csv(manifest_path)
    if "core_smiles" not in df.columns:
        raise ValueError("Manifest must contain core_smiles")
    df = df[df.get("status", "kept") == "kept"].copy() if "status" in df.columns else df.copy()
    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        core = row.get("core_smiles", None)
        if pd.isna(core) or not str(core).strip():
            continue
        key = normalize_smiles(core)
        label = int(row.get("activity_label", 0)) if not pd.isna(row.get("activity_label", np.nan)) else 0
        pmic_train = row.get("pmic_train", np.nan)
        mask = int(row.get("pmic_regression_mask", 0)) if not pd.isna(row.get("pmic_regression_mask", np.nan)) else 0
        item = {
            "core_smiles": key,
            "activity_label": label,
            "pmic_train": float(pmic_train) if not pd.isna(pmic_train) else np.nan,
            "pmic_regression_mask": mask,
        }
        item["pmic_bin"] = bin_from_values(item["activity_label"], item["pmic_train"], item["pmic_regression_mask"])
        mapping[key] = choose_better(mapping.get(key), item)
    return mapping


def attach_metadata(data, meta: Dict[str, Any], weight: float) -> Any:
    label = int(meta.get("activity_label", 0))
    pmic_train = meta.get("pmic_train", np.nan)
    mask = int(meta.get("pmic_regression_mask", 0))
    bin_name = str(meta.get("pmic_bin", "unknown"))
    data.sample_weight = torch.tensor([float(weight)], dtype=torch.float32)
    data.activity_label = torch.tensor([label], dtype=torch.long)
    data.pmic_train = torch.tensor([float(pmic_train) if not pd.isna(pmic_train) else float("nan")], dtype=torch.float32)
    data.pmic_regression_mask = torch.tensor([mask], dtype=torch.long)
    data.pmic_bin_id = torch.tensor([int(BIN_TO_ID.get(bin_name, BIN_TO_ID["unknown"]))], dtype=torch.long)
    return data


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    processed_dir = output_root / args.processed_folder
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else output_root / "mrsa_scheme_a_manifest.csv"
    metadata = build_metadata_from_manifest(manifest_path)
    if not metadata:
        raise RuntimeError("No metadata loaded from manifest")

    weights = load_weights(args.bin_weights)

    if args.backup_existing and processed_dir.exists() and any(processed_dir.iterdir()):
        backup = processed_dir.with_name(processed_dir.name + "_backup_before_adaptive")
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(processed_dir, backup)
        print(f"Backed up existing processed folder to: {backup}")

    mol_to_torch_geometric, compute_all_statistics, save_statistics = import_utils_data()
    split_items = load_raw_pickle(raw_dir)
    summary = {"weights": weights, "splits": {}}

    for split, items in split_items.items():
        pyg = []
        bin_counts = defaultdict(int)
        missing_meta = 0
        for smiles, conformers in tqdm(items, desc=f"Convert {split} with adaptive weights"):
            key = normalize_smiles(smiles)
            meta = metadata.get(key)
            if meta is None:
                missing_meta += 1
                if args.strict_manifest:
                    raise KeyError(f"Missing metadata for core_smiles={key}")
                meta = {
                    "activity_label": 0,
                    "pmic_train": np.nan,
                    "pmic_regression_mask": 0,
                    "pmic_bin": "unknown",
                }
            bin_name = str(meta.get("pmic_bin", "unknown"))
            weight = weights.get(bin_name, 1.0) if split == "train" else 1.0
            bin_counts[bin_name] += len(conformers)
            for mol in conformers:
                data = mol_to_torch_geometric(mol, smiles, atom_encoder=PRETRAIN_ATOM_ENCODER)
                data = attach_metadata(data, meta, weight)
                pyg.append(data)
        if not pyg:
            raise RuntimeError(f"No PyG data for split {split}")
        save_pyg_dataset(pyg, processed_dir / f"{split}_h.pt")
        stats = compute_all_statistics(pyg, PRETRAIN_ATOM_ENCODER, additional_features=True)
        save_statistics(stats, processed_dir, split)
        summary["splits"][split] = {
            "graphs": len(pyg),
            "bin_counts": dict(bin_counts),
            "missing_metadata": int(missing_meta),
        }

    out_json = processed_dir / "adaptive_weight_processed_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Adaptive weighted processed files written")
    print(f"Processed dir: {processed_dir.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
