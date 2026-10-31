#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample Megalodon molecules and rank with Scheme A predictor.

Output meaning
- pred_pmic_ug_ml is predicted active MIC from the active only regressor
- activity_label is predicted activity probability for generated molecules
- activity_label_binary is thresholded probability
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import omegaconf
from omegaconf import OmegaConf
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from megalodon.data.molecule_datamodule import MoleculeDataModule
from megalodon.data.statistics import Statistics
from megalodon.metrics.molecule_evaluation_callback import MoleculeEvaluationCallback
from megalodon.models.module import Graph3DInterpolantModel

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and score MRSA candidates with Scheme A predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--predictor", type=str, required=True)
    parser.add_argument("--n_graphs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="mrsa_generated_schemeA")
    parser.add_argument("--save_name", type=str, default="mrsa_candidates_schemeA")
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--statistics_split", type=str, default="train")
    parser.add_argument("--activity_threshold", type=float, default=0.5)
    parser.add_argument("--w_mic", type=float, default=0.40)
    parser.add_argument("--w_activity_prob", type=float, default=0.35)
    parser.add_argument("--w_qed", type=float, default=0.15)
    parser.add_argument("--w_lipinski", type=float, default=0.05)
    parser.add_argument("--w_alert", type=float, default=0.05)
    parser.add_argument("--mic_floor", type=float, default=0.01)
    parser.add_argument("--mic_ceiling", type=float, default=100.0)
    parser.add_argument("--save_raw_pickle", action="store_true")
    parser.add_argument("--compute_3d_metrics", action="store_true",
                        help="Enable 3D geometry metrics during sampling. Disabled by default for robust MRSA ranking.")
    parser.add_argument("--compare_group", type=str, default="",
                        help="Optional group name such as pretrained_baseline, scratch_mrsa or finetune_mrsa")
    return parser.parse_args()


def featurize_mol(mol: Chem.Mol, n_bits: int, radius: int) -> np.ndarray:
    """
    Must match scripts/train_activity_predictor.py exactly.

    Feature layout:
    Morgan fingerprint n_bits
    + 9 scaled descriptors
    = n_bits + 9
    """
    arr = np.zeros((n_bits,), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)

    desc = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            mol.GetNumHeavyAtoms(),
        ],
        dtype=np.float32,
    )

    desc_scaled = desc.copy()
    desc_scaled[0] = desc_scaled[0] / 1000.0
    desc_scaled[2] = desc_scaled[2] / 300.0
    desc_scaled[3:] = desc_scaled[3:] / 50.0

    return np.concatenate([arr, desc_scaled], axis=0)

def load_model(ckpt_path: str, cfg):
    torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
    model = Graph3DInterpolantModel.load_from_checkpoint(
        ckpt_path,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
    )
    return model


def build_filter_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


def alert_count(mol: Chem.Mol, catalog: FilterCatalog) -> Tuple[int, str]:
    try:
        matches = catalog.GetMatches(mol)
        names = [m.GetDescription() for m in matches]
        return len(names), ";".join(names[:20])
    except Exception:
        return 0, ""


def lipinski_score(mol: Chem.Mol) -> float:
    rules = [
        Descriptors.MolWt(mol) <= 500,
        Descriptors.MolLogP(mol) <= 5,
        Lipinski.NumHDonors(mol) <= 5,
        Lipinski.NumHAcceptors(mol) <= 10,
        rdMolDescriptors.CalcTPSA(mol) <= 140,
    ]
    return float(sum(rules) / len(rules))


def mic_score(pred_pmic: float, floor: float, ceiling: float) -> float:
    if pred_pmic <= 0 or not np.isfinite(pred_pmic):
        return 0.0
    log_floor = np.log10(floor)
    log_ceil = np.log10(ceiling)
    val = (log_ceil - np.log10(pred_pmic)) / (log_ceil - log_floor)
    return float(np.clip(val, 0.0, 1.0))


def get_rdkit_mol(obj) -> Optional[Chem.Mol]:
    candidates = []

    if isinstance(obj, Chem.Mol):
        candidates.append(obj)

    for attr in ("raw_rdkit_mol", "rdkit_mol", "mol"):
        try:
            if hasattr(obj, attr):
                candidate = getattr(obj, attr)
                if candidate is not None:
                    candidates.append(candidate)
        except Exception:
            pass

    for candidate in candidates:
        try:
            mol = Chem.Mol(candidate)
            if mol is None or mol.GetNumAtoms() == 0:
                continue

            try:
                mol.UpdatePropertyCache(strict=False)
            except Exception:
                pass

            try:
                Chem.SanitizeMol(mol)
                return mol
            except Exception:
                pass

            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                )
                return mol
            except Exception:
                pass

            try:
                smiles = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)
                mol2 = Chem.MolFromSmiles(smiles)
                if mol2 is not None:
                    return mol2
            except Exception:
                pass

        except Exception:
            continue

    return None

def score_molecules(molecules: List[object], predictor_bundle: Dict, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clf = predictor_bundle["classifier"]
    reg = predictor_bundle["regressor"]
    n_bits = int(predictor_bundle["n_bits"])
    radius = int(predictor_bundle["radius"])
    catalog = build_filter_catalog()

    rows = []
    stats = {
        "input_objects": int(len(molecules)),
        "rdkit_mol_found": 0,
        "scored_before_dedup": 0,
        "scoring_failed": 0,
        "duplicate_smiles": 0,
        "unique_after_dedup": 0,
    }

    for i, obj in enumerate(molecules):
        mol = get_rdkit_mol(obj)
        if mol is None:
            continue

        stats["rdkit_mol_found"] += 1

        try:
            mol_no_h = Chem.RemoveHs(Chem.Mol(mol), sanitize=False)
            try:
                mol_no_h.UpdatePropertyCache(strict=False)
            except Exception:
                pass

            try:
                Chem.SanitizeMol(mol_no_h)
            except Exception:
                Chem.SanitizeMol(
                    mol_no_h,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                )

            smiles = Chem.MolToSmiles(mol_no_h, canonical=True, isomericSmiles=True)
            if not smiles:
                stats["scoring_failed"] += 1
                continue

            feat_1d = featurize_mol(mol_no_h, n_bits, radius)
            if expected_dim is not None and int(expected_dim) != int(feat_1d.shape[0]):
                raise ValueError(f"Feature dimension mismatch: got {feat_1d.shape[0]}, expected {expected_dim}")
            feat = feat_1d.reshape(1, -1)
            activity_prob = float(clf.predict_proba(feat)[0, 1])
            pred_log10 = float(reg.predict(feat)[0])
            pred_pmic = float(10 ** pred_log10)

            q = float(QED.qed(mol_no_h))
            lip = lipinski_score(mol_no_h)
            alerts, alert_names = alert_count(mol_no_h, catalog)
            alert_score = 1.0 / (1.0 + float(alerts))
            mscore = mic_score(pred_pmic, args.mic_floor, args.mic_ceiling)

            total = (
                args.w_mic * mscore
                + args.w_activity_prob * activity_prob
                + args.w_qed * q
                + args.w_lipinski * lip
                + args.w_alert * alert_score
            )

            stats["scored_rows"] += 1
            rows.append({
                "smiles": smiles,
                "pred_pmic_ug_ml": pred_pmic,
                "pred_log10_pmic": pred_log10,
                "activity_label": activity_prob,
                "activity_label_probability": activity_prob,
                "activity_label_binary": int(activity_prob >= args.activity_threshold),
                "mic_score": mscore,
                "qed": q,
                "lipinski_score": lip,
                "alert_count": int(alerts),
                "alert_score": alert_score,
                "alert_names": alert_names,
                "mol_wt": float(Descriptors.MolWt(mol_no_h)),
                "logp": float(Descriptors.MolLogP(mol_no_h)),
                "tpsa": float(rdMolDescriptors.CalcTPSA(mol_no_h)),
                "hbd": int(Lipinski.NumHDonors(mol_no_h)),
                "hba": int(Lipinski.NumHAcceptors(mol_no_h)),
                "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol_no_h)),
                "rings": int(rdMolDescriptors.CalcNumRings(mol_no_h)),
                "total_score": float(total),
                "passed_filter": int(pred_pmic < args.mic_ceiling and activity_prob >= args.activity_threshold and alerts == 0),
                "source_index": i,
                "_mol": mol,
            })

        except Exception:
            stats["scoring_failed"] += 1
            continue

    if not rows:
        print("Score stats:", json.dumps(stats, ensure_ascii=False, indent=2))
        return pd.DataFrame(), pd.DataFrame()

    all_scored = pd.DataFrame(rows)
    all_scored = all_scored.sort_values("total_score", ascending=False).reset_index(drop=True)
    all_scored.insert(0, "raw_rank", np.arange(1, len(all_scored) + 1))

    duplicate_count = int(all_scored.duplicated(subset=["smiles"]).sum())
    stats["scored_before_dedup"] = int(len(all_scored))
    stats["duplicate_smiles"] = duplicate_count

    ranked = all_scored.drop_duplicates(subset=["smiles"], keep="first").copy()
    ranked = ranked.sort_values("total_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    stats["unique_after_dedup"] = int(len(ranked))
    print("Score stats:", json.dumps(stats, ensure_ascii=False, indent=2))

    return ranked, all_scored


def write_sdf(df: pd.DataFrame, out_path: Path, top_k: int) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.head(top_k).iterrows():
            mol = row.get("_mol")
            if mol is None:
                continue
            try:
                mol.SetProp("rank", str(int(row["rank"])))
                mol.SetProp("smiles", str(row["smiles"]))
                mol.SetProp("total_score", f"{row['total_score']:.6f}")
                mol.SetProp("pred_pmic_ug_ml", f"{row['pred_pmic_ug_ml']:.6g}")
                mol.SetProp("activity_label_probability", f"{row['activity_label_probability']:.6f}")
                block = Chem.MolToMolBlock(mol, kekulize=False)
                props = []
                for key in ["rank", "smiles", "total_score", "pred_pmic_ug_ml", "activity_label_probability"]:
                    props.append(f">  <{key}>\n{mol.GetProp(key)}\n")
                f.write(block + "\n" + "\n".join(props) + "\n$$$$\n")
            except Exception:
                continue


def main() -> None:
    args = parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config_path)
    model = load_model(args.ckpt_path, cfg)
    model.to(args.device)
    model.eval()

    datamodule = MoleculeDataModule(
        dataset_root=cfg.data.dataset_root,
        processed_folder=cfg.data.processed_folder,
        batch_size=args.batch_size,
        inference_batch_size=args.batch_size,
        data_loader_type="midi",
    )
    try:
        statistics = Statistics.load_statistics(
            statistics_dir=f"{cfg.data.dataset_root}/{datamodule.processed_folder}",
            split_name=args.statistics_split,
        )
    except Exception as exc:
        print(f"Warning statistics load failed: {exc}")
        statistics = None

    eval_callback = MoleculeEvaluationCallback(
        n_graphs=args.n_graphs,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        compute_train_data_metrics=False,
        compute_3D_metrics=args.compute_3d_metrics,
        compute_energy_metrics=False,
        energy_metrics_args=None,
        statistics=statistics,
        scale_coords=cfg.evaluation.scale_coords,
        preserve_aromatic=OmegaConf.select(cfg.evaluation, "preserve_aromatic", default=True),
    )

    result = eval_callback.evaluate_molecules(model, return_molecules=True)
    molecules = result.get("molecules", [])
    bundle = joblib.load(args.predictor)
    ranked, all_scored = score_molecules(molecules, bundle, args)

    hidden_cols_all = [c for c in all_scored.columns if c.startswith("_")] if len(all_scored) else []
    all_scored_csv_path = save_path / f"{args.save_name}_all_scored.csv"
    if len(all_scored):
        all_scored.drop(columns=hidden_cols_all).to_csv(all_scored_csv_path, index=False, encoding="utf-8-sig")

    if len(ranked) == 0:
        debug_path = save_path / f"{args.save_name}_debug_no_valid.txt"
        debug_lines = [
            f"molecule_count={len(molecules)}",
            "No valid generated molecules after scoring",
            "Check whether generated molecule objects contain rdkit_mol",
        ]
        for i, obj in enumerate(molecules[:20]):
            debug_lines.append(f"object_{i}_type={type(obj)}")
            debug_lines.append(f"object_{i}_attrs={sorted([a for a in dir(obj) if not a.startswith('__')])[:80]}")
        debug_path.write_text("\n".join(debug_lines), encoding="utf-8")
        raise RuntimeError(f"No valid generated molecules after scoring. Debug saved to {debug_path}")

    hidden_cols = [c for c in ranked.columns if c.startswith("_")]
    csv_path = save_path / f"{args.save_name}_ranked.csv"
    ranked.drop(columns=hidden_cols).to_csv(csv_path, index=False, encoding="utf-8-sig")

    sdf_path = save_path / f"{args.save_name}_top{args.top_k}.sdf"
    write_sdf(ranked, sdf_path, args.top_k)

    summary = {
        "compare_group": args.compare_group,
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "config_path": str(Path(args.config_path).resolve()),
        "predictor": str(Path(args.predictor).resolve()),
        "statistics_split": args.statistics_split,
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "activity_threshold": args.activity_threshold,
        "mic_floor": args.mic_floor,
        "mic_ceiling": args.mic_ceiling,
        "n_graphs_requested": args.n_graphs,
        "valid_unique_scored": int(len(ranked)),
        "top_k": args.top_k,
        "weights": {
            "w_mic": args.w_mic,
            "w_activity_prob": args.w_activity_prob,
            "w_qed": args.w_qed,
            "w_lipinski": args.w_lipinski,
            "w_alert": args.w_alert,
        },
        "csv": str(csv_path.resolve()),
        "all_scored_csv": str(all_scored_csv_path.resolve()),
        "sdf": str(sdf_path.resolve()),
    }
    summary_path = save_path / f"{args.save_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_raw_pickle:
        with open(save_path / f"{args.save_name}_raw_results.pkl", "wb") as f:
            pickle.dump(result, f)

    print("Scheme A sampling and ranking finished")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
