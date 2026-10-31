#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
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
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from megalodon.metrics.molecule import get_molecules
from megalodon.metrics.molecule_evaluation_callback import full_atom_decoder
from megalodon.metrics.molecule_stability_2d import is_valid_mol
from megalodon.models.module import Graph3DInterpolantModel

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust MRSA guided sampling with Scheme A predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--n_graphs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--save_path", default="outputs/compare_guided")
    parser.add_argument("--save_name", default="mrsa_candidates")
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compare_group", default="")
    parser.add_argument("--activity_threshold", type=float, default=0.5)
    parser.add_argument("--mic_floor", type=float, default=0.01)
    parser.add_argument("--mic_ceiling", type=float, default=100.0)

    parser.add_argument("--w_mic", type=float, default=0.40)
    parser.add_argument("--w_activity_prob", type=float, default=0.35)
    parser.add_argument("--w_qed", type=float, default=0.15)
    parser.add_argument("--w_lipinski", type=float, default=0.05)
    parser.add_argument("--w_alert", type=float, default=0.05)

    parser.add_argument("--save_raw_pickle", action="store_true")
    return parser.parse_args()


def load_model(ckpt_path: str, cfg):
    torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
    model = Graph3DInterpolantModel.load_from_checkpoint(
        ckpt_path,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
    )
    return model


def sample_molecules(model, cfg, args: argparse.Namespace) -> List[object]:
    mols = []
    scale_coords = float(OmegaConf.select(cfg, "evaluation.scale_coords", default=1.0))
    dataset_info = {"atom_decoder": full_atom_decoder}

    with torch.no_grad():
        while len(mols) < args.n_graphs:
            current = min(args.n_graphs - len(mols), args.batch_size)
            generated = model.sample(current, timesteps=args.timesteps)
            generated["x"] = scale_coords * generated["x"]
            batch_mols = get_molecules(generated, dataset_info)
            mols.extend(batch_mols)
            print(f"Generated molecule objects: {len(mols)}/{args.n_graphs}", flush=True)

    return mols


def mol_to_valid_smiles_and_mol(mol_obj) -> Optional[Tuple[str, Chem.Mol]]:
    candidates = []

    for attr in ("raw_rdkit_mol", "rdkit_mol", "mol"):
        try:
            value = getattr(mol_obj, attr, None)
            if value is not None:
                candidates.append(value)
        except Exception:
            pass

    if isinstance(mol_obj, Chem.Mol):
        candidates.append(mol_obj)

    for candidate in candidates:
        try:
            status = is_valid_mol(candidate)
            if status != 5:
                continue

            mol = Chem.Mol(candidate)
            try:
                mol.UpdatePropertyCache(strict=False)
            except Exception:
                pass

            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if smiles:
                return smiles, mol
        except Exception:
            continue

    return None


def featurize_mol(mol: Chem.Mol, n_bits: int, radius: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)

    desc = np.array(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
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
        Crippen.MolLogP(mol) <= 5,
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


def score_pairs(
    valid_pairs: List[Tuple[int, str, Chem.Mol]],
    predictor_bundle: Dict,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    clf = predictor_bundle["classifier"]
    reg = predictor_bundle["regressor"]
    n_bits = int(predictor_bundle.get("n_bits", 2048))
    radius = int(predictor_bundle.get("radius", 2))

    expected_dim = predictor_bundle.get("feature_dim", None)
    if expected_dim is None:
        expected_dim = getattr(clf, "n_features_in_", None)
    if expected_dim is None:
        expected_dim = getattr(reg, "n_features_in_", None)

    catalog = build_filter_catalog()

    rows = []
    failed_examples = []

    for source_index, smiles, mol3d in valid_pairs:
        try:
            mol2d = Chem.MolFromSmiles(smiles)
            if mol2d is None:
                raise ValueError("MolFromSmiles returned None")

            feat_1d = featurize_mol(mol2d, n_bits=n_bits, radius=radius)
            if expected_dim is not None and int(feat_1d.shape[0]) != int(expected_dim):
                raise ValueError(f"feature dimension mismatch got {feat_1d.shape[0]} expected {expected_dim}")

            feat = feat_1d.reshape(1, -1)
            activity_prob = float(clf.predict_proba(feat)[0, 1])
            pred_log10 = float(reg.predict(feat)[0])
            pred_pmic = float(10 ** pred_log10)

            q = float(QED.qed(mol2d))
            lip = lipinski_score(mol2d)
            alerts, alert_names = alert_count(mol2d, catalog)
            alert_score = 1.0 / (1.0 + float(alerts))
            mscore = mic_score(pred_pmic, args.mic_floor, args.mic_ceiling)

            weighted_mic_score = args.w_mic * mscore
            weighted_activity_score = args.w_activity_prob * activity_prob
            weighted_qed_score = args.w_qed * q
            weighted_lipinski_score = args.w_lipinski * lip
            weighted_alert_score = args.w_alert * alert_score

            total_score = (
                weighted_mic_score
                + weighted_activity_score
                + weighted_qed_score
                + weighted_lipinski_score
                + weighted_alert_score
            )

            rows.append(
                {
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
                    "weighted_mic_score": weighted_mic_score,
                    "weighted_activity_score": weighted_activity_score,
                    "weighted_qed_score": weighted_qed_score,
                    "weighted_lipinski_score": weighted_lipinski_score,
                    "weighted_alert_score": weighted_alert_score,
                    "mol_wt": float(Descriptors.MolWt(mol2d)),
                    "logp": float(Crippen.MolLogP(mol2d)),
                    "tpsa": float(rdMolDescriptors.CalcTPSA(mol2d)),
                    "hbd": int(rdMolDescriptors.CalcNumHBD(mol2d)),
                    "hba": int(rdMolDescriptors.CalcNumHBA(mol2d)),
                    "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol2d)),
                    "rings": int(rdMolDescriptors.CalcNumRings(mol2d)),
                    "total_score": float(total_score),
                    "passed_filter": int(pred_pmic < args.mic_ceiling and activity_prob >= args.activity_threshold and alerts == 0),
                    "source_index": int(source_index),
                    "_mol": mol3d,
                }
            )
        except Exception as exc:
            if len(failed_examples) < 10:
                failed_examples.append(
                    {
                        "source_index": int(source_index),
                        "smiles": smiles,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            continue

    all_scored = pd.DataFrame(rows)
    if len(all_scored) == 0:
        stats = {
            "valid_pairs": int(len(valid_pairs)),
            "scored_before_dedup": 0,
            "ranked_unique": 0,
            "expected_feature_dim": None if expected_dim is None else int(expected_dim),
            "failed_examples": failed_examples,
        }
        return pd.DataFrame(), pd.DataFrame(), stats

    all_scored = all_scored.sort_values("total_score", ascending=False).reset_index(drop=True)
    all_scored.insert(0, "raw_rank", np.arange(1, len(all_scored) + 1))

    ranked = all_scored.drop_duplicates(subset=["smiles"], keep="first").copy()
    ranked = ranked.sort_values("total_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    stats = {
        "valid_pairs": int(len(valid_pairs)),
        "scored_before_dedup": int(len(all_scored)),
        "ranked_unique": int(len(ranked)),
        "expected_feature_dim": None if expected_dim is None else int(expected_dim),
        "feature_dim_used": int(n_bits + 9),
        "failed_examples": failed_examples,
    }
    return ranked, all_scored, stats


def write_sdf(df: pd.DataFrame, out_path: Path, top_k: int) -> None:
    writer = Chem.SDWriter(str(out_path))
    written = 0

    for _, row in df.head(top_k).iterrows():
        mol = row.get("_mol")
        if mol is None:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
        else:
            try:
                mol = Chem.Mol(mol)
            except Exception:
                mol = Chem.MolFromSmiles(str(row["smiles"]))

        if mol is None:
            continue

        try:
            mol.SetProp("rank", str(int(row["rank"])))
            mol.SetProp("smiles", str(row["smiles"]))
            mol.SetProp("total_score", f"{float(row['total_score']):.6f}")
            mol.SetProp("pred_pmic_ug_ml", f"{float(row['pred_pmic_ug_ml']):.6g}")
            mol.SetProp("activity_label_probability", f"{float(row['activity_label_probability']):.6f}")
            writer.write(mol)
            written += 1
        except Exception:
            continue

    writer.close()
    print(f"SDF molecules written: {written} -> {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config_path)

    model = load_model(args.ckpt_path, cfg)
    model.to(args.device)
    model.eval()

    molecules = sample_molecules(model, cfg, args)

    valid_pairs = []
    invalid_count = 0
    for i, mol_obj in enumerate(molecules):
        rec = mol_to_valid_smiles_and_mol(mol_obj)
        if rec is None:
            invalid_count += 1
            continue
        smiles, mol3d = rec
        valid_pairs.append((i, smiles, mol3d))

    print(
        json.dumps(
            {
                "generated_objects": int(len(molecules)),
                "valid_raw_molecules": int(len(valid_pairs)),
                "invalid_raw_molecules": int(invalid_count),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    bundle = joblib.load(args.predictor)
    ranked, all_scored, score_stats = score_pairs(valid_pairs, bundle, args)

    print("Score stats:")
    print(json.dumps(score_stats, indent=2, ensure_ascii=False), flush=True)

    if len(ranked) == 0:
        debug_path = save_path / f"{args.save_name}_debug_no_valid.json"
        debug = {
            "compare_group": args.compare_group,
            "generated_objects": int(len(molecules)),
            "valid_raw_molecules": int(len(valid_pairs)),
            "invalid_raw_molecules": int(invalid_count),
            "score_stats": score_stats,
            "predictor_keys": list(bundle.keys()) if isinstance(bundle, dict) else str(type(bundle)),
            "weights": {
                "w_mic": args.w_mic,
                "w_activity_prob": args.w_activity_prob,
                "w_qed": args.w_qed,
                "w_lipinski": args.w_lipinski,
                "w_alert": args.w_alert,
            },
        }
        debug_path.write_text(json.dumps(debug, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(f"No valid generated molecules after scoring. Debug saved to {debug_path.resolve()}")

    hidden_cols_ranked = [c for c in ranked.columns if c.startswith("_")]
    hidden_cols_all = [c for c in all_scored.columns if c.startswith("_")]

    csv_path = save_path / f"{args.save_name}_ranked.csv"
    all_scored_csv_path = save_path / f"{args.save_name}_all_scored.csv"
    sdf_path = save_path / f"{args.save_name}_top{args.top_k}.sdf"

    ranked.drop(columns=hidden_cols_ranked).to_csv(csv_path, index=False, encoding="utf-8-sig")
    all_scored.drop(columns=hidden_cols_all).to_csv(all_scored_csv_path, index=False, encoding="utf-8-sig")
    write_sdf(ranked, sdf_path, args.top_k)

    summary = {
        "compare_group": args.compare_group,
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "config_path": str(Path(args.config_path).resolve()),
        "predictor": str(Path(args.predictor).resolve()),
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "activity_threshold": args.activity_threshold,
        "mic_floor": args.mic_floor,
        "mic_ceiling": args.mic_ceiling,
        "n_graphs_requested": args.n_graphs,
        "generated_objects": int(len(molecules)),
        "valid_raw_molecules": int(len(valid_pairs)),
        "invalid_raw_molecules": int(invalid_count),
        "all_scored": int(len(all_scored)),
        "valid_unique_scored": int(len(ranked)),
        "top_k": args.top_k,
        "weights": {
            "w_mic": args.w_mic,
            "w_activity_prob": args.w_activity_prob,
            "w_qed": args.w_qed,
            "w_lipinski": args.w_lipinski,
            "w_alert": args.w_alert,
        },
        "score_stats": score_stats,
        "csv": str(csv_path.resolve()),
        "all_scored_csv": str(all_scored_csv_path.resolve()),
        "sdf": str(sdf_path.resolve()),
    }

    summary_path = save_path / f"{args.save_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_raw_pickle:
        raw_path = save_path / f"{args.save_name}_raw_results.pkl"
        with open(raw_path, "wb") as f:
            pickle.dump(
                {
                    "summary": summary,
                    "molecules": molecules,
                    "valid_pairs_smiles": [x[1] for x in valid_pairs],
                },
                f,
            )

    print("Robust Scheme A sampling and ranking finished")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
