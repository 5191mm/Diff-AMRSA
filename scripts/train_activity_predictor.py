#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time

BOOT_TIME = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"[{BOOT_TIME}] BOOT: script entered before heavy imports", flush=True)
print(f"[{BOOT_TIME}] BOOT: python={sys.executable}", flush=True)
print(f"[{BOOT_TIME}] BOOT: argv={' '.join(sys.argv)}", flush=True)

def _sanitize_thread_env(name: str, default: str = "1") -> None:
    value = os.environ.get(name, "")
    if value.strip() == "":
        os.environ[name] = default
        return
    try:
        int(value)
    except Exception:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARN: invalid {name}={value!r}, reset to {default}", flush=True)
        os.environ[name] = default

for _env_name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    _sanitize_thread_env(_env_name, "1")

print(
    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] BOOT: threads "
    f"OMP={os.environ.get('OMP_NUM_THREADS')} "
    f"MKL={os.environ.get('MKL_NUM_THREADS')} "
    f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
    f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')}",
    flush=True,
)

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] IMPORT: stdlib", flush=True)
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] IMPORT: numpy pandas", flush=True)
import numpy as np
import pandas as pd

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] IMPORT: joblib", flush=True)
import joblib

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] IMPORT: sklearn", flush=True)
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] IMPORT: rdkit", flush=True)
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] IMPORT: all imports finished", flush=True)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Scheme A MRSA activity classifier and MIC regressor with visible progress logs."
    )
    parser.add_argument("--input", required=True, help="Scheme A CSV file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--pmic_train_col", default="pmic_train")
    parser.add_argument("--label_col", default="activity_label")
    parser.add_argument("--mask_col", default="pmic_regression_mask")

    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--progress_every", type=int, default=1000)
    parser.add_argument("--model_verbose", type=int, default=1)
    parser.add_argument("--model_type", choices=["extratrees", "randomforest"], default="extratrees")
    parser.add_argument("--max_rows", type=int, default=0, help="For debugging. 0 means use all rows")
    parser.add_argument("--import_only", action="store_true", help="Only test imports and exit")
    parser.add_argument("--save_features", action="store_true", help="Reserved. Disabled to avoid huge files")

    return parser.parse_args()


def mol_to_features(mol: Chem.Mol, radius: int, n_bits: int) -> np.ndarray:
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


def featurize_dataframe(
    df: pd.DataFrame,
    smiles_col: str,
    radius: int,
    n_bits: int,
    progress_every: int,
    tag: str,
) -> Tuple[np.ndarray, List[int], List[str]]:
    features: List[np.ndarray] = []
    kept_indices: List[int] = []
    canonical_smiles: List[str] = []

    total = len(df)
    t0 = time.time()
    log(f"[{tag}] featurization started. rows={total}, radius={radius}, n_bits={n_bits}")

    for k, (idx, value) in enumerate(df[smiles_col].items(), start=1):
        if progress_every > 0 and (k == 1 or k % progress_every == 0 or k == total):
            elapsed = max(time.time() - t0, 1e-9)
            rate = k / elapsed
            log(f"[{tag}] featurized input rows {k}/{total}, kept={len(features)}, rate={rate:.1f} rows/s")

        if value is None or pd.isna(value):
            continue
        smiles = str(value).strip()
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            features.append(mol_to_features(mol, radius=radius, n_bits=n_bits))
            kept_indices.append(idx)
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        except Exception:
            continue

    if not features:
        raise RuntimeError(f"[{tag}] no valid molecules after featurization")

    x = np.vstack(features).astype(np.float32)
    log(f"[{tag}] featurization finished. X shape={x.shape}, valid={len(kept_indices)}/{total}")
    return x, kept_indices, canonical_smiles


def make_classifier(args: argparse.Namespace):
    common = dict(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=args.model_verbose,
        class_weight="balanced",
    )
    if args.model_type == "randomforest":
        return RandomForestClassifier(**common)
    return ExtraTreesClassifier(**common)


def make_regressor(args: argparse.Namespace):
    common = dict(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=args.model_verbose,
    )
    if args.model_type == "randomforest":
        return RandomForestRegressor(**common)
    return ExtraTreesRegressor(**common)


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return None


def train_classifier(args: argparse.Namespace, df: pd.DataFrame) -> Tuple[object, Dict]:
    log("Classifier dataset preparation started")
    cls_df = df[[args.smiles_col, args.label_col]].copy()
    cls_df = cls_df.dropna(subset=[args.smiles_col, args.label_col])
    cls_df[args.label_col] = pd.to_numeric(cls_df[args.label_col], errors="coerce")
    cls_df = cls_df.dropna(subset=[args.label_col])
    cls_df[args.label_col] = cls_df[args.label_col].astype(int)
    cls_df = cls_df[cls_df[args.label_col].isin([0, 1])]

    log(f"Classifier rows after basic filter: {len(cls_df)}")
    log(f"Classifier label counts: {cls_df[args.label_col].value_counts().to_dict()}")

    x, kept_indices, _ = featurize_dataframe(
        cls_df,
        smiles_col=args.smiles_col,
        radius=args.radius,
        n_bits=args.n_bits,
        progress_every=args.progress_every,
        tag="classifier",
    )
    y = cls_df.loc[kept_indices, args.label_col].to_numpy(dtype=np.int64)

    counts = np.bincount(y, minlength=2)
    stratify = y if len(np.unique(y)) == 2 and counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = make_classifier(args)
    log(f"Training classifier started. model={args.model_type}, n_estimators={args.n_estimators}, n_jobs={args.n_jobs}")
    t0 = time.time()
    model.fit(x_train, y_train)
    log(f"Training classifier finished in {(time.time() - t0):.1f}s")

    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "label_counts_total": {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "roc_auc": safe_roc_auc(y_test, y_prob),
        "average_precision": safe_average_precision(y_test, y_prob),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    log(f"Classifier metrics: accuracy={metrics['accuracy']:.4f}, balanced_accuracy={metrics['balanced_accuracy']:.4f}, roc_auc={metrics['roc_auc']}")
    return model, metrics


def train_regressor(args: argparse.Namespace, df: pd.DataFrame) -> Tuple[object, Dict]:
    log("Regressor dataset preparation started")
    reg_df = df[[args.smiles_col, args.pmic_train_col, args.mask_col]].copy()
    reg_df[args.mask_col] = pd.to_numeric(reg_df[args.mask_col], errors="coerce").fillna(0).astype(int)
    reg_df[args.pmic_train_col] = pd.to_numeric(reg_df[args.pmic_train_col], errors="coerce")
    reg_df = reg_df.dropna(subset=[args.smiles_col, args.pmic_train_col])
    reg_df = reg_df[(reg_df[args.mask_col] == 1) & (reg_df[args.pmic_train_col] > 0)]

    log(f"Regressor rows after Scheme A mask filter: {len(reg_df)}")
    if len(reg_df) < 50:
        raise RuntimeError(f"Too few regression rows: {len(reg_df)}")

    x, kept_indices, _ = featurize_dataframe(
        reg_df,
        smiles_col=args.smiles_col,
        radius=args.radius,
        n_bits=args.n_bits,
        progress_every=args.progress_every,
        tag="regressor",
    )
    y_raw = reg_df.loc[kept_indices, args.pmic_train_col].to_numpy(dtype=np.float64)
    y = np.log10(np.clip(y_raw, 1e-12, None)).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = make_regressor(args)
    log(f"Training regressor started. target=log10({args.pmic_train_col}), model={args.model_type}, n_estimators={args.n_estimators}, n_jobs={args.n_jobs}")
    t0 = time.time()
    model.fit(x_train, y_train)
    log(f"Training regressor finished in {(time.time() - t0):.1f}s")

    y_pred = model.predict(x_test)
    raw_pred = np.power(10.0, y_pred)
    raw_true = np.power(10.0, y_test)

    metrics = {
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "target": f"log10({args.pmic_train_col})",
        "log10_mae": float(mean_absolute_error(y_test, y_pred)),
        "log10_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "log10_r2": float(r2_score(y_test, y_pred)),
        "pmic_ug_ml_mae": float(mean_absolute_error(raw_true, raw_pred)),
        "pmic_ug_ml_median_true": float(np.median(raw_true)),
        "pmic_ug_ml_median_pred": float(np.median(raw_pred)),
    }
    log(f"Regressor metrics: log10_mae={metrics['log10_mae']:.4f}, log10_r2={metrics['log10_r2']:.4f}")
    return model, metrics


def main() -> None:
    args = parse_args()

    if args.import_only:
        log("import_only requested. Imports succeeded. Exiting.")
        return

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"Reading CSV started: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    log(f"CSV loaded. rows={len(df)}, cols={list(df.columns)}")

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()
        log(f"Debug max_rows applied. rows={len(df)}")

    required = [args.smiles_col, args.pmic_train_col, args.label_col, args.mask_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    log("Training classifier")
    classifier, classifier_metrics = train_classifier(args, df)

    log("Training regressor")
    regressor, regressor_metrics = train_regressor(args, df)

    package = {
        "version": "scheme_a_verbose_boot_v2",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classifier": classifier,
        "activity_classifier": classifier,
        "regressor": regressor,
        "mic_regressor": regressor,
        "radius": args.radius,
        "n_bits": args.n_bits,
        "feature_dim": args.n_bits + 9,
        "target_transform": "log10_pmic_train",
        "smiles_col": args.smiles_col,
        "pmic_train_col": args.pmic_train_col,
        "label_col": args.label_col,
        "mask_col": args.mask_col,
        "model_type": args.model_type,
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs,
        "random_state": args.random_state,
        "metrics": {
            "classifier": classifier_metrics,
            "regressor": regressor_metrics,
        },
    }

    model_path = outdir / "mrsa_activity_predictor_schemeA.joblib"
    metrics_path = outdir / "mrsa_activity_predictor_schemeA_metrics.json"

    log(f"Saving predictor to {model_path}")
    joblib.dump(package, model_path)

    log(f"Saving metrics to {metrics_path}")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(package["metrics"], f, indent=2, ensure_ascii=False)

    if args.save_features:
        log("save_features requested but full feature saving is disabled in this stable script to avoid huge files")

    log("Scheme A predictor trained successfully")
    log(f"Predictor: {model_path}")
    log(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user")
        raise
    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        raise
