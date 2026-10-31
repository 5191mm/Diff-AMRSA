# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

BASE_DIR = Path(os.environ.get("DIFF_AMRSA_TABLE_DIR", Path(__file__).resolve().parent.parent)).expanduser().resolve()

INPUT_FILES = [
    "1-文献跟踪数据.xlsx",
    "2-课题组无活性数据.xlsx",
    "ChEMBL1.csv",
    "PubMed数据确认汇总.xlsx",
]

OUTPUT_CSV = "processed_drug_mic_dataset.csv"
OUTPUT_XLSX = "processed_drug_mic_dataset.xlsx"
WRITE_XLSX = False
REMOVED_LOG_CSV = "removed_rows_log.csv"
SUMMARY_CSV = "processing_summary.csv"

OUTPUT_COLUMNS = [
    "smiles",
    "concentration_value",
    "concentration_unit",
    "pmic_ug_ml",
    "activity_label",
]

STRICT_MIC_ONLY = True
REMOVE_RELATION_ROWS = True
DEDUP_BY_CANONICAL_SMILES_KEEP_LOWEST_UG_ML = True
ACTIVE_THRESHOLD_UG_ML = 100.0
N_JOBS = min(4, max(1, os.cpu_count() or 1))
MIN_PARALLEL_UNIQUE_SMILES = 2000

SMILES_CANDIDATES = [
    "smiles",
    "smile",
    "canonical_smiles",
    "canonical smiles",
    "pubchem_ext_datasource_smiles",
    "pubchem ext datasource smiles",
]
TYPE_CANDIDATES = ["standard type", "type", "activity type", "assay type"]
VALUE_CANDIDATES = ["standard value", "value", "activity value", "concentration", "mic"]
UNIT_CANDIDATES = ["standard units", "standard unit", "units", "unit", "concentration unit"]
RELATION_CANDIDATES = ["standard relation", "relation", "standard qualifier", "qualifier"]

RELATION_PATTERN = re.compile(r"[<>≤≥]")
NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def normalize_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = text.replace("µ", "u").replace("μ", "u")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    normalized = {normalize_name(col): col for col in df.columns}
    candidate_set = {normalize_name(item) for item in candidates}

    for cand in candidate_set:
        if cand in normalized:
            return normalized[cand]

    for norm_col, original_col in normalized.items():
        for cand in candidate_set:
            if cand in norm_col:
                return original_col
    return None


def read_input_file(path: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    if path.suffix.lower() in [".xlsx", ".xls"]:
        excel = pd.ExcelFile(path)
        for sheet_name in excel.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
            df["__source_file"] = path.name
            df["__source_sheet"] = sheet_name
            df["__source_row"] = df.index + 2
            frames.append(df)
    elif path.suffix.lower() == ".csv":
        encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
        last_error: Optional[Exception] = None
        for encoding in encodings:
            try:
                df = pd.read_csv(path, dtype=object, encoding=encoding, low_memory=False)
                df["__source_file"] = path.name
                df["__source_sheet"] = ""
                df["__source_row"] = df.index + 2
                frames.append(df)
                break
            except Exception as exc:
                last_error = exc
        if not frames:
            raise RuntimeError(f"Failed to read CSV file {path}: {last_error}")
    else:
        raise RuntimeError(f"Unsupported input file type: {path}")
    return frames


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def parse_positive_number(value: Any) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return None
    if RELATION_PATTERN.search(text):
        return None
    text = text.replace(",", "")
    text = text.strip()
    if not NUMBER_PATTERN.match(text):
        return None
    number = float(text)
    if number <= 0:
        return None
    return number


def normalize_unit(unit: Any) -> str:
    text = clean_text(unit)
    text = text.replace("µ", "u").replace("μ", "u")
    text = text.replace("·", ".").replace("/", ".")
    text = text.replace(" ", "")
    text = text.lower()
    text = text.replace("per", ".")
    text = text.replace("-1", "-1")
    return text


def convert_to_ug_ml(value: float, unit: Any, mol_weight: float) -> Optional[float]:
    unit_norm = normalize_unit(unit)

    ug_ml_units = {
        "ug.ml-1",
        "ug.ml^-1",
        "ugml-1",
        "ug.ml",
        "ugml",
        "ug.mL-1".lower(),
        "ug/ml".replace("/", "."),
        "microgram.ml-1",
        "micrograms.ml-1",
        "mcg.ml-1",
        "mcgml-1",
    }
    if unit_norm in ug_ml_units:
        return value

    if unit_norm in {"ng.ml-1", "ngml-1", "ng.ml", "ngml"}:
        return value / 1000.0

    if unit_norm in {"mg.ml-1", "mgml-1", "mg.ml", "mgml"}:
        return value * 1000.0

    if unit_norm in {"mg.l-1", "mgl-1", "mg.l", "mgl"}:
        return value

    if unit_norm in {"g.l-1", "gl-1", "g.l", "gl"}:
        return value * 1000.0

    if unit_norm in {"nm", "nanomolar", "nanomol.l-1", "nmol.l-1"}:
        return value * mol_weight / 1_000_000.0

    if unit_norm in {"um", "umol.l-1", "umol.l", "micromolar", "micromol.l-1"}:
        return value * mol_weight / 1000.0

    if unit_norm in {"mm", "mmol.l-1", "mmol.l", "millimolar", "millimol.l-1"}:
        return value * mol_weight

    if unit_norm in {"m", "mol.l-1", "mol.l", "molar"}:
        return value * mol_weight * 1000.0

    if unit_norm in {"pm", "pmol.l-1", "pmol.l", "picomolar", "picomol.l-1"}:
        return value * mol_weight / 1_000_000_000.0

    return None


def standardize_molecule(smiles: Any) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    text = clean_text(smiles)
    if not text:
        return None, None, "missing_smiles"

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None, None, "invalid_smiles"

    mol_weight = float(Descriptors.MolWt(mol))
    if mol_weight <= 0:
        return None, None, "invalid_mol_weight"

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return canonical_smiles, mol_weight, None

def standardize_molecule_for_pool(smiles: str) -> Tuple[str, Tuple[Optional[str], Optional[float], Optional[str]]]:
    return smiles, standardize_molecule(smiles)


def standardize_many_smiles(smiles_values: pd.Series) -> Dict[str, Tuple[Optional[str], Optional[float], Optional[str]]]:
    smiles_list = [clean_text(item) for item in smiles_values.drop_duplicates().tolist()]
    if N_JOBS <= 1 or len(smiles_list) < MIN_PARALLEL_UNIQUE_SMILES:
        return {smiles: standardize_molecule(smiles) for smiles in smiles_list}

    cache: Dict[str, Tuple[Optional[str], Optional[float], Optional[str]]] = {}
    with Pool(processes=N_JOBS) as pool:
        for smiles, result in pool.imap_unordered(standardize_molecule_for_pool, smiles_list, chunksize=200):
            cache[smiles] = result
    return cache


def is_missing_key(*values: Any) -> bool:
    return any(clean_text(value) == "" for value in values)


def relation_has_less_or_greater(value: Any) -> bool:
    text = clean_text(value)
    if not text:
        return False
    return bool(RELATION_PATTERN.search(text))



def make_removed_records(rows: pd.DataFrame, reason_col: str = "reason") -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if rows.empty:
        return records
    cols = [
        "source_file",
        "source_sheet",
        "source_row",
        reason_col,
        "raw_smiles",
        "raw_type",
        "raw_value",
        "raw_unit",
        "raw_relation",
    ]
    for item in rows[cols].itertuples(index=False, name=None):
        records.append(
            {
                "source_file": item[0],
                "source_sheet": item[1],
                "source_row": item[2],
                "reason": item[3],
                "raw_smiles": item[4],
                "raw_type": item[5],
                "raw_value": item[6],
                "raw_unit": item[7],
                "raw_relation": item[8],
            }
        )
    return records


def build_standard_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    smiles_col = find_column(df, SMILES_CANDIDATES)
    type_col = find_column(df, TYPE_CANDIDATES)
    value_col = find_column(df, VALUE_CANDIDATES)
    unit_col = find_column(df, UNIT_CANDIDATES)
    relation_col = find_column(df, RELATION_CANDIDATES)

    removed: List[Dict[str, Any]] = []
    output_rows: List[Dict[str, Any]] = []

    required = {
        "smiles_col": smiles_col,
        "type_col": type_col,
        "value_col": value_col,
        "unit_col": unit_col,
    }
    missing_required = [name for name, col in required.items() if col is None]
    if missing_required:
        source_file = clean_text(df["__source_file"].iloc[0]) if "__source_file" in df.columns and len(df) else ""
        source_sheet = clean_text(df["__source_sheet"].iloc[0]) if "__source_sheet" in df.columns and len(df) else ""
        removed.append(
            {
                "source_file": source_file,
                "source_sheet": source_sheet,
                "source_row": "",
                "reason": "missing_required_columns:" + ";".join(missing_required),
                "raw_smiles": "",
                "raw_type": "",
                "raw_value": "",
                "raw_unit": "",
                "raw_relation": "",
            }
        )
        return pd.DataFrame(columns=OUTPUT_COLUMNS + ["__canonical_smiles"]), removed

    raw = pd.DataFrame(
        {
            "source_file": df.get("__source_file", ""),
            "source_sheet": df.get("__source_sheet", ""),
            "source_row": df.get("__source_row", ""),
            "raw_smiles": df[smiles_col],
            "raw_type": df[type_col],
            "raw_value": df[value_col],
            "raw_unit": df[unit_col],
            "raw_relation": df[relation_col] if relation_col else "",
        }
    )

    text_smiles = raw["raw_smiles"].map(clean_text)
    text_type = raw["raw_type"].map(clean_text)
    text_value = raw["raw_value"].map(clean_text)
    text_unit = raw["raw_unit"].map(clean_text)
    text_relation = raw["raw_relation"].map(clean_text)

    reason = pd.Series("", index=raw.index, dtype=object)

    missing_key = (text_smiles == "") | (text_type == "") | (text_value == "") | (text_unit == "")
    reason.loc[missing_key & (reason == "")] = "missing_key_column_value"

    type_norm = text_type.str.upper().str.replace(" ", "", regex=False)
    not_mic = type_norm != "MIC"
    reason.loc[not_mic & (reason == "")] = "not_mic"

    relation_bad = text_relation.str.contains(RELATION_PATTERN, regex=True, na=False)
    reason.loc[relation_bad & (reason == "")] = "relation_contains_less_or_greater"

    value_bad_relation = text_value.str.contains(RELATION_PATTERN, regex=True, na=False)
    reason.loc[value_bad_relation & (reason == "")] = "value_contains_less_or_greater"

    value_clean = text_value.str.replace(",", "", regex=False).str.strip()
    value_numeric = pd.to_numeric(value_clean, errors="coerce")
    invalid_value = value_numeric.isna() | (value_numeric <= 0)
    reason.loc[invalid_value & (reason == "")] = "invalid_or_zero_concentration"

    invalid_unit = (text_unit == "") | (text_unit == "0")
    reason.loc[invalid_unit & (reason == "")] = "missing_or_zero_unit"

    raw["reason"] = reason
    raw["value_numeric"] = value_numeric
    raw["unit_text"] = text_unit

    removed.extend(make_removed_records(raw[raw["reason"] != ""]))

    candidates = raw[raw["reason"] == ""].copy()
    if candidates.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS + ["__canonical_smiles"]), removed

    molecule_cache = standardize_many_smiles(candidates["raw_smiles"].map(clean_text))

    for item in candidates[["raw_smiles", "value_numeric", "unit_text", "source_file", "source_sheet", "source_row", "raw_type", "raw_value", "raw_unit", "raw_relation"]].itertuples(index=False, name=None):
        raw_smiles, value, unit_text_item, source_file, source_sheet, source_row, raw_type, raw_value, raw_unit, raw_relation = item
        raw_smiles_text = clean_text(raw_smiles)
        canonical_smiles, mol_weight, mol_error = molecule_cache.get(raw_smiles_text, (None, None, "invalid_molecule"))

        if mol_error is not None or canonical_smiles is None or mol_weight is None:
            removed.append(
                {
                    "source_file": source_file,
                    "source_sheet": source_sheet,
                    "source_row": source_row,
                    "reason": mol_error or "invalid_molecule",
                    "raw_smiles": raw_smiles,
                    "raw_type": raw_type,
                    "raw_value": raw_value,
                    "raw_unit": raw_unit,
                    "raw_relation": raw_relation,
                }
            )
            continue

        standard_ug_ml = convert_to_ug_ml(float(value), unit_text_item, mol_weight)
        if standard_ug_ml is None or standard_ug_ml <= 0:
            removed.append(
                {
                    "source_file": source_file,
                    "source_sheet": source_sheet,
                    "source_row": source_row,
                    "reason": "unsupported_unit_or_invalid_conversion",
                    "raw_smiles": raw_smiles,
                    "raw_type": raw_type,
                    "raw_value": raw_value,
                    "raw_unit": raw_unit,
                    "raw_relation": raw_relation,
                }
            )
            continue

        label = 1 if 0 < standard_ug_ml < ACTIVE_THRESHOLD_UG_ML else 0
        output_rows.append(
            {
                "smiles": canonical_smiles,
                "concentration_value": float(value),
                "concentration_unit": unit_text_item,
                "pmic_ug_ml": standard_ug_ml,
                "activity_label": label,
                "__canonical_smiles": canonical_smiles,
            }
        )

    return pd.DataFrame(output_rows), removed

def main() -> None:
    all_frames: List[pd.DataFrame] = []
    missing_files: List[str] = []

    for file_name in INPUT_FILES:
        path = BASE_DIR / file_name
        if not path.exists():
            missing_files.append(file_name)
            continue
        all_frames.extend(read_input_file(path))

    if missing_files:
        raise FileNotFoundError("Missing input files: " + "; ".join(missing_files))

    processed_frames: List[pd.DataFrame] = []
    removed_rows: List[Dict[str, Any]] = []

    for frame in all_frames:
        processed, removed = build_standard_frame(frame)
        if not processed.empty:
            processed_frames.append(processed)
        removed_rows.extend(removed)

    if processed_frames:
        result = pd.concat(processed_frames, ignore_index=True)
    else:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS + ["__canonical_smiles"])

    before_dedup = len(result)

    if DEDUP_BY_CANONICAL_SMILES_KEEP_LOWEST_UG_ML and not result.empty:
        result = result.sort_values(
            by=["__canonical_smiles", "pmic_ug_ml"],
            ascending=[True, True],
            kind="mergesort",
        )
        result = result.drop_duplicates(subset=["__canonical_smiles"], keep="first")
    else:
        result = result.drop_duplicates(subset=OUTPUT_COLUMNS, keep="first")

    after_dedup = len(result)
    result = result[OUTPUT_COLUMNS].reset_index(drop=True)

    result.to_csv(BASE_DIR / OUTPUT_CSV, index=False, encoding="utf-8-sig")
    if WRITE_XLSX:
        result.to_excel(BASE_DIR / OUTPUT_XLSX, index=False)

    removed_df = pd.DataFrame(removed_rows)
    removed_df.to_csv(BASE_DIR / REMOVED_LOG_CSV, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {"item": "input_files", "value": len(INPUT_FILES)},
            {"item": "input_sheets_or_tables", "value": len(all_frames)},
            {"item": "valid_rows_before_dedup", "value": before_dedup},
            {"item": "valid_rows_after_dedup", "value": after_dedup},
            {"item": "deduplicated_rows", "value": before_dedup - after_dedup},
            {"item": "removed_rows", "value": len(removed_rows)},
            {"item": "active_label_1", "value": int((result["activity_label"] == 1).sum()) if not result.empty else 0},
            {"item": "inactive_label_0", "value": int((result["activity_label"] == 0).sum()) if not result.empty else 0},
            {"item": "active_threshold_ug_ml", "value": ACTIVE_THRESHOLD_UG_ML},
        ]
    )
    summary.to_csv(BASE_DIR / SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("Done")
    print(f"Output CSV: {OUTPUT_CSV}")
    if WRITE_XLSX:
        print(f"Output XLSX: {OUTPUT_XLSX}")
    print(f"Removed log: {REMOVED_LOG_CSV}")
    print(f"Summary: {SUMMARY_CSV}")
    print(f"Valid rows before dedup: {before_dedup}")
    print(f"Valid rows after dedup: {after_dedup}")
    print(f"Active label 1: {int((result['activity_label'] == 1).sum()) if not result.empty else 0}")
    print(f"Inactive label 0: {int((result['activity_label'] == 0).sum()) if not result.empty else 0}")


if __name__ == "__main__":
    main()
