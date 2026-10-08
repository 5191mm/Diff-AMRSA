#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timeout resumable fixed-set MRSA Scheme A geometry builder for Diff-AMRSA.

Purpose
- Keep fixed training set policy
- No repeat
- No sample weight
- No activity filtering
- Reuse existing resume_cache batches from earlier resumable scripts
- Skip molecules whose RDKit conformer generation takes too long
- Save checkpoint batches frequently so power loss does not erase completed work

Typical 30/10 command
python build_conformers.py --input processed_drug_mic_dataset_schemeA.csv --output_root data/custom_mrsa_schemeA_fixed_30_10 --neutralize_core --num_candidates 30 --select_conformers 10 --num_workers 16 --rdkit_threads 1 --checkpoint_every 20 --molecule_timeout_sec 900 --raw_only

This file expects build_conformers_core.py to be in the same folder or in data_processing/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import queue
import random
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _import_base_module():
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "build_conformers_core.py",
        Path.cwd() / "build_conformers_core.py",
        Path.cwd() / "data_processing" / "build_conformers_core.py",
    ]
    for path in candidates:
        if path.exists() and path.resolve() != here:
            spec = importlib.util.spec_from_file_location("build_conformers_core", str(path))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["build_conformers_core"] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "Cannot find build_conformers_core.py. Put this timeout script in the same folder as it, "
        "or put both under data_processing/."
    )


base = _import_base_module()
PRETRAIN_ATOM_ENCODER = base.PRETRAIN_ATOM_ENCODER
ALLOWED_CHARGES = base.ALLOWED_CHARGES
BuildResult = base.BuildResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Timeout resumable fixed-set MRSA Scheme A geometry builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Scheme A CSV file")
    parser.add_argument("--output_root", type=str, default="data/custom_mrsa_schemeA_fixed")
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
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--rdkit_threads", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=1, help="Kept for CLI compatibility. Timeout mode schedules one molecule per worker.")
    parser.add_argument("--raw_only", action="store_true")
    parser.add_argument("--processed_only", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=20, help="Flush completed BuildResult records every N rows")
    parser.add_argument("--cache_dir", type=str, default=None, help="Resume cache directory. Default output_root/resume_cache")
    parser.add_argument("--no_resume", action="store_true", help="Ignore existing cache and start a new cache run")
    parser.add_argument("--molecule_timeout_sec", type=int, default=900, help="Skip one molecule if its worker runs longer than this many seconds")
    parser.add_argument("--poll_interval_sec", type=float, default=2.0, help="How often the scheduler checks finished and timed out workers")
    parser.add_argument("--heartbeat_sec", type=float, default=60.0, help="Print scheduler heartbeat every N seconds")
    parser.add_argument("--max_hard_kill_wait_sec", type=float, default=5.0, help="Wait after terminate before kill")
    return parser.parse_args()


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


def save_batch(cache_dir: Path, batch_id: int, batch: List[Tuple[int, object]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"batch_{batch_id:06d}.pkl.tmp"
    out = cache_dir / f"batch_{batch_id:06d}.pkl"
    with open(tmp, "wb") as f:
        pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)


def load_cached_batches(cache_dir: Path) -> Tuple[List[Tuple[int, object]], set, int]:
    all_results: List[Tuple[int, object]] = []
    done = set()
    max_id = -1
    if not cache_dir.exists():
        return all_results, done, 0
    for path in sorted(cache_dir.glob("batch_*.pkl")):
        try:
            batch_id = int(path.stem.split("_")[-1])
            max_id = max(max_id, batch_id)
            with open(path, "rb") as f:
                batch = pickle.load(f)
            for idx, res in batch:
                if idx in done:
                    continue
                all_results.append((idx, res))
                done.add(idx)
        except Exception as exc:
            print(f"Warning: failed to load cache batch {path}: {exc}", flush=True)
    return all_results, done, max_id + 1


def make_payload(df: pd.DataFrame, args: argparse.Namespace, idx: int, task_seed: int) -> Tuple[int, Dict, Dict, int]:
    args_dict = vars(args).copy()
    for key in [
        "checkpoint_every",
        "cache_dir",
        "no_resume",
        "molecule_timeout_sec",
        "poll_interval_sec",
        "heartbeat_sec",
        "max_hard_kill_wait_sec",
    ]:
        args_dict.pop(key, None)
    row = df.loc[idx]
    return int(idx), row.to_dict(), args_dict, int(task_seed)


def make_skip_result(row_dict: Dict, args: argparse.Namespace, reason: str, extra: Optional[str] = None) -> object:
    smiles = str(row_dict.get(args.smiles_col, "")).strip()
    res = BuildResult(ok=False, smiles_input=smiles)
    try:
        val = row_dict.get(args.label_col, np.nan)
        res.activity_label = int(val) if not pd.isna(val) else None
    except Exception:
        res.activity_label = None
    try:
        val = row_dict.get(args.pmic_col, np.nan)
        res.pmic_ug_ml = float(val) if not pd.isna(val) else None
    except Exception:
        res.pmic_ug_ml = None
    try:
        val = row_dict.get(args.pmic_train_col, np.nan)
        res.pmic_train = float(val) if not pd.isna(val) else None
    except Exception:
        res.pmic_train = None
    try:
        val = row_dict.get(args.mask_col, np.nan)
        res.pmic_regression_mask = int(val) if not pd.isna(val) else 0
    except Exception:
        res.pmic_regression_mask = 0
    try:
        val = row_dict.get(args.repeat_col, 1)
        res.generation_repeat = int(val) if not pd.isna(val) else 1
    except Exception:
        res.generation_repeat = 1
    res.reason = reason if extra is None else f"{reason}:{extra}"
    return res


def task_worker(task_id: int, payload: Tuple[int, Dict, Dict, int], result_queue) -> None:
    try:
        idx, res = base.build_one_worker(payload)
        result_queue.put((task_id, idx, res, None))
    except BaseException as exc:
        idx = payload[0]
        result_queue.put((task_id, idx, None, repr(exc)))


def safe_terminate_process(proc: mp.Process, hard_wait: float) -> None:
    if not proc.is_alive():
        return
    try:
        proc.terminate()
    except Exception:
        pass
    proc.join(timeout=max(0.1, float(hard_wait)))
    if proc.is_alive():
        try:
            proc.kill()
        except Exception:
            pass
        proc.join(timeout=1.0)


def result_to_manifest_row(idx: int, split: str, res: object) -> Dict:
    return {
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
        "training_set_policy": "fixed_once_no_weight_no_repeat_timeout_resume",
        "dropped_fragments": res.dropped_fragments,
        "unsupported_symbols": res.unsupported_symbols,
        "scaffold": res.scaffold,
    }


def run_timeout_scheduler(df: pd.DataFrame, args: argparse.Namespace, todo_indices: List[int], cache_dir: Path, next_batch_id: int) -> None:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    active: Dict[int, Dict] = {}
    pointer = 0
    task_id = 0
    batch: List[Tuple[int, object]] = []
    batch_id = next_batch_id
    completed_count = 0
    timeout_count = 0
    error_count = 0
    last_heartbeat = time.time()
    pbar = tqdm(total=len(todo_indices), desc="Build conformers remaining", dynamic_ncols=True)

    def flush_batch(force: bool = False) -> None:
        nonlocal batch, batch_id
        if batch and (force or len(batch) >= max(1, int(args.checkpoint_every))):
            save_batch(cache_dir, batch_id, batch)
            print(f"\ncheckpoint saved batch={batch_id} rows={len(batch)}", flush=True)
            batch_id += 1
            batch = []

    def launch_more() -> None:
        nonlocal pointer, task_id
        while pointer < len(todo_indices) and len(active) < max(1, int(args.num_workers)):
            idx = int(todo_indices[pointer])
            pointer += 1
            seed = int(args.seed + idx)
            payload = make_payload(df, args, idx, seed)
            p = ctx.Process(target=task_worker, args=(task_id, payload, result_queue))
            p.start()
            active[task_id] = {
                "process": p,
                "idx": idx,
                "row_dict": payload[1],
                "start": time.time(),
            }
            task_id += 1

    try:
        launch_more()
        while active or pointer < len(todo_indices):
            made_progress = False

            while True:
                try:
                    finished_task_id, idx, res, err = result_queue.get_nowait()
                except queue.Empty:
                    break
                info = active.pop(finished_task_id, None)
                if info is None:
                    continue
                proc = info["process"]
                proc.join(timeout=0.2)
                if err is not None or res is None:
                    error_count += 1
                    res = make_skip_result(info["row_dict"], args, "worker_exception", str(err)[:300])
                batch.append((int(idx), res))
                completed_count += 1
                pbar.update(1)
                made_progress = True
                flush_batch(force=False)

            now = time.time()
            timed_out_ids = []
            for tid, info in list(active.items()):
                proc = info["process"]
                elapsed = now - float(info["start"])
                if elapsed > float(args.molecule_timeout_sec):
                    timed_out_ids.append(tid)
                elif not proc.is_alive() and proc.exitcode not in (None, 0):
                    # Worker died before putting result
                    res = make_skip_result(info["row_dict"], args, "worker_process_exitcode", str(proc.exitcode))
                    active.pop(tid, None)
                    proc.join(timeout=0.2)
                    batch.append((int(info["idx"]), res))
                    completed_count += 1
                    error_count += 1
                    pbar.update(1)
                    made_progress = True
                    flush_batch(force=False)

            for tid in timed_out_ids:
                info = active.pop(tid, None)
                if info is None:
                    continue
                proc = info["process"]
                elapsed = now - float(info["start"])
                safe_terminate_process(proc, args.max_hard_kill_wait_sec)
                res = make_skip_result(info["row_dict"], args, "timeout_conformer_generation", f"{int(elapsed)}s")
                batch.append((int(info["idx"]), res))
                completed_count += 1
                timeout_count += 1
                pbar.update(1)
                made_progress = True
                flush_batch(force=False)

            if made_progress:
                launch_more()

            if now - last_heartbeat >= float(args.heartbeat_sec):
                oldest = 0.0
                if active:
                    oldest = max(now - float(info["start"]) for info in active.values())
                print(
                    f"\nheartbeat done_this_run={completed_count}/{len(todo_indices)} "
                    f"active={len(active)} queued={len(todo_indices)-pointer} "
                    f"timeouts={timeout_count} errors={error_count} oldest_active_sec={int(oldest)}",
                    flush=True,
                )
                last_heartbeat = now

            if not made_progress:
                time.sleep(max(0.1, float(args.poll_interval_sec)))

        flush_batch(force=True)
        pbar.close()
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating active workers and saving completed batch.", flush=True)
        for info in list(active.values()):
            safe_terminate_process(info["process"], args.max_hard_kill_wait_sec)
        flush_batch(force=True)
        pbar.close()
        raise
    finally:
        try:
            result_queue.close()
        except Exception:
            pass


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
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_root / "resume_cache"

    if args.processed_only:
        split_items = base.load_raw_pickle(raw_dir)
        base.save_processed(split_items, processed_dir)
        print("Processed PyG files built from existing raw pickle files", flush=True)
        print(f"Output root: {output_root.resolve()}", flush=True)
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
    idx_to_split: Dict[int, str] = {}
    for split, idxs in splits_idx.items():
        for idx in idxs:
            idx_to_split[int(idx)] = split

    all_indices = [int(i) for i in df.index.tolist()]

    if args.no_resume and cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)

    cached_results, done_indices, next_batch_id = load_cached_batches(cache_dir)
    todo_indices = [idx for idx in all_indices if idx not in done_indices]

    if args.num_workers > 1 and args.rdkit_threads == 0:
        print("Warning: --num_workers > 1 with --rdkit_threads 0 can oversubscribe CPU. Reset rdkit_threads to 1.", flush=True)
        args.rdkit_threads = 1

    print("Training set policy: fixed once, no repeat, no sample weight, no activity filtering", flush=True)
    print("Timeout mode: enabled", flush=True)
    print(f"Input rows: {len(all_indices)}", flush=True)
    print(f"Cached rows: {len(done_indices)}", flush=True)
    print(f"Remaining rows: {len(todo_indices)}", flush=True)
    print(f"Cache dir: {cache_dir.resolve()}", flush=True)
    print(f"RDKit conformer workers: {args.num_workers}", flush=True)
    print(f"RDKit threads per worker: {args.rdkit_threads}", flush=True)
    print(f"Conformer setting: num_candidates={args.num_candidates}, select_conformers={args.select_conformers}", flush=True)
    print(f"Molecule timeout: {args.molecule_timeout_sec} sec", flush=True)

    if todo_indices:
        run_timeout_scheduler(df, args, todo_indices, cache_dir, next_batch_id)

    # Reload all cache batches to build final files only when complete
    all_results, done_indices, _ = load_cached_batches(cache_dir)
    if len(done_indices) < len(all_indices):
        missing_count = len(all_indices) - len(done_indices)
        print(f"Not complete yet: {missing_count} rows still missing. Re-run same command to continue.", flush=True)
        return

    split_items: Dict[str, List[Tuple[str, List[object]]]] = {"train": [], "val": [], "test": []}
    manifest_rows = []
    skipped = Counter()

    for idx, res in sorted(all_results, key=lambda x: x[0]):
        split = idx_to_split.get(int(idx), "train")
        if res.ok and res.mols is not None and res.smiles_core is not None:
            split_items[split].append((res.smiles_core, res.mols))
        else:
            skipped[res.reason] += 1
        manifest_rows.append(result_to_manifest_row(int(idx), split, res))

    if not split_items["train"] or not split_items["val"] or not split_items["test"]:
        raise RuntimeError({k: len(v) for k, v in split_items.items()})

    base.save_raw_pickle(split_items, raw_dir)
    if not args.raw_only:
        base.save_processed(split_items, processed_dir)
    else:
        print("Raw-only mode enabled. Skipped PyG .pt conversion.", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_root / "mrsa_scheme_a_manifest.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_csv": str(input_path.resolve()),
        "output_root": str(output_root.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "atom_encoder": PRETRAIN_ATOM_ENCODER,
        "allowed_charges": sorted(ALLOWED_CHARGES),
        "num_candidates": args.num_candidates,
        "select_conformers": args.select_conformers,
        "training_set_policy": "fixed_once_no_weight_no_repeat_timeout_resume",
        "use_train_repeat": False,
        "sample_weight_used": False,
        "activity_filter_used": False,
        "num_workers": int(args.num_workers),
        "rdkit_threads": int(args.rdkit_threads),
        "molecule_timeout_sec": int(args.molecule_timeout_sec),
        "checkpoint_every": int(args.checkpoint_every),
        "raw_only": bool(args.raw_only),
        "split_items": {k: len(v) for k, v in split_items.items()},
        "split_conformers": {k: int(sum(len(c) for _, c in v)) for k, v in split_items.items()},
        "skipped_reasons": dict(skipped),
    }
    (output_root / "mrsa_scheme_a_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("MRSA Scheme A timeout-resumable geometry dataset built", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
