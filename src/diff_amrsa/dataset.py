"""Strict split extraction and activity annotation from existing processed GEOM files."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from .chemistry import canonicalize, descriptors


def activity_lookup(table_path):
    lookup = {}
    conflicts = set()
    with open(table_path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            try:
                smi, _ = canonicalize(r['smiles'])
                y = int(float(r['activity_label']))
            except (TypeError, ValueError, KeyError):
                continue
            if y not in (0, 1):
                continue
            if smi in lookup and lookup[smi] != y:
                conflicts.add(smi)
            lookup[smi] = y
    for smi in conflicts:
        del lookup[smi]
    return lookup, conflicts


def export_processed_split(dataset_root, table_path, output_dir, processed_folder='processed'):
    """Use authoritative train_h.pt / val_h.pt / test_h.pt assignments, never recreate a random split."""
    from megalodon.data.molecule_dataset import MoleculeDataset
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    lookup, conflicts = activity_lookup(table_path)
    all_identities, counts = {}, {}
    for split in ('train', 'val', 'test'):
        ds = MoleculeDataset(root=str(dataset_root), processed_folder=processed_folder, split=split)
        seen = set()
        records = []
        for item in ds:
            try:
                smi, mol = canonicalize(item.smiles)
            except (TypeError, ValueError):
                continue
            if smi in seen or smi not in lookup:
                continue
            seen.add(smi)
            records.append({**descriptors(mol), 'activity': lookup[smi]})
        if not records:
            raise ValueError(f'Empty labeled split {split}. Check preprocessing / table identity mapping')
        for r in records:
            previous = all_identities.setdefault(r['smiles'], split)
            if previous != split:
                raise ValueError(f'Data leakage: {r["smiles"]} found in {previous} and {split}')
        with open(root / f'{split}.jsonl', 'w', encoding='utf-8') as out:
            for r in records:
                out.write(json.dumps(r, ensure_ascii=False) + '\n')
        counts[split] = {'labeled_unique': len(records), 'processed_conformers': len(ds)}
    manifest = {'source_dataset_root': str(dataset_root), 'source_activity_table': str(table_path),
                'conflicting_activity_identities_excluded': len(conflicts), 'counts': counts,
                'nonoverlapping_canonical_identities': True}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def read_split(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def verify_train_only_library(library, split_dir):
    """Abort if any retrieval identity belongs to val/test or is outside declared train."""
    root = Path(split_dir)
    train = {r['smiles'] for r in read_split(root / 'train.jsonl')}
    val = {r['smiles'] for r in read_split(root / 'val.jsonl')}
    test = {r['smiles'] for r in read_split(root / 'test.jsonl')}
    identities = {r['smiles'] for r in library.rows}
    if identities & (val | test):
        raise ValueError('Retrieval library contains val/test molecular identities')
    outside = identities - train
    if outside:
        raise ValueError(f'Retrieval library contains {len(outside)} identities outside the training partition')
    if not identities:
        raise ValueError('Retrieval library empty')
    return len(identities)
