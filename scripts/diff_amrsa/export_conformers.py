#!/usr/bin/env python3
"""Export genuine training/validation geometries from existing processed Megalodon splits."""
import argparse
from pathlib import Path
from rdkit import Chem
from megalodon.data.molecule_dataset import MoleculeDataset
from diff_amrsa.chemistry import canonicalize
from diff_amrsa.dataset import read_split


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',required=True)
    p.add_argument('--split-dir',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--max-per-molecule',type=int,default=3)
    a=p.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    for split in ('train','val'):
        ids={r['smiles'] for r in read_split(Path(a.split_dir)/f'{split}.jsonl')}
        ds=MoleculeDataset(root=a.dataset_root,processed_folder='processed',split=split)
        writer=Chem.SDWriter(str(out/f'{split}.sdf'))
        counts={}
        for data in ds:
            try:
                key,_=canonicalize(data.smiles)
                if key not in ids or counts.get(key,0)>=a.max_per_molecule:continue
                mol=data.mol
                if mol is None or mol.GetNumConformers()==0: continue
                Chem.SanitizeMol(mol)
                writer.write(mol)
                counts[key]=counts.get(key,0)+1
            except (ValueError, RuntimeError, TypeError):
                continue
        writer.close()
        print(split, 'unique molecules:',len(counts),'conformers:',sum(counts.values()))


if __name__ == '__main__':
    main()
