"""PyG conversion and training priors for the MRSA conformer preprocessing scripts.

This module restores the missing dependency of build_conformers_core.py.
Frequency priors use every supplied conformer. Expensive reference-geometry
histograms use a deterministic bounded subset by default.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import pickle

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP2D,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.OTHER,
]


def mol_to_torch_geometric(mol, smiles, atom_encoder, **kwargs):
    """Convert an RDKit conformer to a PyG Data using pretrained atom classes."""
    from torch_geometric.data import Data

    if mol.GetNumConformers() == 0:
        raise ValueError('RDKit molecule has no 3D coordinates')
    types = [atom_encoder[a.GetSymbol()] for a in mol.GetAtoms()]
    charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    if any(c < -2 or c > 3 for c in charges):
        raise ValueError('Formal charge is outside pretrained [-2, 3] range')
    if any(int(b.GetBondTypeAsDouble()) not in (1, 2, 3) and
           b.GetBondType() != Chem.BondType.AROMATIC for b in mol.GetBonds()):
        raise ValueError('Unsupported bond type')

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = 4 if bond.GetBondType() == Chem.BondType.AROMATIC else int(bond.GetBondTypeAsDouble())
        edge_index.extend(((i, j), (j, i)))
        edge_attr.extend((order, order))
    edges = torch.tensor(edge_index, dtype=torch.long).T.contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    xyz = torch.as_tensor(mol.GetConformer().GetPositions().copy(), dtype=torch.float32)
    xyz -= xyz.mean(dim=0, keepdim=True)
    return Data(
        x=torch.tensor(types, dtype=torch.long),
        edge_index=edges,
        edge_attr=torch.tensor(edge_attr, dtype=torch.long),
        pos=xyz,
        charges=torch.tensor(charges, dtype=torch.long),
        is_aromatic=torch.tensor([int(a.GetIsAromatic()) for a in mol.GetAtoms()], dtype=torch.long),
        is_in_ring=torch.tensor([int(a.IsInRing()) for a in mol.GetAtoms()], dtype=torch.long),
        hybridization=torch.tensor([HYBRIDIZATIONS.index(a.GetHybridization()) if a.GetHybridization() in HYBRIDIZATIONS else 0 for a in mol.GetAtoms()], dtype=torch.long),
        smiles=smiles,
        mol=mol,
    )


def save_pyg_dataset(graphs, path):
    """Write the (Data, slices) format expected by MoleculeDataset."""
    from torch_geometric.data.collate import collate

    if not graphs:
        raise ValueError('No molecular graphs to save')
    data, slices, _ = collate(graphs[0].__class__, graphs, increment=False, add_batch=False)
    torch.save((data, slices), path)


def _prior(counts, size):
    counts = np.asarray(counts, dtype=np.float64)[:size]
    if counts.size < size:
        counts = np.pad(counts, (0, size - counts.size))
    if not np.isfinite(counts).all() or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError('Invalid empirical prior')
    return (counts / counts.sum()).astype(np.float32)


def _collect_geom_stats(mols):
    """Collect reference distance/angle/torsion values in upstream key format."""
    from collections import defaultdict
    lengths, angles, dihedrals = defaultdict(lambda: [[], 0]), defaultdict(lambda: [[], 0]), defaultdict(lambda: [[], 0])
    from megalodon.metrics.geometry import (
        compute_bond_lengths, compute_bond_angles, compute_torsion_angles
    )
    for mol in mols:
        if mol is None or mol.GetNumConformers() == 0:
            continue
        for target, fn in ((lengths, compute_bond_lengths), (angles, compute_bond_angles),
                           (dihedrals, compute_torsion_angles)):
            try:
                values = fn(mol)
            except (ValueError, RuntimeError, OverflowError):
                continue
            for key, (observations, count) in values.items():
                target[key][0].extend(observations)
                target[key][1] += count
    return dict(lengths), dict(angles), dict(dihedrals)


def compute_all_statistics(graphs, atom_encoder, additional_features=True, max_geometry_samples=5000):
    """Compute empirical categorical priors and bounded 3D reference statistics.

    Geometric metrics on a subset are descriptive. Recomputing the full
    original corpus reference metrics may require additional resources.
    """
    if not graphs:
        raise ValueError('Cannot compute statistics from an empty data split')
    atom_counts = np.zeros(len(atom_encoder), dtype=np.float64)
    bond_counts = np.zeros(5, dtype=np.float64)
    charges = np.zeros(6, dtype=np.float64)
    rings = np.zeros(2, dtype=np.float64)
    aromatic = np.zeros(2, dtype=np.float64)
    hybridization = np.zeros(9, dtype=np.float64)
    sizes, valencies = Counter(), Counter()
    smiles = set()
    for data in graphs:
        n = int(data.x.numel())
        sizes[n] += 1
        atom_counts += np.bincount(data.x.detach().cpu().numpy().astype(int), minlength=len(atom_encoder))[:len(atom_encoder)]
        c = data.charges.detach().cpu().numpy().astype(int)
        charges += np.bincount(c + 2, minlength=6)[:6]
        rings += np.bincount(data.is_in_ring.detach().cpu().numpy().astype(int), minlength=2)[:2]
        aromatic += np.bincount(data.is_aromatic.detach().cpu().numpy().astype(int), minlength=2)[:2]
        hybridization += np.bincount(data.hybridization.detach().cpu().numpy().astype(int), minlength=9)[:9]
        bonds = data.edge_attr.detach().cpu().numpy().astype(int)
        bond_counts += np.bincount(bonds, minlength=5)[:5]
        bond_counts[0] += max(0, n * (n - 1) - len(bonds))
        if hasattr(data, 'mol') and data.mol is not None:
            for atom in data.mol.GetAtoms():
                valencies[(atom.GetSymbol(), int(atom.GetTotalValence()))] += 1
        smiles.add(str(data.smiles))
    # Deterministic geometry subset: evenly spaced conformers
    count = min(len(graphs), max_geometry_samples)
    idx = np.linspace(0, len(graphs) - 1, count, dtype=np.int64)
    mols = [graphs[int(i)].mol for i in idx if hasattr(graphs[int(i)], 'mol')]
    lengths, angles, dihedrals = _collect_geom_stats(mols)
    return {
        'num_nodes': dict(sizes),
        'atom_types': _prior(atom_counts, len(atom_encoder)),
        'bond_types': _prior(bond_counts, 5),
        'charges': _prior(charges, 6),
        'valency': dict(valencies),
        'is_in_ring': _prior(rings, 2),
        'is_aromatic': _prior(aromatic, 2),
        'hybridization': _prior(hybridization, 9),
        'bond_lengths': lengths, 'angles': angles, 'dihedrals': dihedrals,
        'smiles': sorted(smiles), 'geometry_sample_count': count,
    }


def save_statistics(stats, processed_dir, split):
    root = Path(processed_dir)
    root.mkdir(parents=True, exist_ok=True)
    for field in ('atom_types', 'bond_types', 'charges', 'is_in_ring', 'is_aromatic', 'hybridization'):
        np.save(root / f'{split}_{field}_h.npy', stats[field])
    np.save(root / f'{split}_charges_prior_h.npy', stats['charges'])
    for field, output in [('num_nodes', 'n_h'), ('valency', 'valency_h'),
                          ('bond_lengths', 'bond_lengths_h'), ('angles', 'angles_h'),
                          ('dihedrals', 'dihedrals_h'), ('smiles', 'smiles')]:
        with open(root / f'{split}_{output}.pickle', 'wb') as f:
            pickle.dump(stats[field], f, protocol=pickle.HIGHEST_PROTOCOL)
