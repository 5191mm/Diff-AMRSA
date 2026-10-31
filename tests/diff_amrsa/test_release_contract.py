"""Regression tests for the public repository cleanup and shared utilities."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from diff_amrsa.activity_features import feature
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_processing.utils_data import compute_all_statistics, save_statistics


def test_activity_features_are_reusable():
    x = feature(Chem.MolFromSmiles('CCO'))
    assert x.shape == (2057,)
    assert np.isfinite(x).all()


def test_conformer_statistics_include_training_priors(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    assert AllChem.EmbedMolecule(mol, randomSeed=12) == 0
    edges = []
    for bond in mol.GetBonds():
        edges.extend((int(bond.GetBondTypeAsDouble()),) * 2)
    graph = SimpleNamespace(
        x=torch.tensor([{'H': 0, 'C': 2, 'O': 4}[a.GetSymbol()] for a in mol.GetAtoms()]),
        edge_attr=torch.tensor(edges),
        charges=torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()]),
        is_aromatic=torch.tensor([int(a.GetIsAromatic()) for a in mol.GetAtoms()]),
        is_in_ring=torch.tensor([int(a.IsInRing()) for a in mol.GetAtoms()]),
        hybridization=torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
        smiles='CCO', mol=mol,
    )
    atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6,
                    'Si': 7, 'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12,
                    'I': 13, 'Hg': 14, 'Bi': 15}
    stats = compute_all_statistics([graph], atom_encoder)
    assert stats['atom_types'].shape == (16,)
    assert stats['bond_types'].shape == (5,)
    assert stats['charges'].shape == (6,)
    for key in ('atom_types', 'bond_types', 'charges'):
        assert np.isclose(stats[key].sum(), 1.0)
    assert stats['geometry_sample_count'] == 1
    save_statistics(stats, tmp_path, 'train')
    for name in ('train_n_h.pickle', 'train_atom_types_h.npy',
                 'train_bond_types_h.npy', 'train_charges_prior_h.npy',
                 'train_bond_lengths_h.pickle', 'train_dihedrals_h.pickle'):
        assert (tmp_path / name).is_file()


def test_release_has_clear_primary_entrypoints():
    assert (ROOT / 'scripts/run_pipeline.sh').is_file()
    assert (ROOT / 'data_processing/build_conformers.py').is_file()
    assert not (ROOT / 'process_mrsa_geom_scheme_a_fixedset_mp.py').exists()
    assert not (ROOT / '.git').exists()
