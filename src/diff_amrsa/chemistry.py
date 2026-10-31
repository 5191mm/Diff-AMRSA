"""RDKit-based, atom-index-preserving graph and retrieval descriptors."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

ATOM_VOCAB = ("H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi", "Se")
ATOM_INDEX = {s: i for i, s in enumerate(ATOM_VOCAB)}
BOND_VOCAB = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
_FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_FRAGMENT_SMARTS = {
    "aromatic": "a", "amine": "[NX3;H2,H1;!$(NC=O)]", "amide": "[NX3][CX3](=[OX1])",
    "carbonyl": "[CX3]=[OX1]", "carboxyl": "[CX3](=O)[OX2H1]",
    "hydroxyl": "[OX2H]", "halogen": "[F,Cl,Br,I]", "n_hetero": "[n]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]", "heterocycle": "[R;!#6]",
    "ether": "[OD2]([#6])[#6]", "cation": "[*+]", "anion": "[*-]",
    "donor": "[N,O,S;H1,H2]", "acceptor": "[N,O,S;H0]",
}
_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in _FRAGMENT_SMARTS.items()}


def canonicalize(smiles: str) -> tuple[str, Chem.Mol]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {str(smiles)[:80]}")
    pieces = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not pieces:
        raise ValueError("Empty structure")
    mol = max(pieces, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()))
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), mol


def descriptors(mol: Chem.Mol) -> dict:
    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    frags = sorted(k for k, pattern in _PATTERNS.items() if pattern is not None and mol.HasSubstructMatch(pattern))
    return {"smiles": smi, "scaffold": scaf or "", "fragments": frags}


def fingerprint(mol: Chem.Mol):
    return _FPG.GetFingerprint(mol)


def tanimoto(a, b) -> float:
    return float(DataStructs.TanimotoSimilarity(a, b))


def bond_index(bond: Chem.Bond) -> int:
    t = bond.GetBondType()
    return BOND_VOCAB.index(t) if t in BOND_VOCAB else 0


@dataclass
class Graph:
    atoms: torch.Tensor
    charges: torch.Tensor
    edges: torch.Tensor
    bonds: torch.Tensor
    xyz: Optional[torch.Tensor] = None

    def to(self, device):
        return Graph(*(v.to(device) if v is not None else None for v in
                       (self.atoms, self.charges, self.edges, self.bonds, self.xyz)))

    @property
    def n(self):
        return self.atoms.shape[0]


def graph_from_mol(mol: Chem.Mol, with_xyz=False) -> Graph:
    atoms = [ATOM_INDEX.get(a.GetSymbol(), len(ATOM_VOCAB)) for a in mol.GetAtoms()]
    charges = [max(-3, min(3, int(a.GetFormalCharge()))) + 3 for a in mol.GetAtoms()]
    if not atoms:
        raise ValueError("Molecule has no atoms")
    src, dst, et = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend([i, j]); dst.extend([j, i]); et.extend([bond_index(bond)] * 2)
    edges = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    xyz = None
    if with_xyz:
        if not mol.GetNumConformers():
            raise ValueError("3D conformer required")
        xyz = torch.tensor(np.asarray(mol.GetConformer().GetPositions()).copy(), dtype=torch.float32)
        if not torch.isfinite(xyz).all():
            raise ValueError("Nonfinite coordinates")
    return Graph(torch.tensor(atoms, dtype=torch.long), torch.tensor(charges, dtype=torch.long),
                 edges, torch.tensor(et, dtype=torch.long), xyz)


def proper_kabsch_rmsd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[-1] != 3:
        raise ValueError("Coordinate tensors must match and have shape [N,3]")
    p, q = pred - pred.mean(0), target - target.mean(0)
    u, _, vh = torch.linalg.svd(p.T @ q, full_matrices=False)
    d = torch.linalg.det(u @ vh)
    correction = torch.diag(torch.stack((d.new_tensor(1.), d.new_tensor(1.), torch.where(d < 0, -d.new_tensor(1.), d.new_tensor(1.)))))
    rot = u @ correction @ vh
    aligned = p @ rot
    return ((aligned - q).square().sum(-1).mean() + 1e-12).sqrt()


def coords_to_mol(mol: Chem.Mol, xyz: torch.Tensor) -> Chem.Mol:
    out = Chem.Mol(mol)
    if xyz.shape != (mol.GetNumAtoms(), 3) or not torch.isfinite(xyz).all():
        raise ValueError("Coordinates invalid or atom count changed")
    if not out.GetNumConformers():
        out.AddConformer(Chem.Conformer(out.GetNumAtoms()), assignId=True)
    conf = out.GetConformer()
    xyz_np = xyz.detach().cpu().numpy()
    for i in range(out.GetNumAtoms()):
        conf.SetAtomPosition(i, xyz_np[i].tolist())
    return out
