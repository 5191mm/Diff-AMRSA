"""Reusable molecular activity features for training and inference."""
from __future__ import annotations

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator

GEN=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)


def feature(mol):
    bits=np.zeros(2048,dtype=np.float32)
    DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(mol),bits)
    desc=np.asarray([Descriptors.MolWt(mol),Crippen.MolLogP(mol),
                     rdMolDescriptors.CalcTPSA(mol),rdMolDescriptors.CalcNumHBA(mol),
                     rdMolDescriptors.CalcNumHBD(mol),rdMolDescriptors.CalcNumRotatableBonds(mol),
                     rdMolDescriptors.CalcNumRings(mol),rdMolDescriptors.CalcNumAromaticRings(mol),
                     mol.GetNumHeavyAtoms()],dtype=np.float32)
    desc[0]/=1000.;desc[2]/=300.;desc[3:]/=50.
    return np.concatenate((bits,desc))
