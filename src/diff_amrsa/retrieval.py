"""Training-only molecular reference retrieval with exact four-term paper scoring."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from .chemistry import canonicalize, descriptors, fingerprint, graph_from_mol

DEFAULT_WEIGHTS = (0.4, 0.2, 0.2, 0.2)


def make_library(smiles, activities, encoder, device='cpu'):
    rows, embs = [], []
    encoder.to(device).eval()
    seen = set()
    with torch.no_grad():
        for smi, label in zip(smiles, activities):
            try:
                canonical, mol = canonicalize(smi)
                y = int(label)
                if canonical in seen or y not in (0, 1):
                    continue
                graph = graph_from_mol(mol).to(device)
                emb = encoder(graph).cpu().numpy().astype(np.float32)
                rows.append({**descriptors(mol), 'activity': y})
                embs.append(emb)
                seen.add(canonical)
            except (ValueError, TypeError, OverflowError):
                continue
    if not rows:
        raise ValueError('No labeled train references generated')
    return rows, np.stack(embs)


def save_library(path, rows, embeddings, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if embeddings.shape[0] != len(rows):
        raise ValueError('Embedding/reference rows mismatch')
    np.savez_compressed(path, embeddings=embeddings.astype(np.float32),
                        rows=np.array([json.dumps(r) for r in rows]),
                        metadata=json.dumps(metadata or {}))


class RetrievalLibrary:
    def __init__(self, rows, embeddings, top_k=5, tau=0.2, weights=DEFAULT_WEIGHTS):
        if len(rows) != len(embeddings) or not rows:
            raise ValueError('Invalid library content')
        if any(x < 0 for x in weights) or abs(sum(weights)-1) > 1e-7 or tau <= 0:
            raise ValueError('Invalid normalized retrieval coefficients or temperature')
        self.rows = rows
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.fps = [fingerprint(Chem.MolFromSmiles(row['smiles'])) for row in rows]
        self.top_k, self.tau, self.weights = top_k, tau, weights
        self.smiles = {row['smiles'] for row in rows}

    @classmethod
    def load(cls, path, **kwargs):
        with np.load(path, allow_pickle=False) as data:
            rows = [json.loads(x) for x in data['rows']]
            emb = data['embeddings']
        return cls(rows, emb, **kwargs)

    @lru_cache(maxsize=50000)
    def query(self, smiles, exclude_identity=True):
        canonical, mol = canonicalize(smiles)
        q = descriptors(mol)
        tan = BulkTanimotoSimilarity(fingerprint(mol), self.fps)
        fs = set(q['fragments'])
        scores = []
        a, b, c, d = self.weights
        for i, (row, sim) in enumerate(zip(self.rows, tan)):
            if exclude_identity and row['smiles'] == canonical:
                continue
            reference_fragments = set(row['fragments'])
            union = fs | reference_fragments
            frag = len(fs & reference_fragments)/len(union) if union else 0.0
            scaffold = int(bool(q['scaffold']) and q['scaffold'] == row['scaffold'])
            score = a*float(sim)+b*scaffold+c*frag+d*int(row['activity']==1)
            scores.append((score, i))
        if not scores:
            raise ValueError('Library empty after excluding query identity')
        top = sorted(scores, reverse=True)[:self.top_k]
        scores_np = np.array([x[0] for x in top], dtype=np.float64)
        scaled = (scores_np - scores_np.max())/self.tau
        w = np.exp(scaled); w /= w.sum()
        idx = [i for _, i in top]
        context = (w[:, None]*self.embeddings[idx]).sum(0)
        norm = np.linalg.norm(context)
        if norm > 1e-8:
            context /= norm
        return torch.from_numpy(context.astype(np.float32)), [
            {**self.rows[i], 'score': float(s), 'weight': float(weight)}
            for (s, i), weight in zip(top, w)]

    def batch_context(self, smiles, device=None):
        out = torch.stack([self.query(s)[0] for s in smiles])
        return out.to(device) if device else out
