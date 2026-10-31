"""Graph message passing encoder and scaffold/activity supervised contrastive loss."""
from __future__ import annotations
from collections import defaultdict
import random
import torch
from torch import nn
import torch.nn.functional as F
from .chemistry import Graph


class GraphEncoder(nn.Module):
    def __init__(self, dim=128, layers=3, out_dim=128):
        super().__init__()
        self.dim, self.out_dim = dim, out_dim
        self.atom_emb = nn.Embedding(18, dim)
        self.charge_emb = nn.Embedding(7, dim)
        self.bond_emb = nn.Embedding(4, dim)
        self.messages = nn.ModuleList(nn.Sequential(nn.Linear(2*dim, dim), nn.SiLU(), nn.Linear(dim, dim)) for _ in range(layers))
        self.updates = nn.ModuleList(nn.Sequential(nn.Linear(2*dim, dim), nn.SiLU(), nn.LayerNorm(dim)) for _ in range(layers))
        self.readout = nn.Sequential(nn.Linear(dim*2, dim), nn.SiLU(), nn.Linear(dim, out_dim))

    def atom_features(self, graph: Graph):
        h = self.atom_emb(graph.atoms) + self.charge_emb(graph.charges)
        for msg, update in zip(self.messages, self.updates):
            agg = torch.zeros_like(h)
            if graph.edges.shape[1]:
                src, dst = graph.edges
                m = msg(torch.cat((h[src], self.bond_emb(graph.bonds)), dim=-1))
                agg.index_add_(0, dst, m)
                degree = torch.bincount(dst, minlength=graph.n).clamp(min=1).to(h.dtype).unsqueeze(1)
                agg = agg / degree
            h = h + update(torch.cat((h, agg), dim=-1))
        return h

    def forward(self, graph: Graph):
        h = self.atom_features(graph)
        pooled = torch.cat((h.mean(0), h.max(0).values), -1)
        return F.normalize(self.readout(pooled), dim=-1, eps=1e-8)


def contrastive_loss(embeddings, scaffolds, labels, temperature=0.1):
    """Paper objective: positives share nonempty scaffold and activity, negatives differ in activity."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    labels = torch.as_tensor(labels, device=embeddings.device)
    n = embeddings.shape[0]
    equal_scaffold = torch.tensor([[bool(scaffolds[i]) and scaffolds[i] == scaffolds[j]
                                    for j in range(n)] for i in range(n)], device=embeddings.device)
    labeled = labels >= 0
    same_label = labels[:, None] == labels[None, :]
    eye = torch.eye(n, device=embeddings.device, dtype=torch.bool)
    positive = equal_scaffold & same_label & labeled[:, None] & labeled[None, :] & ~eye
    negative = ~same_label & labeled[:, None] & labeled[None, :] & ~eye
    comparison = positive | negative
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return embeddings.sum() * 0.0, 0
    logits = F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).T / temperature
    denom = torch.logsumexp(logits.masked_fill(~comparison, float('-inf')), 1)
    positive_logprob = (logits - denom[:, None]).masked_fill(~positive, 0)
    per_anchor = -positive_logprob.sum(1) / positive.sum(1).clamp(min=1)
    return per_anchor[valid].mean(), int(valid.sum().item())


def contrastive_batch_indices(rows, batch_size=32, seed=42):
    """Yield scaffold-positive pairs with same-scaffold hard negatives where available."""
    rnd = random.Random(seed)
    groups = defaultdict(list); scaffold_groups = defaultdict(lambda: defaultdict(list)); by_label = defaultdict(list)
    for i, r in enumerate(rows):
        y = int(r['activity'])
        if y not in (0, 1):
            continue
        scaf = r['scaffold']
        by_label[y].append(i)
        if scaf:
            groups[(scaf, y)].append(i); scaffold_groups[scaf][y].append(i)
    eligible = [indices for indices in groups.values() if len(indices) > 1]
    if not eligible or not by_label[0] or not by_label[1]:
        raise ValueError("Need positive scaffold/activity pairs and both activity classes")
    eligible_keys = [k for k, v in groups.items() if len(v) >= 2]
    batch_size = max(3, batch_size)
    groups_per_batch = max(1, batch_size // 3)
    for _ in range(max(1, len(rows) // batch_size)):
        inds = []
        for scaf, label in rnd.choices(eligible_keys, k=groups_per_batch):
            i, j = rnd.sample(groups[(scaf, label)], 2)
            hard = scaffold_groups[scaf][1 - label]
            neg = rnd.choice(hard) if hard else rnd.choice(by_label[1 - label])
            inds.extend([i, j, neg])
        yield inds
