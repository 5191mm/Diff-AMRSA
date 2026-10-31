"""Atom-aligned graph-preserving latent conditional flow matching conformer refiner."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .chemistry import Graph, coords_to_mol, proper_kabsch_rmsd


class CoordinateAutoencoder(nn.Module):
    """Atom-indexed coordinate latent with graph-conditioned atom representations."""
    def __init__(self, hidden=128, latent=32):
        super().__init__()
        self.latent_dim = latent
        self.atom = nn.Embedding(18, hidden)
        self.charge = nn.Embedding(7, hidden)
        self.bond = nn.Embedding(4, hidden)
        self.graph_layer = nn.Sequential(nn.Linear(2*hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.encoder = nn.Sequential(nn.Linear(hidden+3, hidden), nn.SiLU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(hidden+latent, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def graph_features(self, graph: Graph):
        h = self.atom(graph.atoms) + self.charge(graph.charges)
        if graph.edges.numel():
            src, dst = graph.edges
            msg = self.graph_layer(torch.cat((h[src], self.bond(graph.bonds)), -1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)
            degree = torch.bincount(dst, minlength=graph.n).clamp(min=1).to(h.dtype).unsqueeze(-1)
            h = h + agg / degree
        return h

    def encode(self, graph, xyz):
        if xyz.shape != (graph.n, 3):
            raise ValueError('Atom order and coordinate dimensions must match')
        return self.encoder(torch.cat((self.graph_features(graph), xyz - xyz.mean(0)), -1))

    def decode(self, graph, latent):
        coords = self.decoder(torch.cat((self.graph_features(graph), latent), -1))
        return coords - coords.mean(0)

    def forward(self, graph, xyz):
        return self.decode(graph, self.encode(graph, xyz))


class ConditionalVelocity(nn.Module):
    def __init__(self, latent=32, cond_dim=128, hidden=128):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(latent+cond_dim*2+1, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, latent))

    def forward(self, z, t, retrieved_context, graph_embedding):
        if retrieved_context.ndim == 2:
            retrieved_context = retrieved_context.squeeze(0)
        if graph_embedding.ndim == 2:
            graph_embedding = graph_embedding.squeeze(0)
        assert retrieved_context.ndim == graph_embedding.ndim == 1
        cc = torch.cat((retrieved_context, graph_embedding, z.new_tensor([float(t)])), 0)
        return self.layers(torch.cat((z, cc[None].expand(z.shape[0], -1)), -1))


class LatentConformerRefiner(nn.Module):
    def __init__(self, latent=32, cond_dim=128, hidden=128, steps=5):
        super().__init__()
        self.autoencoder = CoordinateAutoencoder(hidden=hidden, latent=latent)
        self.velocity = ConditionalVelocity(latent=latent, cond_dim=cond_dim, hidden=hidden)
        self.steps = steps

    def reconstruction_loss(self, graph, target):
        prediction = self.autoencoder(graph, target)
        return proper_kabsch_rmsd(prediction, target)

    def fm_loss(self, graph, target, context, graph_embedding, sigma=0.15):
        if not torch.isfinite(target).all():
            raise ValueError('Target conformer contains nonfinite coordinates')
        source = target + sigma * torch.randn_like(target)
        source = source - source.mean(0)
        with torch.no_grad():
            z_src = self.autoencoder.encode(graph, source)
            z_dst = self.autoencoder.encode(graph, target - target.mean(0))
        t = torch.rand((), device=target.device)
        z_t = (1-t)*z_src + t*z_dst
        true_velocity = z_dst-z_src
        predicted_velocity = self.velocity(z_t, t, context, graph_embedding)
        return (predicted_velocity - true_velocity).square().mean()

    @torch.no_grad()
    def refine(self, graph, initial_xyz, context, graph_embedding, steps=None):
        if steps is None:
            steps = self.steps
        if steps <= 0:
            raise ValueError('steps must be positive')
        z = self.autoencoder.encode(graph, initial_xyz)
        dt = 1.0 / steps
        for i in range(steps):
            # Midpoint ODE solver with exactly N_f integration steps
            t = i*dt
            v1 = self.velocity(z, t, context, graph_embedding)
            v2 = self.velocity(z + 0.5*dt*v1, t+0.5*dt, context, graph_embedding)
            z = z + dt*v2
        coords = self.autoencoder.decode(graph, z)
        if not torch.isfinite(coords).all() or coords.shape != initial_xyz.shape:
            raise ValueError('Refinement produced invalid coordinates')
        return coords


def refine_rdkit_mol(mol, graph, context, graph_embedding, model, *, max_displacement=5.0):
    """Copy molecular graph verbatim, changing coordinates only; reject unsafe geometry."""
    source = graph.xyz
    if source is None:
        raise ValueError('Input molecule has no aligned 3D coordinates')
    xyz = model.refine(graph, source, context, graph_embedding)
    if torch.linalg.vector_norm(xyz - xyz.mean(0), dim=-1).max() > max_displacement * max(1, graph.n):
        raise ValueError('Refined coordinates exceed safety bound')
    result = coords_to_mol(mol, xyz)
    from rdkit import Chem
    if Chem.MolToSmiles(result, isomericSmiles=False) != Chem.MolToSmiles(mol, isomericSmiles=False):
        raise AssertionError('Molecular graph changed during coordinate refinement')
    return result
