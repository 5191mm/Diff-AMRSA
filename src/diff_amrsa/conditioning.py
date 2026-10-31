"""Retrieval-conditioned Megalodon using non-invasive atom-embedding hooks.

The existing Megalodon state_dict and original source files remain unchanged.
The adapter enters the denoising network at every timestep, not just reranking.
"""
from __future__ import annotations
import torch
from torch import nn


class ContextInjector(nn.Module):
    def __init__(self, context_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self._context = None
        self._node_batch = None
        self._handles = []

    def set_context(self, context):
        if context is None:
            raise ValueError('A retrieval embedding is required for conditioned diffusion')
        if context.ndim != 2:
            raise ValueError('Context must be [batch_size, embedding_dim]')
        self._context = context

    def attach(self, dynamics):
        """Attach to both ordinary and EMA denoisers to support training/inference."""
        self.detach()
        model = dynamics.model if hasattr(dynamics, 'model') else dynamics
        targets = [model]
        if getattr(dynamics, 'ema', False):
            targets.append(dynamics.ema_model.module)
        for target in targets:
            if not hasattr(target, 'atom_embedder'):
                raise TypeError('Expected mega_large/megav3 with atom_embedder')
            self._handles.append(target.register_forward_pre_hook(self._get_batch))
            self._handles.append(target.atom_embedder.register_forward_hook(self._inject))

    def _get_batch(self, _module, args):
        arg = args[0]
        batch = arg['batch'] if isinstance(arg, dict) else arg
        if not isinstance(batch, torch.Tensor):
            raise RuntimeError('Cannot obtain Megalodon atom batch indices')
        self._node_batch = batch.long()

    def _inject(self, _module, _args, atom_features):
        if self._context is None or self._node_batch is None:
            raise RuntimeError('Retrieval context was not set before denoising')
        indices = self._node_batch
        context = self._context.to(device=atom_features.device, dtype=atom_features.dtype)
        n_graphs = int(indices.max().item()) + 1
        if context.shape[0] == 1:
            context = context.expand(n_graphs, -1)
        if context.shape[0] != n_graphs:
            raise ValueError(f'Context batch {context.shape[0]} != denoiser batch {n_graphs}')
        addition = self.net(context)[indices]
        if addition.shape != atom_features.shape:
            raise RuntimeError('Adapter embedding dimension mismatch')
        return atom_features + addition

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
