#!/usr/bin/env python3
"""Two-stage paper-aligned training: autoencoder then conditional latent flow matching."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import torch
from rdkit import Chem
from diff_amrsa.chemistry import canonicalize, graph_from_mol
from diff_amrsa.dataset import read_split
from diff_amrsa.graph_encoder import GraphEncoder
from diff_amrsa.retrieval import RetrievalLibrary
from diff_amrsa.flow import LatentConformerRefiner


def load_sdf(path, allowed):
    out = []
    for mol in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True):
        if mol is None or not mol.GetNumConformers():
            continue
        try:
            # Explicit atom order inherited from conformer; do not canonicalize the actual mol here
            smi, _ = canonicalize(Chem.MolToSmiles(mol))
            if smi not in allowed:
                continue
            out.append((smi, mol))
        except (ValueError, TypeError):
            continue
    if not out:
        raise ValueError(f'No conformers in {path} whose molecular identities are in the declared split')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-sdf', required=True)
    p.add_argument('--val-sdf', required=True)
    p.add_argument('--split-dir', required=True)
    p.add_argument('--library', required=True)
    p.add_argument('--graph-checkpoint', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--autoencoder-epochs', type=int, default=20)
    p.add_argument('--flow-epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--sigma-x', type=float, default=0.15)
    p.add_argument('--max-conformers', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    train_ids = {r['smiles'] for r in read_split(Path(a.split_dir)/'train.jsonl')}
    val_ids = {r['smiles'] for r in read_split(Path(a.split_dir)/'val.jsonl')}
    if train_ids & val_ids:
        raise ValueError('Leaky split')
    train = load_sdf(a.train_sdf, train_ids)
    val = load_sdf(a.val_sdf, val_ids)
    if a.max_conformers:
        train = train[:a.max_conformers]; val = val[:max(1, a.max_conformers//10)]
    library = RetrievalLibrary.load(a.library)
    if {r['smiles'] for r in library.rows} - train_ids:
        raise ValueError('Retrieval library includes identities outside train partition')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph_encoder = GraphEncoder().to(device)
    graph_encoder.load_state_dict(torch.load(a.graph_checkpoint, map_location=device, weights_only=True)['state_dict'])
    graph_encoder.eval()
    for parameter in graph_encoder.parameters():
        parameter.requires_grad_(False)
    model = LatentConformerRefiner().to(device)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    def get_example(example):
        smi, mol = example
        graph = graph_from_mol(mol, with_xyz=True).to(device)
        ctx = library.query(smi)[0].to(device)
        with torch.no_grad():
            u = graph_encoder(graph).detach()
        return graph, graph.xyz, ctx, u

    for stage, epochs in [('autoencoder', a.autoencoder_epochs), ('flow', a.flow_epochs)]:
        if stage == 'flow':
            checkpoint = torch.load(out/'best_autoencoder.pt', map_location=device, weights_only=True)
            model.autoencoder.load_state_dict(checkpoint['autoencoder'])
            for parameter in model.autoencoder.parameters():
                parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=a.lr)
        best = float('inf')
        for epoch in range(epochs):
            model.train(); random.shuffle(train); total = 0.; count = 0
            for example in train:
                graph, xyz, ctx, u = get_example(example)
                if stage == 'autoencoder':
                    loss = model.reconstruction_loss(graph, xyz)
                else:
                    loss = model.fm_loss(graph, xyz, ctx, u, sigma=a.sigma_x)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step(); total += float(loss.detach()); count += 1
            model.eval(); vtotal = 0.; vcount = 0
            with torch.no_grad():
                for example in val:
                    graph, xyz, ctx, u = get_example(example)
                    loss = model.reconstruction_loss(graph, xyz) if stage=='autoencoder' else model.fm_loss(graph,xyz,ctx,u,a.sigma_x)
                    vtotal += float(loss); vcount += 1
            validation = vtotal / max(1, vcount)
            print(json.dumps({'stage':stage,'epoch':epoch+1,'train_loss':total/max(1,count),'val_loss':validation}),flush=True)
            if validation < best:
                best = validation
                ckpt = {'stage':stage, 'epoch':epoch+1,
                        'latent_dim':model.autoencoder.latent_dim, 'steps':model.steps,
                        'autoencoder':model.autoencoder.state_dict(), 'velocity':model.velocity.state_dict()}
                torch.save(ckpt, out/f'best_{stage}.pt')
    print(f'Exported best_flow.pt to {out}')


if __name__ == '__main__':
    main()
