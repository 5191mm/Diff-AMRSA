#!/usr/bin/env python3
"""Train g_phi by paper-style supervised contrastive loss; export train-only retrieval library."""
import argparse
import json
import random
from pathlib import Path
import torch
from rdkit import Chem
from diff_amrsa.chemistry import graph_from_mol
from diff_amrsa.dataset import read_split
from diff_amrsa.graph_encoder import GraphEncoder, contrastive_batch_indices, contrastive_loss
from diff_amrsa.retrieval import make_library, save_library


def evaluate(encoder, rows, device, tau, max_batches=15):
    try:
        indices_iter = contrastive_batch_indices(rows, batch_size=32, seed=991)
    except ValueError:
        return float('nan'), 0
    losses, anchors = [], 0
    with torch.no_grad():
        for i, indices in enumerate(indices_iter):
            if i >= max_batches:
                break
            subset = [rows[j] for j in indices]
            embeddings = torch.stack([encoder(graph_from_mol(Chem.MolFromSmiles(r['smiles'])).to(device)) for r in subset])
            loss, valid = contrastive_loss(embeddings, [r['scaffold'] for r in subset],
                                            [r['activity'] for r in subset], tau)
            if valid:
                losses.append(float(loss.cpu()))
                anchors += valid
    return (sum(losses)/len(losses), anchors) if losses else (float('nan'),0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--tau-c', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    train, val = read_split(Path(a.split_dir)/'train.jsonl'), read_split(Path(a.split_dir)/'val.jsonl')
    if {r['smiles'] for r in train} & {r['smiles'] for r in val}:
        raise ValueError('Train/validation overlap detected')
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    encoder = GraphEncoder().to(a.device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=a.lr)
    best = float('inf')
    history = []
    # Validation is evaluated on labeled pairs only; no test information is used
    for epoch in range(a.epochs):
        encoder.train(); total, count = 0., 0
        for indices in contrastive_batch_indices(train, a.batch_size, seed=a.seed+epoch):
            rows = [train[i] for i in indices]
            embs = torch.stack([encoder(graph_from_mol(Chem.MolFromSmiles(r['smiles'])).to(a.device)) for r in rows])
            loss, valid = contrastive_loss(embs, [r['scaffold'] for r in rows],
                                           [r['activity'] for r in rows], a.tau_c)
            if not valid:
                continue
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()); count += 1
        encoder.eval()
        valid_loss, val_anchors = evaluate(encoder, val, a.device, a.tau_c)
        if val_anchors == 0:
            raise ValueError('Validation has no eligible scaffold+activity positive pairs; define valid split with eligible pairs')
        entry = {'epoch': epoch+1, 'train_loss': total/max(count,1), 'val_loss': valid_loss, 'val_anchors': val_anchors}
        print(json.dumps(entry), flush=True); history.append(entry)
        if valid_loss < best:
            best = valid_loss
            torch.save({'state_dict': encoder.state_dict(), 'epoch': epoch+1, 'tau_c': a.tau_c}, out/'graph_encoder.pt')
    checkpoint = torch.load(out/'graph_encoder.pt', map_location=a.device, weights_only=True)
    encoder.load_state_dict(checkpoint['state_dict']); encoder.eval()
    rows, embeddings = make_library([r['smiles'] for r in train], [r['activity'] for r in train], encoder, device=a.device)
    save_library(out/'train_retrieval.npz', rows, embeddings,
                 {'scope':'train_only', 'encoder_epoch': checkpoint['epoch'], 'top_k': 5, 'tau_r':0.2})
    (out/'graph_training.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(f'Train references: {len(rows)}; exported to {out / "train_retrieval.npz"}')


if __name__ == '__main__':
    main()
