#!/usr/bin/env python3
"""For retrieval-only ablation: either frozen random graph encoder or trained contrastive encoder."""
import argparse
from pathlib import Path
import torch
from diff_amrsa.dataset import read_split
from diff_amrsa.graph_encoder import GraphEncoder
from diff_amrsa.retrieval import make_library, save_library


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--graph-checkpoint',default=None)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    torch.manual_seed(42)
    encoder=GraphEncoder().to(a.device)
    if a.graph_checkpoint:
        encoder.load_state_dict(torch.load(a.graph_checkpoint,map_location=a.device,weights_only=True)['state_dict'])
    train=read_split(Path(a.split_dir)/'train.jsonl')
    rows,embeddings=make_library([r['smiles'] for r in train],[r['activity'] for r in train],encoder,a.device)
    save_library(a.output,rows,embeddings,{'scope':'train_only',
                'encoder':'trained_contrastive' if a.graph_checkpoint else 'untrained_random_ablation',
                'weights':[0.4,0.2,0.2,0.2], 'k':5,'tau_r':0.2})
    print(f'Wrote {len(rows)} training references to {a.output}')


if __name__=='__main__':main()
