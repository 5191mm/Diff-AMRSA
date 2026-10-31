#!/usr/bin/env python3
"""Generate fresh candidate graph+coordinates with retrieval context, then refine only xyz."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from rdkit import Chem
from omegaconf import OmegaConf
from megalodon.metrics.molecule import get_molecules
from megalodon.metrics.molecule_evaluation_callback import full_atom_decoder
from diff_amrsa.conditional_model import initialize_conditioned
from diff_amrsa.chemistry import graph_from_mol, canonicalize
from diff_amrsa.graph_encoder import GraphEncoder
from diff_amrsa.flow import LatentConformerRefiner, refine_rdkit_mol


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--conditioned-checkpoint',required=True)
    p.add_argument('--library',required=True)
    p.add_argument('--split-dir',required=True)
    p.add_argument('--graph-checkpoint',required=True)
    p.add_argument('--flow-checkpoint',default=None)
    p.add_argument('--query-smiles',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--n-graphs',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=100)
    p.add_argument('--timesteps',type=int,default=250)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--disable-flow',action='store_true',help='For geometry ablation')
    p.add_argument('--zero-context',action='store_true',help='Diagnostic only, not a separately trained ablation')
    a=p.parse_args()
    torch.manual_seed(a.seed)
    cfg=OmegaConf.load(a.config)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=initialize_conditioned(cfg, str(Path(a.library).resolve())).to(device)
    state=torch.load(a.conditioned_checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state.get('state_dict',state),strict=True)
    model.eval()
    query, _=canonicalize(a.query_smiles)
    from diff_amrsa.dataset import verify_train_only_library
    verify_train_only_library(model.library, a.split_dir)
    model.set_query(query)
    ctx, retrieved=model.library.query(query)
    ctx=ctx.to(device)
    if a.zero_context:
        ctx=torch.zeros_like(ctx)
        model.injector.set_context(ctx.unsqueeze(0))
    encoder=GraphEncoder().to(device)
    encoder.load_state_dict(torch.load(a.graph_checkpoint,map_location='cpu',weights_only=True)['state_dict'])
    encoder.eval()
    refiner=None
    if not a.disable_flow:
        if not a.flow_checkpoint: raise ValueError('--flow-checkpoint required unless --disable-flow')
        flow_ckpt=torch.load(a.flow_checkpoint,map_location='cpu',weights_only=True)
        if flow_ckpt['stage']!='flow':raise ValueError('A fully trained flow checkpoint is required')
        refiner=LatentConformerRefiner(latent=flow_ckpt['latent_dim'],steps=flow_ckpt['steps']).to(device)
        refiner.autoencoder.load_state_dict(flow_ckpt['autoencoder'])
        refiner.velocity.load_state_dict(flow_ckpt['velocity']);refiner.eval()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    writer_initial=Chem.SDWriter(str(out/'initial.sdf'))
    writer_refined=Chem.SDWriter(str(out/'refined.sdf'))
    summary={'requested':a.n_graphs,'valid_initial':0,'attempted_refine':0,'refine_success':0,'written_candidates':0,'failures':[],
             'query_smiles':query,'retrieval':retrieved,'timesteps':a.timesteps,'flow_enabled':not a.disable_flow,'zero_context_diagnostic':a.zero_context}
    for start in range(0,a.n_graphs,a.batch_size):
        amount=min(a.batch_size,a.n_graphs-start)
        with torch.no_grad():
            generated=model.sample(amount,timesteps=a.timesteps)
        generated['x']=generated['x'] * float(cfg.evaluation.scale_coords)
        molecules=get_molecules(generated, {'atom_decoder':full_atom_decoder})
        for j,wrapped in enumerate(molecules):
            mol=wrapped.raw_rdkit_mol
            try:
                Chem.SanitizeMol(mol)
                writer_initial.write(mol)
                summary['valid_initial']+=1
                if refiner is None:
                    refined=Chem.Mol(mol)
                else:
                    graph=graph_from_mol(mol,with_xyz=True).to(device)
                    with torch.no_grad():
                        embed=encoder(graph)
                    summary['attempted_refine']+=1
                    refined=refine_rdkit_mol(mol,graph,ctx,embed,refiner)
                writer_refined.write(refined)
                summary['written_candidates']+=1
                if refiner is not None:
                    summary['refine_success']+=1
            except Exception as err:
                summary['failures'].append({'index':start+j,'error':str(err)[:220]})
        print(f'Processed {start+amount}/{a.n_graphs}',flush=True)
    writer_initial.close();writer_refined.close()
    (out/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('retrieval','failures')},indent=2))


if __name__=='__main__':
    main()
