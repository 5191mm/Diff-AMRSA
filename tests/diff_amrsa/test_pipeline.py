import numpy as np
import pytest
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem
from diff_amrsa.chemistry import canonicalize, graph_from_mol, proper_kabsch_rmsd
from diff_amrsa.graph_encoder import GraphEncoder, contrastive_loss
from diff_amrsa.retrieval import RetrievalLibrary, make_library
from diff_amrsa.flow import LatentConformerRefiner, refine_rdkit_mol
from diff_amrsa.conditioning import ContextInjector
from diff_amrsa.screening import pareto_rank


def test_canonical_and_graph():
    smi, mol = canonicalize('Cl.CCO')
    assert smi == 'CCO'
    graph = graph_from_mol(mol)
    assert graph.n == 3 and graph.edges.shape == (2, 4)


def test_supervised_contrastive_gradient():
    model = GraphEncoder(dim=16, out_dim=8)
    molecules = [Chem.MolFromSmiles(s) for s in ('CCO', 'CCCO', 'CCN', 'CCCN')]
    emb = torch.stack([model(graph_from_mol(m)) for m in molecules])
    loss, valid = contrastive_loss(emb, ['scaf','scaf','scaf','other'], [1,1,0,0], .1)
    assert valid == 2 and torch.isfinite(loss)
    loss.backward()
    assert model.atom_emb.weight.grad is not None


def test_library_topk_weighting_no_identical():
    enc = GraphEncoder(dim=16, out_dim=8)
    rows, embeddings = make_library(['CCO','CCN','c1ccccc1','CCC'], [1,0,1,0], enc)
    lib = RetrievalLibrary(rows, embeddings, top_k=2)
    context, retrieved = lib.query('CCO')
    assert context.shape == (8,)
    assert all(r['smiles'] != 'CCO' for r in retrieved)
    assert sum(r['weight'] for r in retrieved) == pytest.approx(1.)
    assert all(0. <= r['score'] <= 1. for r in retrieved)


def test_kabsch_proper_rotation():
    x = torch.tensor([[1.,0,0],[-1.,0,0],[0,2,0],[0,0,3.]])
    rotation = torch.tensor([[0.,1.,0],[-1.,0,0],[0,0,1.]])
    assert float(proper_kabsch_rmsd(x@rotation + 12, x)) < 1e-4
    assert float(proper_kabsch_rmsd(x*torch.tensor([-1.,1.,1.]), x)) > 0.01


def test_flow_preserves_bonds():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    assert AllChem.EmbedMolecule(mol, randomSeed=23) == 0
    graph = graph_from_mol(mol, with_xyz=True)
    model = LatentConformerRefiner(latent=8, cond_dim=6, hidden=16, steps=5)
    assert torch.isfinite(model.reconstruction_loss(graph, graph.xyz))
    loss = model.fm_loss(graph, graph.xyz, torch.randn(6), torch.randn(6))
    assert torch.isfinite(loss)
    refined = refine_rdkit_mol(mol,graph,torch.randn(6),torch.randn(6),model)
    assert refined.GetNumBonds() == mol.GetNumBonds()
    assert refined.GetNumAtoms() == mol.GetNumAtoms()
    for b1,b2 in zip(refined.GetBonds(), mol.GetBonds()):
        assert b1.GetBondType()==b2.GetBondType()


class DummyDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_embedder=nn.Linear(2,4)

    def forward(self,batch,time):
        return self.atom_embedder(batch['h_t'])


def test_condition_injector_changes_denoiser_output():
    net=DummyDynamics()
    injector=ContextInjector(context_dim=3,hidden_dim=4)
    injector.attach(net)
    with torch.no_grad():
        injector.net[-1].weight.normal_(0,0.1)
    batch={'h_t':torch.ones(3,2),'batch':torch.tensor([0,0,1])}
    injector.set_context(torch.zeros(2,3));a=net(batch,None).detach()
    injector.set_context(torch.ones(2,3));b=net(batch,None).detach()
    assert not torch.allclose(a,b)
    injector.detach()


def test_pareto_ranking():
    rows=[{'activity_probability':0.8,'QED':0.7,'domain_relevance':0.6},
          {'activity_probability':0.6,'QED':0.7,'domain_relevance':0.4},
          {'activity_probability':0.9,'QED':0.1,'domain_relevance':0.7}]
    ranked=pareto_rank(rows)
    assert ranked[0]['pareto_front']==1
    assert ranked[-1]['pareto_front']==2


def test_pb_inspired_is_not_exact_posebusters():
    from diff_amrsa.screening import posebusters_inspired
    mol=Chem.AddHs(Chem.MolFromSmiles('CCO'))
    AllChem.EmbedMolecule(mol,randomSeed=4)
    flags=posebusters_inspired(mol)
    assert {'bond_lengths','bond_angles','no_clashes','ring_planarity','valence','energy','pb_inspired_all'} <= flags.keys()
    assert all(isinstance(v,bool) for v in flags.values())
