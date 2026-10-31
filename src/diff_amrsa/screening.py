"""Multiobjective candidate screening and reliability-constrained Pareto prioritization."""
from __future__ import annotations
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from .chemistry import descriptors, fingerprint


def descriptors_for_ranking(mol):
    return {'MW':float(Descriptors.MolWt(mol)), 'LogP':float(Crippen.MolLogP(mol)),
            'TPSA':float(rdMolDescriptors.CalcTPSA(mol)), 'QED':float(QED.qed(mol)),
            'HBA':int(rdMolDescriptors.CalcNumHBA(mol)), 'HBD':int(rdMolDescriptors.CalcNumHBD(mol)),
            'RotBonds':int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            'Rings':int(rdMolDescriptors.CalcNumRings(mol))}


def geometric_checks(mol):
    """Simple geometry/force-field controls; not a substitute for PoseBusters."""
    out={'finite_coords':False,'uff_success':False,'uff_rmsd':None}
    if not mol.GetNumConformers(): return out
    try:
        positions=np.asarray(mol.GetConformer().GetPositions())
        if positions.shape!=(mol.GetNumAtoms(),3) or not np.isfinite(positions).all():
            return out
        out['finite_coords']=True
        if not AllChem.UFFHasAllMoleculeParams(mol): return out
        m=Chem.Mol(mol)
        status=AllChem.UFFOptimizeMolecule(m,maxIters=250)
        out['uff_success']=(status==0)
        if status==0:
            # Proper-alignment RMSD, original and relaxed atom order preserved
            from .chemistry import proper_kabsch_rmsd
            import torch
            before=torch.as_tensor(positions,dtype=torch.float64)
            after=torch.as_tensor(np.asarray(m.GetConformer().GetPositions()).copy(),dtype=torch.float64)
            out['uff_rmsd']=float(proper_kabsch_rmsd(before,after))
    except (ValueError,RuntimeError):
        return out
    return out


def domain_descriptors(mol, reference_library):
    d=descriptors(mol)
    similarities=DataStructs.BulkTanimotoSimilarity(fingerprint(mol),reference_library.fps)
    fragments=set(d['fragments'])
    covered_reference_fragments=set().union(*(set(r['fragments']) for r in reference_library.rows))
    return {'tanimoto_max':float(max(similarities)),
            'scaffold_seen':bool(d['scaffold']) and any(d['scaffold']==r['scaffold'] for r in reference_library.rows),
            'fragment_coverage':float(len(fragments & covered_reference_fragments)/max(1,len(fragments))),
            'reference_scaffold':d['scaffold']}



def posebusters_inspired(mol):
    """Transparent heuristic analogue of six 3D checks, NOT the PoseBusters implementation.

    Reference thresholds should be replaced by an identical PoseBusters config for publication.
    """
    import math
    flags = {k: False for k in ('bond_lengths','bond_angles','no_clashes','ring_planarity','valence','energy')}
    if not mol.GetNumConformers():
        return {**flags, 'pb_inspired_all':False}
    try:
        checked = Chem.Mol(mol)
        Chem.SanitizeMol(checked)
        flags['valence'] = True
        xyz = np.asarray(mol.GetConformer().GetPositions())
        if xyz.shape != (mol.GetNumAtoms(),3) or not np.isfinite(xyz).all():
            return {**flags, 'pb_inspired_all':False}
        periodic = Chem.GetPeriodicTable()
        flags['bond_lengths'] = True
        for bond in mol.GetBonds():
            i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            a,b=mol.GetAtomWithIdx(i),mol.GetAtomWithIdx(j)
            cov=periodic.GetRcovalent(a.GetAtomicNum())+periodic.GetRcovalent(b.GetAtomicNum())
            if cov<=0:continue
            dist=np.linalg.norm(xyz[i]-xyz[j])
            if dist < 0.65*cov or dist > 1.65*cov:
                flags['bond_lengths']=False
                break
        flags['bond_angles'] = True
        for atom in mol.GetAtoms():
            if atom.IsInRingSize(3) or atom.IsInRingSize(4):
                continue
            neighbors=[n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum()>1]
            if len(neighbors)<2:continue
            c=atom.GetIdx();hyb=atom.GetHybridization()
            ideal = 180. if hyb==Chem.HybridizationType.SP else 120. if hyb==Chem.HybridizationType.SP2 else 109.5
            tolerance=45. if ideal==109.5 else 35.
            for pos,i in enumerate(neighbors):
                for j in neighbors[pos+1:]:
                    u,v=xyz[i]-xyz[c],xyz[j]-xyz[c]
                    denom=np.linalg.norm(u)*np.linalg.norm(v)
                    if denom<1e-8:
                        flags['bond_angles']=False;break
                    angle=math.degrees(math.acos(float(np.clip(u.dot(v)/denom,-1,1))))
                    if abs(angle-ideal)>tolerance:
                        flags['bond_angles']=False;break
                if not flags['bond_angles']:break
            if not flags['bond_angles']:break
        excluded=set()
        for atom in mol.GetAtoms():
            i=atom.GetIdx()
            excluded.add((i,i))
            for neighbor in atom.GetNeighbors():
                j=neighbor.GetIdx()
                excluded.add(tuple(sorted((i,j))))
                for neighbor2 in neighbor.GetNeighbors():
                    excluded.add(tuple(sorted((i,neighbor2.GetIdx()))))
        heavy=[atom for atom in mol.GetAtoms() if atom.GetAtomicNum()>1]
        flags['no_clashes']=True
        for pos,a in enumerate(heavy):
            for b in heavy[pos+1:]:
                i,j=a.GetIdx(),b.GetIdx()
                if (i,j) in excluded:continue
                vdW=periodic.GetRvdw(a.GetAtomicNum())+periodic.GetRvdw(b.GetAtomicNum())
                if np.linalg.norm(xyz[i]-xyz[j])<0.58*vdW:
                    flags['no_clashes']=False;break
            if not flags['no_clashes']:break
        flags['ring_planarity']=True
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring)<5 or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):continue
            points=xyz[list(ring)];centered=points-points.mean(0)
            _,_,vectors=np.linalg.svd(centered,full_matrices=False)
            if np.max(np.abs(centered@vectors[-1]))>0.25:
                flags['ring_planarity']=False;break
        flags['energy']=False
        if AllChem.UFFHasAllMoleculeParams(mol):
            initial=AllChem.UFFGetMoleculeForceField(mol).CalcEnergy()
            optimized=Chem.Mol(mol)
            if AllChem.UFFOptimizeMolecule(optimized,maxIters=300)==0:
                minimum=AllChem.UFFGetMoleculeForceField(optimized).CalcEnergy()
                flags['energy']=bool(np.isfinite(initial) and np.isfinite(minimum)
                                     and initial-minimum<=100.+0.5*mol.GetNumHeavyAtoms())
    except (ValueError,RuntimeError):
        pass
    return {**flags,'pb_inspired_all':all(flags.values())}

def pareto_rank(rows, directions=('activity_probability','QED','domain_relevance')):
    """Pareto non-dominated sorting followed by balanced normalized tie-break, no hidden paper numbers."""
    if not rows:return []
    vals=np.asarray([[float(r[k]) for k in directions] for r in rows],dtype=np.float64)
    if not np.isfinite(vals).all():
        raise ValueError('Candidate objectives must be finite')
    minv=vals.min(0); maxv=vals.max(0)
    norm=(vals-minv)/np.maximum(maxv-minv,1e-8)
    remaining=set(range(len(rows)))
    fronts=[]
    while remaining:
        front=[]
        for i in sorted(remaining):
            if not any(np.all(vals[j]>=vals[i]) and np.any(vals[j]>vals[i]) for j in remaining if j!=i):
                front.append(i)
        if not front:
            raise RuntimeError('Pareto sorting failed')
        fronts.append(front)
        remaining.difference_update(front)
    result=[]
    for level, front in enumerate(fronts,1):
        for index in sorted(front,key=lambda i:(-float(norm[i].mean()),i)):
            result.append({**rows[index], 'pareto_front':level,'balance_score':float(norm[index].mean())})
    return result
