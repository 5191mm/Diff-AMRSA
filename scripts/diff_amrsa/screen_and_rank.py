#!/usr/bin/env python3
"""Generate evidence-based tables from real SDF predictions; Pareto ranking is not retrieval scoring."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from rdkit import Chem
from diff_amrsa.chemistry import canonicalize
from diff_amrsa.retrieval import RetrievalLibrary
from diff_amrsa.screening import descriptors_for_ranking, domain_descriptors, geometric_checks,posebusters_inspired,pareto_rank


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sdf',required=True)
    p.add_argument('--library',required=True)
    p.add_argument('--activity-proxy',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--top-k',type=int,default=100)
    p.add_argument('--min-qed',type=float,default=0.3)
    p.add_argument('--require-uff',action='store_true')
    a=p.parse_args()
    import joblib
    from diff_amrsa.activity_features import feature
    proxy=joblib.load(a.activity_proxy)
    library=RetrievalLibrary.load(a.library)
    rows=[]; seen=set(); counts={'raw':0,'invalid':0,'duplicate':0,'filtered':0,'accepted':0}
    for mol in Chem.SDMolSupplier(a.sdf,removeHs=False,sanitize=False):
        counts['raw']+=1
        try:
            if mol is None: raise ValueError('invalid mol')
            Chem.SanitizeMol(mol)
            if len(Chem.GetMolFrags(mol))!=1:raise ValueError('disconnected')
            smi,_=canonicalize(Chem.MolToSmiles(mol))
            if smi in seen:
                counts['duplicate']+=1;continue
            seen.add(smi)
            d=descriptors_for_ranking(mol)
            geo=geometric_checks(mol)
            geom_flags=posebusters_inspired(mol)
            if d['QED']<a.min_qed or (a.require_uff and not geo['uff_success']):
                counts['filtered']+=1;continue
            dom=domain_descriptors(mol,library)
            x=feature(mol).reshape(1,-1)
            activity=float(proxy['classifier'].predict_proba(x)[0,1])
            mic=float(proxy['regressor'].predict(x)[0])
            confidence=abs(activity-0.5)*2
            # Reliability is a recorded guardrail; activity probability is a proxy, not measured MIC
            reliability=bool(geo['finite_coords'] and confidence>=0.1)
            if not reliability:
                counts['filtered']+=1;continue
            rows.append({'smiles':smi,**d,**geo,**geom_flags,**dom,'activity_probability':activity,
                         'predicted_log10_mic_ug_ml':mic,
                         'domain_relevance':0.5*dom['tanimoto_max']+0.5*dom['fragment_coverage'],
                         'activity_confidence':confidence})
            counts['accepted']+=1
        except (ValueError,RuntimeError,TypeError):
            counts['invalid']+=1
    ranked=pareto_rank(rows)
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    if ranked:
        with open(out/'all_ranked.csv','w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(ranked[0]));writer.writeheader();writer.writerows(ranked)
        with open(out/'top100.csv','w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(ranked[0]));writer.writeheader();writer.writerows(ranked[:a.top_k])
    (out/'summary.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
