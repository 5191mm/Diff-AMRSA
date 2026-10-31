#!/usr/bin/env python3
"""RF activity probability and active-only exact MIC regressor using the original train partition."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from diff_amrsa.chemistry import canonicalize
from diff_amrsa.dataset import read_split

from diff_amrsa.activity_features import feature


def main():
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    import joblib
    p=argparse.ArgumentParser()
    p.add_argument('--split-dir',required=True)
    p.add_argument('--activity-table',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--n-estimators',type=int,default=200)
    a=p.parse_args()
    groups={k:read_split(Path(a.split_dir)/f'{k}.jsonl') for k in ('train','val','test')}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if {r['smiles'] for r in groups[left]} & {r['smiles'] for r in groups[right]}:
            raise ValueError(f'{left}/{right} identities overlap')
    table={}
    with open(a.activity_table,encoding='utf-8-sig',newline='') as file:
        for row in csv.DictReader(file):
            try:
                smi,_=canonicalize(row['smiles'])
                table[smi]=row
            except (ValueError,TypeError):pass
    X={};Y={};M={}
    for key,rows in groups.items():
        usable=[r for r in rows if r['smiles'] in table]
        if not usable:
            raise ValueError(f'No labeled structures found in {key}; check data split and activity table')
        X[key]=np.stack([feature(Chem.MolFromSmiles(r['smiles'])) for r in usable])
        Y[key]=np.asarray([r['activity'] for r in usable],dtype=int)
        M[key]=[table[r['smiles']] for r in usable]
    clf=RandomForestClassifier(n_estimators=a.n_estimators,random_state=42,n_jobs=-1)
    clf.fit(X['train'],Y['train'])
    def eligible(row):
        try:
            mic=float(row['pmic_ug_ml'])
            return int(float(row['activity_label']))==1 and int(float(row['pmic_regression_mask']))==1 and 0<mic<100
        except (TypeError,ValueError,KeyError):return False
    idx_train=[i for i,r in enumerate(M['train']) if eligible(r)]
    idx_val=[i for i,r in enumerate(M['val']) if eligible(r)]
    if not idx_train:
        raise ValueError('No eligible training MIC records; regression requires active and exact MIC labels')
    reg=RandomForestRegressor(n_estimators=a.n_estimators,random_state=42,n_jobs=-1)
    reg.fit(X['train'][idx_train],np.log10([float(M['train'][i]['pmic_ug_ml']) for i in idx_train]))
    metrics={'train_size':len(Y['train']),'val_size':len(Y['val']),'test_size':len(Y['test']),
             'mic_train_size':len(idx_train),'mic_val_size':len(idx_val)}
    for split in ('val','test'):
        if len(set(Y[split]))==2:
            metrics[f'{split}_auc']=float(roc_auc_score(Y[split],clf.predict_proba(X[split])[:,1]))
    if idx_val:
        metrics['val_mic_log10_mae']=float(mean_absolute_error(
            np.log10([float(M['val'][i]['pmic_ug_ml']) for i in idx_val]),reg.predict(X['val'][idx_val])))
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    joblib.dump({'classifier':clf,'regressor':reg,'feature_name':'morgan2048_rdkit9'},out/'activity_proxy.joblib')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
