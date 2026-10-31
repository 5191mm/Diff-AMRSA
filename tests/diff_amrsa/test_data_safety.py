import json
import numpy as np
import pytest
from diff_amrsa.dataset import verify_train_only_library
from diff_amrsa.retrieval import RetrievalLibrary


def test_train_only_libraries_are_verified(tmp_path):
    for split, values in [('train',['CCO','CCN']),('val',['CCC']),('test',['CO'])]:
        with open(tmp_path/f'{split}.jsonl','w') as f:
            for smi in values:
                f.write(json.dumps({'smiles':smi})+'\n')
    rows=[{'smiles':'CCO','scaffold':'','fragments':[],'activity':1}]
    lib=RetrievalLibrary(rows,np.zeros((1,128),dtype=np.float32))
    assert verify_train_only_library(lib,tmp_path)==1
    bad=RetrievalLibrary(rows+[{'smiles':'CCC','scaffold':'','fragments':[],'activity':0}],np.zeros((2,128),dtype=np.float32))
    with pytest.raises(ValueError,match='val/test'):
        verify_train_only_library(bad,tmp_path)
