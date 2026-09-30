#!/usr/bin/env python3
from pathlib import Path
import importlib.util,json,numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parents[1]
def lm(n,f):
 s=importlib.util.spec_from_file_location(n,R/'code'/f);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
pca=lm('pca','train_masked_pca_mlp.py');deep=lm('deep','train_deeponet_absolute_n.py')
for g in ('CO2','CH4','N2'):
 d=R/'data'/g;i=pd.read_csv(d/'mof_ids.csv');f=np.load(d/'feature_values.npz');v=np.load(d/'mpd_values.npz');m=np.load(d/'mpd_masks.npz')['valid_mask']
 assert len(i)==2095 and i.split.value_counts().to_dict()=={'train':1676,'val':210,'test':209}
 assert v['lnpi'].shape==m.shape and np.array_equal(np.isfinite(v['lnpi']),m) and np.array_equal(m.sum(1)-1,i.Nmax.to_numpy())
 e=np.load(R/'data'/f'reweighting_energy_{g}.npz');assert np.array_equal(e['mof_ids'].astype(str),i.mof_id.astype(str))
 rows=np.where(i.split.eq('test'))[0][:2]
 c=json.loads((R/'results/models/indirect/pca_mlp'/g/'config.json').read_text());fs=np.load(R/'results/models/indirect/pca_mlp'/g/'feature_scaler.npz');cs=np.load(R/'results/models/indirect/pca_mlp'/g/'coefficient_scaler.npz');x=(f['features'][rows]-fs['mean'])/fs['scale']
 pm=pca.MLP(x.shape[1],30,c['hidden'],c['dropout']);pm.load_state_dict(torch.load(R/'results/models/indirect/pca_mlp'/g/'best_model.pt',map_location='cpu',weights_only=True));pm.eval()
 with torch.no_grad():coef=pm(torch.tensor(x,dtype=torch.float32)).numpy()*cs['scale']+cs['mean']
 lr=np.load(R/'results/models/indirect/low_rank'/g/'masked_low_rank_model.npz');assert (lr['mean_curve'][None,:]+coef@lr['basis'].T).shape==(2,206)
 dc=json.loads((R/'results/models/direct/deeponet'/g/'config.json').read_text());ds=np.load(R/'results/models/direct/deeponet'/g/'feature_scaler.npz');x=(f['features'][rows]-ds['mean'])/ds['scale'];dm=deep.DeepONet(x.shape[1],dc['p'],dc['hidden_dim']);dm.load_state_dict(torch.load(R/'results/models/direct/deeponet'/g/'best_model.pt',map_location='cpu',weights_only=True));dm.eval()
 with torch.no_grad(): pred=deep.masked_log_softmax(dm(torch.tensor(x,dtype=torch.float32),torch.tensor(v['n_axis'][:,None]/205,dtype=torch.float32)),torch.tensor(m[rows])).numpy()
 assert pred.shape==(2,206)
 j=rows[0];q=m[j];ln=v['lnpi'][j,q];en=e['energy_K'][j,q];n=np.arange(q.sum());z=ln+np.log(100/.1)*n+np.log(298/273)*n-en*(1/273-1/298);z-=z.max();w=np.exp(z);assert np.isfinite((w*n).sum()/w.sum())
print('SMOKE TEST PASS: data/order/splits, PCA+MLP inference, DeepONet inference, reweighting')
