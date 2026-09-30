#!/usr/bin/env python3
"""Evaluate low-rank reconstruction at 273 K on train+validation only."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from evaluate_manuscript_298K import ROOT, mean_loading

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranks',nargs='+',type=int,default=[30])
    p.add_argument('--fits-root',type=Path,default=ROOT/'results/models/indirect/low_rank',help='GAS/ files for one rank, or GAS/rank_XX/ files for multiple ranks')
    p.add_argument('--output',type=Path,default=ROOT/'_outputs/rank_evaluation_273K')
    args=p.parse_args()
    counts=[]
    for gas in ('CO2','CH4','N2'):
        ids=pd.read_csv(ROOT/f'data/{gas}/mof_ids.csv')
        values=np.load(ROOT/f'data/{gas}/mpd_values.npz',allow_pickle=False)
        masks=np.load(ROOT/f'data/{gas}/mpd_masks.npz',allow_pickle=False)['valid_mask']
        energy=np.load(ROOT/f'data/reweighting_energy_{gas}.npz',allow_pickle=False)
        if not np.array_equal(energy['mof_ids'],ids.mof_id.to_numpy()):raise ValueError('Energy order mismatch')
        selected=np.where(ids.split.isin(['train','val']))[0]
        if len(selected)!=1886:raise ValueError('Rank population is not 1886')
        for rank in args.ranks:
            folder=args.fits_root/gas/f'rank_{rank:02d}'
            if not folder.exists():folder=args.fits_root/gas
            model_path=folder/'masked_low_rank_model.npz';coef_path=folder/'latent_coefficients.csv'
            if not model_path.exists() or not coef_path.exists():raise FileNotFoundError(f'Rank {rank} artifacts missing for {gas}; supply --fits-root')
            model=np.load(model_path,allow_pickle=False)
            if model['basis'].shape[1]!=rank:raise ValueError(f'{gas}: requested rank {rank}, but {model_path} has rank {model["basis"].shape[1]}')
            coef=pd.read_csv(coef_path).set_index('mof_id')
            totals=dict(both_fail_count=0,tv_only_failure_count=0,rmsre_only_failure_count=0,both_pass_count=0)
            for i in selected:
                mask=masks[i];n=values['n_axis'][mask].astype(float);a=np.asarray(values['lnpi'][i,mask],float)
                a=a-np.logaddexp.reduce(a)
                b=(model['mean_curve']+(model['basis'] * coef.loc[ids.mof_id[i],[f'LC{k}' for k in range(1,rank+1)]].to_numpy(float)[None, :]).sum(axis=1))[mask]
                b=b-np.logaddexp.reduce(b)
                ua=mean_loading(a,n,273.,energy['energy_K'][i,mask]);ub=mean_loading(b,n,273.,energy['energy_K'][i,mask])
                tv=.5*np.abs(np.exp(a)-np.exp(b)).sum();rms=np.sqrt(np.mean(((ub-ua)/np.maximum(ua,max(1e-8,1e-6*ua.max())))**2))
                tvpass=tv<=.05;rmspass=rms<=.2
                key='both_pass_count' if tvpass and rmspass else 'both_fail_count' if not tvpass and not rmspass else 'tv_only_failure_count' if not tvpass else 'rmsre_only_failure_count'
                totals[key]+=1
            row=dict(gas=gas,rank=rank,n_structures=len(selected),**totals)
            row.update({k.replace('_count','_percent'):100*v/len(selected) for k,v in totals.items()})
            counts.append(row)
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(counts).to_csv(args.output/'rank_contingency_counts.csv',index=False)
    (args.output/'settings.json').write_text(json.dumps(dict(temperature_K=273.,population='training + validation',population_size=1886,ranks=args.ranks,reference_temperature_K=298.,reference_pressure_bar=.1,pressure_grid='100 log-spaced points 1e-5 to 100 bar',historical_count_table_overwritten=False),indent=2)+'\n')
    print('Rank evaluation complete; historical released counts unchanged')
if __name__=='__main__':main()
