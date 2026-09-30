#!/usr/bin/env python3
"""Evaluate released test predictions using manuscript conditions, without training."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRESSURES = np.geomspace(1e-5, 100.0, 100)


def mean_loading(lnpi, n, temperature=298.0, energy_K=None):
    """Original pressure-ratio reweighting, at fixed framework volume."""
    x = np.asarray(lnpi, dtype=float)[None, :] + np.log(PRESSURES[:, None] / 0.1) * n[None, :]
    if temperature != 298.0:
        if energy_K is None:
            raise ValueError('Temperature reweighting requires aligned energy_K')
        x = x + np.log(298.0 / temperature) * n[None, :] - energy_K[None, :] * (1.0 / temperature - 1.0 / 298.0)
    weights = np.exp(x - x.max(axis=1, keepdims=True))
    return (weights * n[None, :]).sum(axis=1) / weights.sum(axis=1)


def evaluate(temperature=298.0, output=None, verify=True):
    output = Path(output) if output else ROOT / '_outputs' / f'evaluation_{temperature:g}K'
    manifest = pd.read_csv(ROOT / 'data/mof_manifest.csv').set_index('mof_id')
    rows = []
    for gas in ('CO2', 'CH4', 'N2'):
        ids = pd.read_csv(ROOT / f'data/{gas}/mof_ids.csv')
        masks = np.load(ROOT / f'data/{gas}/mpd_masks.npz', allow_pickle=False)['valid_mask']
        truth = np.load(ROOT / f'data/{gas}/mpd_values.npz', allow_pickle=False)['lnpi']
        energies = np.load(ROOT / f'data/reweighting_energy_{gas}.npz', allow_pickle=False)
        if not np.array_equal(energies['mof_ids'], ids.mof_id.to_numpy()):
            raise ValueError(f'{gas}: energy ordering mismatch')
        id_row = {m: i for i, m in enumerate(ids.mof_id)}
        for name, label in (('pca_plus_mlp', 'PCA+MLP'), ('deeponet', 'DeepONet')):
            archive = np.load(ROOT / f'results/test_predictions/{gas}_{name}_test_predictions.npz', allow_pickle=False)
            if not np.array_equal(archive['mof_ids'], ids.loc[ids.split.eq('test'), 'mof_id'].to_numpy()):
                raise ValueError(f'{gas}/{label}: test ordering mismatch')
            for k, mof in enumerate(archive['mof_ids']):
                i = id_row[str(mof)]
                mask = archive['valid_mask'][k]
                if not np.array_equal(mask, masks[i]) or not np.allclose(archive['reference_ln_probability'][k, mask], truth[i, mask]):
                    raise ValueError(f'{gas}/{mof}: target mismatch')
                n = archive['n_axis'][mask].astype(float)
                a = np.asarray(archive['reference_ln_probability'][k, mask], float)
                b = np.asarray(archive['predicted_ln_probability'][k, mask], float)
                if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                    raise ValueError(f'{gas}/{mof}: nonfinite valid log-probability')
                energy = energies['energy_K'][i, mask]
                ua = mean_loading(a, n, temperature, energy)
                ub = mean_loading(b, n, temperature, energy)
                floor = max(1e-8, 1e-6 * float(ua.max()))
                rms = float(np.sqrt(np.mean(((ub - ua) / np.maximum(ua, floor)) ** 2)))
                rows.append(dict(gas=gas, model=label, mof_id=str(mof), parent_group_id=int(manifest.loc[str(mof), 'parent_group_id']), Nmax=int(n[-1]), tv_distance=float(0.5 * np.abs(np.exp(a) - np.exp(b)).sum()), rmsre=rms, rmsre_percent=100*rms, rmsre_floor_mean_N_supercell=floor))
    structure = pd.DataFrame(rows)
    group = structure.groupby(['gas', 'model', 'parent_group_id'], as_index=False).agg(n_structures=('mof_id', 'size'), mean_tv=('tv_distance', 'mean'), mean_rmsre=('rmsre', 'mean'))
    group['mean_rmsre_percent'] = group.mean_rmsre * 100
    summaries = []
    for (gas, model), g in group.groupby(['gas', 'model']):
        summaries.append(dict(gas=gas, model=model, n_test_parent_groups=len(g), mean_tv=g.mean_tv.mean(), tv_q25=g.mean_tv.quantile(.25), tv_q75=g.mean_tv.quantile(.75), median_rmsre_percent=g.mean_rmsre_percent.median(), rmsre_q25_percent=g.mean_rmsre_percent.quantile(.25), rmsre_q75_percent=g.mean_rmsre_percent.quantile(.75)))
    if verify:
        if temperature != 298.0:
            raise ValueError('Released manuscript-table verification applies only at 298 K')
        ref = pd.read_csv(ROOT / 'results/tables/test_structure_metrics.csv')
        merged = structure.merge(ref, on=['gas','model','mof_id'], suffixes=('_new','_released'), validate='one_to_one')
        if len(merged) != len(structure): raise ValueError('Incomplete structure-table comparison')
        for col in ('tv_distance', 'rmsre'):
            if not np.allclose(merged[col+'_new'], merged[col+'_released'], atol=1e-7, rtol=1e-6): raise ValueError(f'Released {col} differs')
        ref = pd.read_csv(ROOT / 'results/tables/test_parent_group_metrics.csv')
        merged = group.merge(ref, on=['gas','model','parent_group_id'], suffixes=('_new','_released'), validate='one_to_one')
        if len(merged) != len(group): raise ValueError('Incomplete group-table comparison')
        for col in ('mean_tv', 'mean_rmsre'):
            if not np.allclose(merged[col+'_new'], merged[col+'_released'], atol=1e-7, rtol=1e-6): raise ValueError(f'Released {col} differs')
    output.mkdir(parents=True, exist_ok=True)
    structure.to_csv(output/'test_structure_metrics.csv', index=False)
    group.to_csv(output/'test_parent_group_metrics.csv', index=False)
    pd.DataFrame(summaries).to_csv(output/'table2_summary.csv', index=False)
    (output/'settings.json').write_text(json.dumps(dict(temperature_K=temperature, reference_temperature_K=298, reference_pressure_bar=.1, pressure_min_bar=1e-5, pressure_max_bar=100, pressure_count=100, pressure_spacing='log', uptake_unit='molecules per simulation supercell', verified_against_released_298K_tables=verify), indent=2)+'\n')
    print(f'Evaluated {len(structure)} structure/model/gas rows and {len(group)} group/model/gas rows at {temperature:g} K; verification={verify}')
    return structure, group

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    evaluate(output=args.output)
