#!/usr/bin/env python3
"""Build new-split geom5+Nmax+energy datasets for PCA+MLP and DeepONet."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
PROJECT = STAGE.parent
ROOT = PROJECT.parent
GASES = ("CO2", "CH4", "N2")
ENERGY_TABLES = {
    "CO2": ROOT / "energy_histogram_ablation/02_feature_tables/CO2/co2_mof_level_energy_hist_mask0p5_kjmol.csv",
    "CH4": ROOT / "energy_histogram_ablation/02_feature_tables/CH4/ch4_mof_level_energy_hist_mask0p5_kjmol.csv",
    "N2": ROOT / "energy_histogram_ablation/02_feature_tables/N2/n2_mof_level_energy_hist_mask0p5_kjmol.csv",
}


def main() -> None:
    manifest = {}
    output_root = STAGE / "data"
    output_root.mkdir(parents=True, exist_ok=True)
    for gas in GASES:
        source = ROOT / "Nmax_ablation" / gas / "protocols/diversity_aware/data"
        current = PROJECT / "03_pca_sensitivity_273K" / "inputs" / gas
        target = output_root / gas
        target.mkdir(parents=True, exist_ok=True)

        old_index = pd.read_csv(source / "mof_ids.csv", dtype={"mof_id": str})
        index = pd.read_csv(current / "mof_ids.csv", dtype={"mof_id": str})
        if old_index.mof_id.duplicated().any() or index.mof_id.duplicated().any():
            raise ValueError(f"{gas}: duplicate MOF IDs")
        missing_base = sorted(set(index.mof_id) - set(old_index.mof_id))
        if missing_base:
            raise ValueError(f"{gas}: {len(missing_base)} MOFs lack base features")

        base = np.load(source / "feature_values.npz", allow_pickle=False)
        base_values = np.asarray(base["features"], float)
        base_names = [str(x) for x in base["feature_names"]]
        row_map = pd.Series(np.arange(len(old_index)), index=old_index.mof_id)
        old_rows = row_map.loc[index.mof_id].to_numpy(int)
        base_values = base_values[old_rows]
        if "Nmax" not in old_index:
            raise ValueError(f"{gas}: base index does not contain Nmax")
        index["Nmax"] = old_index.iloc[old_rows]["Nmax"].to_numpy()

        energy = pd.read_csv(ENERGY_TABLES[gas], dtype={"mof_id": str})
        if energy.mof_id.duplicated().any():
            raise ValueError(f"{gas}: duplicate energy-feature IDs")
        missing_energy = sorted(set(index.mof_id) - set(energy.mof_id))
        if missing_energy:
            raise ValueError(f"{gas}: {len(missing_energy)} MOFs lack energy features")
        energy = energy.set_index("mof_id").loc[index.mof_id]
        energy_values = energy.to_numpy(float)
        if not np.isfinite(base_values).all() or not np.isfinite(energy_values).all():
            raise ValueError(f"{gas}: non-finite model features")

        features = np.column_stack((base_values, energy_values))
        feature_names = base_names + list(energy.columns)
        np.savez_compressed(
            target / "feature_values.npz",
            features=features,
            feature_names=np.asarray(feature_names, dtype="U"),
        )
        current_values = np.load(current / "mpd_values.npz", allow_pickle=False)
        lnpi = np.asarray(current_values["lnpi"], float)
        np.savez_compressed(target / "mpd_values.npz", lnpi=lnpi,
                            n_axis=np.arange(lnpi.shape[1], dtype=int))
        shutil.copy2(current / "mpd_masks.npz", target / "mpd_masks.npz")
        index.to_csv(target / "mof_ids.csv", index=False)

        splits = {
            split: index.loc[index.split.eq(split), "mof_id"].tolist()
            for split in ("train", "val", "test")
        }
        (target / "splits.json").write_text(
            json.dumps(splits, indent=2) + "\n", encoding="utf-8"
        )
        manifest[gas] = {
            "structures": len(index),
            "split_counts": {k: len(v) for k, v in splits.items()},
            "base_feature_count": len(base_names),
            "energy_feature_count": len(energy.columns),
            "total_feature_count": features.shape[1],
            "feature_names": feature_names,
            "base_feature_source": str(source / "feature_values.npz"),
            "energy_feature_source": str(ENERGY_TABLES[gas]),
            "mpd_source": str(current),
        }

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()



