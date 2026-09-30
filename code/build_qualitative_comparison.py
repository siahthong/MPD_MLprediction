#!/usr/bin/env python3
"""Build paired diagnostics and representative current-model comparisons."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
PROJECT = STAGE.parent
ROOT = PROJECT.parent
OUT = STAGE / "qualitative_model_comparison"
sys.path.insert(0, str(PROJECT / "03_pca_sensitivity_273K/scripts"))
import evaluate_and_export_273K as canonical  # noqa: E402

GASES = ("CO2", "CH4", "N2")
COLORS = {"true": "black", "PCA+MLP": "#0072B2", "DeepONet": "#D55E00"}
MODEL_FILES = {
    "PCA+MLP": ("rank30_pca_mlp_energy", "reconstructed_mpds.npz"),
    "DeepONet": ("deeponet_energy", "predicted_mpds.npz"),
}
CATEGORY_LABELS = {
    "A": "DeepONet wins MPD and isotherm",
    "B": "DeepONet wins MPD; PCA+MLP wins isotherm",
    "C": "PCA+MLP wins MPD; DeepONet wins isotherm",
    "D": "PCA+MLP wins MPD and isotherm",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.\-\[\]]+", "_", value)


def energy_for(gas: str, mof_id: str, n_valid: int) -> np.ndarray:
    folder, filename, _ = canonical.ENERGY_FILES[gas]
    pipeline = ROOT / "MPD_isotherm_reweight_pipeline" / folder
    bulk = ROOT / "bulk_isotherms_from_MPD" / folder
    roots = [pipeline] if gas == "CO2" else [bulk, pipeline]
    for root in roots:
        path = root / mof_id / filename
        if path.exists():
            value = canonical.load_energy(path)
            if len(value) >= n_valid:
                return value[:n_valid]
    raise ValueError(f"{gas}/{mof_id}: no compatible energy vector")


def category(row: pd.Series) -> str:
    deep_tv = row.deep_tv <= row.pca_tv
    deep_iso = row.deep_rmsre <= row.pca_rmsre
    return "A" if deep_tv and deep_iso else "B" if deep_tv else "C" if deep_iso else "D"


def choose_cases(group: pd.DataFrame) -> pd.DataFrame:
    """Choose strong, typical, and near-boundary cases without duplicates."""
    chosen = []
    for label, order in (
        ("strong", group.advantage_magnitude.sort_values(ascending=False).index),
        ("typical", (group.advantage_magnitude - group.advantage_magnitude.median()).abs().sort_values().index),
        ("near_tie", group.advantage_magnitude.sort_values().index),
    ):
        idx = next((x for x in order if x not in {v[1] for v in chosen}), None)
        if idx is not None:
            chosen.append((label, idx))
    rows = []
    for label, idx in chosen:
        row = group.loc[idx].copy()
        row["case_type"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def export_case(gas: str, row: pd.Series, payload: dict) -> None:
    mof_id, i = str(row.mof_id), int(row.row_index)
    true = payload["truth"][i]
    mask = payload["masks"][i]
    n = np.where(mask)[0]
    true_lnpi = canonical.normalize(true[mask])
    pca_lnpi = canonical.normalize(payload["PCA+MLP"][i, mask])
    deep_lnpi = canonical.normalize(payload["DeepONet"][i, mask])
    true_p, pca_p, deep_p = np.exp(true_lnpi), np.exp(pca_lnpi), np.exp(deep_lnpi)
    energy = energy_for(gas, mof_id, len(n))
    true_iso = canonical.reweight(true_lnpi, energy)
    pca_iso = canonical.reweight(pca_lnpi, energy)
    deep_iso = canonical.reweight(deep_lnpi, energy)
    floor = max(1e-8, 1e-6 * float(true_iso.max()))
    pca_rel = (pca_iso - true_iso) / np.maximum(true_iso, floor)
    deep_rel = (deep_iso - true_iso) / np.maximum(true_iso, floor)

    folder = OUT / "representative_cases" / gas / f"category_{row.category}" / safe_name(mof_id)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "N": n, "true_lnPi": true_lnpi, "PCA_MLP_lnPi": pca_lnpi,
        "DeepONet_lnPi": deep_lnpi, "true_probability": true_p,
        "PCA_MLP_probability": pca_p, "DeepONet_probability": deep_p,
        "PCA_MLP_TV_contribution": 0.5 * np.abs(pca_p - true_p),
        "DeepONet_TV_contribution": 0.5 * np.abs(deep_p - true_p),
    }).to_csv(folder / "mpd.csv", index=False)
    pd.DataFrame({
        "pressure_bar": canonical.PRESSURES, "true_mean_N_supercell": true_iso,
        "PCA_MLP_mean_N_supercell": pca_iso, "DeepONet_mean_N_supercell": deep_iso,
        "PCA_MLP_relative_error": pca_rel, "DeepONet_relative_error": deep_rel,
    }).to_csv(folder / "isotherm_273K.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(n, true_p, color=COLORS["true"], lw=2.2, label="True")
    ax.plot(n, pca_p, color=COLORS["PCA+MLP"], lw=1.8, label="PCA+MLP")
    ax.plot(n, deep_p, color=COLORS["DeepONet"], lw=1.8, label="DeepONet")
    ax.set(xlabel="N (molecules/supercell)", ylabel="Probability", title="Reference-condition MPD")
    ax = axes[0, 1]
    ax.plot(n, 0.5*np.abs(pca_p-true_p), color=COLORS["PCA+MLP"], lw=1.8, label="PCA+MLP")
    ax.plot(n, 0.5*np.abs(deep_p-true_p), color=COLORS["DeepONet"], lw=1.8, label="DeepONet")
    ax.set(xlabel="N (molecules/supercell)", ylabel="TV contribution", title="Macrostate-resolved MPD error")
    ax = axes[1, 0]
    ax.semilogx(canonical.PRESSURES, true_iso, color=COLORS["true"], lw=2.2, label="True")
    ax.semilogx(canonical.PRESSURES, pca_iso, color=COLORS["PCA+MLP"], lw=1.8, label="PCA+MLP")
    ax.semilogx(canonical.PRESSURES, deep_iso, color=COLORS["DeepONet"], lw=1.8, label="DeepONet")
    ax.set(xlabel="Pressure (bar)", ylabel="Mean N/supercell", title="Reweighted isotherm at 273 K")
    ax = axes[1, 1]
    ax.semilogx(canonical.PRESSURES, pca_rel, color=COLORS["PCA+MLP"], lw=1.8, label="PCA+MLP")
    ax.semilogx(canonical.PRESSURES, deep_rel, color=COLORS["DeepONet"], lw=1.8, label="DeepONet")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(xlabel="Pressure (bar)", ylabel="Relative error", title="Pressure-resolved isotherm error")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{gas} | {mof_id}\nCategory {row.category}: {CATEGORY_LABELS[row.category]} | {row.case_type}\n"
        f"TV: PCA+MLP={row.pca_tv:.3f}, DeepONet={row.deep_tv:.3f}; "
        f"RMSRE: PCA+MLP={row.pca_rmsre:.3f}, DeepONet={row.deep_rmsre:.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)


def aggregate_plots(paired: pd.DataFrame) -> None:
    folder = OUT / "aggregate_plots"
    folder.mkdir(parents=True, exist_ok=True)
    for gas in GASES:
        d = paired.loc[paired.gas.eq(gas)]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        specs = [
            ("pca_tv", "deep_tv", "PCA+MLP TV", "DeepONet TV", "Paired MPD TV"),
            ("pca_rmsre", "deep_rmsre", "PCA+MLP RMSRE", "DeepONet RMSRE", "Paired 273 K RMSRE"),
            ("delta_tv", "delta_rmsre", "ΔTV (DeepONet − PCA+MLP)", "ΔRMSRE (DeepONet − PCA+MLP)", "Agreement of model advantage"),
        ]
        for ax, (x, y, xl, yl, title) in zip(axes, specs):
            for cat, group in d.groupby("category"):
                ax.scatter(group[x], group[y], s=25, alpha=0.7, label=f"{cat}: {CATEGORY_LABELS[cat]}")
            if x.startswith("pca"):
                low = min(d[x].min(), d[y].min()); high = max(d[x].max(), d[y].max())
                ax.plot([low, high], [low, high], "k--", lw=1)
            else:
                ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
            ax.set(xlabel=xl, ylabel=yl, title=title)
            ax.grid(alpha=0.2)
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
        fig.suptitle(f"{gas}: current-model paired test comparison (n={len(d)})")
        fig.tight_layout(rect=(0, 0.12, 1, 0.93))
        fig.savefig(folder / f"{gas}_paired_diagnostics.png", dpi=180)
        plt.close(fig)


def main() -> None:
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(STAGE / "evaluation/test_metrics_by_mof.csv", dtype={"mof_id": str})
    pca = metrics.loc[metrics.model.eq("PCA+MLP")].rename(columns={
        "tv_distance": "pca_tv", "isotherm_rmsre_273K": "pca_rmsre"
    })[["gas", "mof_id", "pca_tv", "pca_rmsre"]]
    deep = metrics.loc[metrics.model.eq("DeepONet")].rename(columns={
        "tv_distance": "deep_tv", "isotherm_rmsre_273K": "deep_rmsre"
    })[["gas", "mof_id", "deep_tv", "deep_rmsre"]]
    paired = pca.merge(deep, on=["gas", "mof_id"], validate="one_to_one")
    paired["delta_tv"] = paired.deep_tv - paired.pca_tv
    paired["delta_rmsre"] = paired.deep_rmsre - paired.pca_rmsre
    paired["category"] = paired.apply(category, axis=1)
    paired["category_description"] = paired.category.map(CATEGORY_LABELS)
    paired["advantage_magnitude"] = paired.delta_tv.abs() + paired.delta_rmsre.abs()

    cases = []
    payloads = {}
    for gas in GASES:
        data = STAGE / "data" / gas
        index = pd.read_csv(data / "mof_ids.csv", dtype={"mof_id": str})
        row_map = pd.Series(np.arange(len(index)), index=index.mof_id)
        paired.loc[paired.gas.eq(gas), "row_index"] = row_map.loc[
            paired.loc[paired.gas.eq(gas), "mof_id"]
        ].to_numpy()
        payload = {
            "truth": np.asarray(np.load(data / "mpd_values.npz")["lnpi"], float),
            "masks": np.asarray(np.load(data / "mpd_masks.npz")["valid_mask"], bool),
        }
        for model, (run, filename) in MODEL_FILES.items():
            payload[model] = np.asarray(np.load(STAGE / "model_runs" / run / gas / filename)["predicted_lnpi"], float)
        payloads[gas] = payload
        for cat in "ABCD":
            group = paired.loc[paired.gas.eq(gas) & paired.category.eq(cat)]
            if len(group):
                cases.append(choose_cases(group))

    paired["row_index"] = paired.row_index.astype(int)
    representatives = pd.concat(cases, ignore_index=True)
    paired.to_csv(OUT / "metrics/paired_test_comparison.csv", index=False)
    counts = paired.groupby(["gas", "category", "category_description"], sort=False).size().rename("count").reset_index()
    totals = paired.groupby("gas").size()
    counts["fraction"] = counts.apply(lambda r: r["count"] / totals[r["gas"]], axis=1)
    counts.to_csv(OUT / "metrics/category_counts.csv", index=False)
    representatives.to_csv(OUT / "metrics/representative_cases.csv", index=False)
    aggregate_plots(paired)
    for _, row in representatives.iterrows():
        export_case(str(row.gas), row, payloads[str(row.gas)])

    manifest = {
        "test_mofs_per_gas": 209,
        "categories": CATEGORY_LABELS,
        "representative_selection": "strong, typical, and near-tie within each nonempty gas/category",
        "temperature_K": 273.0,
        "pressure_points": len(canonical.PRESSURES),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(counts.to_string(index=False))
    print(f"representative_cases={len(representatives)}")


if __name__ == "__main__":
    main()
