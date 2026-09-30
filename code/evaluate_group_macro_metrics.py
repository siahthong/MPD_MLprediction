#!/usr/bin/env python3
"""Calculate test metrics with equal weight assigned to each parent group."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
PROJECT = STAGE.parent
OUTPUT = STAGE / "evaluation" / "group_macro_metrics"
GASES = ("CO2", "CH4", "N2")


def per_mof_standardized_coefficient_mse(gas: str) -> pd.DataFrame:
    run = STAGE / "model_runs" / "rank30_pca_mlp_energy" / gas
    predictions = pd.read_csv(
        run / "latent_coefficient_predictions.csv", dtype={"mof_id": str}
    )
    scaler = np.load(run / "coefficient_scaler.npz", allow_pickle=False)
    scale = np.asarray(scaler["scale"], dtype=float)
    true_columns = [f"true_LC{i}" for i in range(1, 31)]
    predicted_columns = [f"predicted_LC{i}" for i in range(1, 31)]
    if len(scale) != 30:
        raise ValueError(f"{gas}: coefficient scaler is not rank 30")
    true = predictions[true_columns].to_numpy(float)
    predicted = predictions[predicted_columns].to_numpy(float)
    predictions["standardized_coefficient_mse"] = np.mean(
        ((predicted - true) / scale[None, :]) ** 2, axis=1
    )
    return predictions[["mof_id", "split", "standardized_coefficient_mse"]]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    group_map = pd.read_csv(
        PROJECT / "02_groups_split" / "outputs" / "split_assignment.csv",
        dtype={"coreid": str, "group_id": str},
    ).rename(columns={"coreid": "mof_id", "split": "group_map_split"})
    if group_map.mof_id.duplicated().any():
        raise ValueError("Duplicate MOF IDs in split assignment")

    model_metrics = pd.read_csv(
        STAGE / "evaluation" / "test_metrics_by_mof.csv", dtype={"mof_id": str}
    )
    coefficient_mse = pd.concat(
        [
            per_mof_standardized_coefficient_mse(gas).assign(gas=gas)
            for gas in GASES
        ],
        ignore_index=True,
    )
    coefficient_mse = coefficient_mse.loc[
        coefficient_mse["split"].isin(["test"])
    ].drop(columns="split")

    per_mof = model_metrics.merge(
        group_map[["mof_id", "group_id", "group_map_split"]],
        on="mof_id",
        how="left",
        validate="many_to_one",
    )
    if per_mof[["group_id", "group_map_split"]].isna().any().any():
        raise ValueError("Some test MOFs are missing their parent group")
    if not per_mof.group_map_split.eq("test").all():
        raise ValueError("Metric and group-map split labels disagree")
    per_mof = per_mof.merge(
        coefficient_mse.rename(columns={"standardized_coefficient_mse": "pca_mlp_standardized_coefficient_mse"}),
        on=["gas", "mof_id"],
        how="left",
        validate="many_to_one",
    )
    per_mof.loc[
        ~per_mof.model.eq("PCA+MLP"), "pca_mlp_standardized_coefficient_mse"
    ] = np.nan
    per_mof["structures_in_test_group"] = per_mof.groupby(
        ["gas", "model", "group_id"]
    )["mof_id"].transform("size")

    group_rows = []
    for (gas, model, group_id), data in per_mof.groupby(
        ["gas", "model", "group_id"], sort=False
    ):
        row = {
            "gas": gas,
            "model": model,
            "group_id": group_id,
            "n_structures": len(data),
            "mean_tv": data.tv_distance.mean(),
            "sd_tv": data.tv_distance.std(ddof=1) if len(data) > 1 else np.nan,
            "mean_rmsre_273K": data.isotherm_rmsre_273K.mean(),
            "sd_rmsre_273K": data.isotherm_rmsre_273K.std(ddof=1) if len(data) > 1 else np.nan,
        }
        if model == "PCA+MLP":
            row["mean_standardized_coefficient_mse"] = (
                data.pca_mlp_standardized_coefficient_mse.mean()
            )
            row["sd_standardized_coefficient_mse"] = (
                data.pca_mlp_standardized_coefficient_mse.std(ddof=1)
                if len(data) > 1
                else np.nan
            )
        else:
            row["mean_standardized_coefficient_mse"] = np.nan
            row["sd_standardized_coefficient_mse"] = np.nan
        group_rows.append(row)
    per_group = pd.DataFrame(group_rows)

    summary_rows = []
    for (gas, model), data in per_group.groupby(["gas", "model"], sort=False):
        multi = data.loc[data.n_structures > 1]
        summary_rows.append(
            {
                "gas": gas,
                "model": model,
                "n_test_structures": int(data.n_structures.sum()),
                "n_test_groups": len(data),
                "n_singleton_groups": int(data.n_structures.eq(1).sum()),
                "n_multi_structure_groups": int(data.n_structures.gt(1).sum()),
                "largest_test_group": int(data.n_structures.max()),
                "group_macro_mean_standardized_coefficient_mse": (
                    data.mean_standardized_coefficient_mse.mean()
                    if model == "PCA+MLP"
                    else np.nan
                ),
                "group_median_standardized_coefficient_mse": (
                    data.mean_standardized_coefficient_mse.median()
                    if model == "PCA+MLP"
                    else np.nan
                ),
                "group_macro_mean_tv": data.mean_tv.mean(),
                "group_median_tv": data.mean_tv.median(),
                "group_macro_mean_rmsre_273K": data.mean_rmsre_273K.mean(),
                "group_median_rmsre_273K": data.mean_rmsre_273K.median(),
                "mean_within_group_sd_tv_multi_groups": multi.sd_tv.mean(),
                "mean_within_group_sd_rmsre_multi_groups": multi.sd_rmsre_273K.mean(),
                "mean_within_group_sd_mse_multi_groups": (
                    multi.sd_standardized_coefficient_mse.mean()
                    if model == "PCA+MLP"
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    comparison_rows = []
    for (gas, model), data in per_mof.groupby(["gas", "model"], sort=False):
        macro = summary.loc[summary.gas.eq(gas) & summary.model.eq(model)].iloc[0]
        comparison_rows.append(
            {
                "gas": gas,
                "model": model,
                "test_structures": len(data),
                "test_groups": int(macro.n_test_groups),
                "structure_weighted_mean_standardized_coefficient_mse": (
                    data.pca_mlp_standardized_coefficient_mse.mean()
                    if model == "PCA+MLP"
                    else np.nan
                ),
                "group_macro_mean_standardized_coefficient_mse": macro.group_macro_mean_standardized_coefficient_mse,
                "structure_weighted_mean_tv": data.tv_distance.mean(),
                "group_macro_mean_tv": macro.group_macro_mean_tv,
                "structure_weighted_mean_rmsre_273K": data.isotherm_rmsre_273K.mean(),
                "group_macro_mean_rmsre_273K": macro.group_macro_mean_rmsre_273K,
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    presentation_rows = []
    for (gas, model), data in per_mof.groupby(["gas", "model"], sort=False):
        groups = per_group.loc[per_group.gas.eq(gas) & per_group.model.eq(model)]

        def mean_median(values: pd.Series) -> str:
            return f"{values.mean():.4f} / {values.median():.4f}"

        presentation_rows.append(
            {
                "Gas": gas,
                "Model": model,
                "Coefficient MSE S": (
                    mean_median(data.pca_mlp_standardized_coefficient_mse)
                    if model == "PCA+MLP" else "N/A"
                ),
                "Coefficient MSE G": (
                    mean_median(groups.mean_standardized_coefficient_mse)
                    if model == "PCA+MLP" else "N/A"
                ),
                "MPD TV S": mean_median(data.tv_distance),
                "MPD TV G": mean_median(groups.mean_tv),
                "RMSRE S": mean_median(data.isotherm_rmsre_273K),
                "RMSRE G": mean_median(groups.mean_rmsre_273K),
            }
        )
    presentation = pd.DataFrame(presentation_rows)

    # Reconciliation: weighting group means by group size must recover MOF means.
    checks = []
    for (gas, model), groups in per_group.groupby(["gas", "model"], sort=False):
        source = per_mof.loc[per_mof.gas.eq(gas) & per_mof.model.eq(model)]
        weighted_tv = np.average(groups.mean_tv, weights=groups.n_structures)
        weighted_rmsre = np.average(groups.mean_rmsre_273K, weights=groups.n_structures)
        checks.append(
            {
                "gas": gas,
                "model": model,
                "per_mof_mean_tv": source.tv_distance.mean(),
                "group_size_weighted_mean_tv": weighted_tv,
                "tv_absolute_difference": abs(source.tv_distance.mean() - weighted_tv),
                "per_mof_mean_rmsre_273K": source.isotherm_rmsre_273K.mean(),
                "group_size_weighted_mean_rmsre_273K": weighted_rmsre,
                "rmsre_absolute_difference": abs(source.isotherm_rmsre_273K.mean() - weighted_rmsre),
            }
        )
    checks = pd.DataFrame(checks)
    if checks[["tv_absolute_difference", "rmsre_absolute_difference"]].to_numpy().max() > 1e-12:
        raise RuntimeError("Group aggregation failed reconciliation")

    per_mof.to_csv(OUTPUT / "per_mof_test_metrics_with_group.csv", index=False)
    per_group.to_csv(OUTPUT / "per_group_test_metrics.csv", index=False)
    summary.to_csv(OUTPUT / "group_macro_test_summary.csv", index=False)
    comparison.to_csv(OUTPUT / "structure_vs_group_macro_summary.csv", index=False)
    presentation.to_csv(OUTPUT / "mean_median_presentation_table.csv", index=False)
    checks.to_csv(OUTPUT / "aggregation_reconciliation.csv", index=False)
    print(summary.to_string(index=False))
    print("aggregation_reconciliation=PASS")


if __name__ == "__main__":
    main()


