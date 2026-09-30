#!/usr/bin/env python3
"""Fit smooth weighted low-rank MPD bases using observed training entries only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def stable_logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def normalize_valid(values: np.ndarray) -> np.ndarray:
    return values - stable_logsumexp(values)


def second_difference_penalty(n_columns: int) -> np.ndarray:
    if n_columns < 3:
        return np.zeros((n_columns, n_columns), dtype=float)
    difference = np.zeros((n_columns - 2, n_columns), dtype=float)
    for row in range(n_columns - 2):
        difference[row, row : row + 3] = [1.0, -2.0, 1.0]
    return difference.T @ difference


def smooth_weighted_curve(
    numerator: np.ndarray,
    denominator: np.ndarray,
    penalty: np.ndarray,
    smoothness: float,
    ridge: float,
) -> np.ndarray:
    system = np.diag(denominator + ridge) + smoothness * penalty
    return np.linalg.solve(system, numerator)


def observed_rmse(
    values_filled: np.ndarray,
    mask: np.ndarray,
    mean_curve: np.ndarray,
    scores: np.ndarray,
    basis: np.ndarray,
) -> float:
    prediction = mean_curve[None, :] + scores @ basis.T
    residual = (prediction - values_filled)[mask]
    return float(np.sqrt(np.mean(residual**2)))


def initialize_factorization(
    centered_values: np.ndarray,
    mask: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    filled = np.where(mask, centered_values, 0.0)
    u, singular_values, vt = np.linalg.svd(filled, full_matrices=False)
    scores = u[:, :rank] * singular_values[:rank]
    basis = vt[:rank, :].T
    return scores, basis


def update_scores(
    centered_values: np.ndarray,
    mask: np.ndarray,
    basis: np.ndarray,
    ridge: float,
) -> np.ndarray:
    n_rows = centered_values.shape[0]
    rank = basis.shape[1]
    scores = np.zeros((n_rows, rank), dtype=float)
    identity = np.eye(rank)
    for row in range(n_rows):
        valid = mask[row]
        design = basis[valid]
        target = centered_values[row, valid]
        scores[row] = np.linalg.solve(
            design.T @ design + ridge * identity,
            design.T @ target,
        )
    return scores


def update_basis(
    centered_values: np.ndarray,
    mask: np.ndarray,
    scores: np.ndarray,
    basis: np.ndarray,
    penalty: np.ndarray,
    smoothness: float,
    ridge: float,
) -> np.ndarray:
    prediction = scores @ basis.T
    for component in range(basis.shape[1]):
        component_scores = scores[:, component]
        residual = centered_values - prediction + np.outer(
            component_scores, basis[:, component]
        )
        denominator = np.sum(mask * component_scores[:, None] ** 2, axis=0)
        numerator = np.sum(
            mask * component_scores[:, None] * residual,
            axis=0,
        )
        updated = smooth_weighted_curve(
            numerator,
            denominator,
            penalty,
            smoothness=smoothness,
            ridge=ridge,
        )
        prediction += np.outer(component_scores, updated - basis[:, component])
        basis[:, component] = updated
    return basis


def orthogonalize(scores: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(basis)
    return scores @ r.T, q


def fit_weighted_factorization(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    rank: int,
    smoothness: float,
    ridge: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    n_columns = train_values.shape[1]
    penalty = second_difference_penalty(n_columns)
    values_filled = np.where(train_mask, train_values, 0.0)

    counts = train_mask.sum(axis=0).astype(float)
    sums = np.sum(values_filled, axis=0)
    mean_curve = smooth_weighted_curve(
        sums,
        counts,
        penalty,
        smoothness=smoothness,
        ridge=ridge,
    )
    centered = values_filled - mean_curve[None, :]
    scores, basis = initialize_factorization(centered, train_mask, rank)
    scores = update_scores(centered, train_mask, basis, ridge)

    history = []
    previous_rmse = float("inf")
    for iteration in range(1, max_iterations + 1):
        basis = update_basis(
            centered,
            train_mask,
            scores,
            basis,
            penalty,
            smoothness=smoothness,
            ridge=ridge,
        )
        scores = update_scores(centered, train_mask, basis, ridge)
        scores, basis = orthogonalize(scores, basis)
        rmse = observed_rmse(values_filled, train_mask, mean_curve, scores, basis)
        relative_change = (
            abs(previous_rmse - rmse) / max(previous_rmse, 1e-12)
            if np.isfinite(previous_rmse)
            else np.nan
        )
        history.append(
            {
                "iteration": iteration,
                "observed_train_rmse": rmse,
                "relative_rmse_change": relative_change,
            }
        )
        if np.isfinite(relative_change) and relative_change < tolerance:
            break
        previous_rmse = rmse

    component_variance = np.var(scores, axis=0)
    order = np.argsort(component_variance)[::-1]
    scores = scores[:, order]
    basis = basis[:, order]
    return mean_curve, scores, basis, pd.DataFrame(history)


def infer_scores(
    values: np.ndarray,
    masks: np.ndarray,
    mean_curve: np.ndarray,
    basis: np.ndarray,
    ridge: float,
) -> np.ndarray:
    centered = np.where(masks, values - mean_curve[None, :], 0.0)
    return update_scores(centered, masks, basis, ridge)


def evaluate_split(
    split_name: str,
    row_indices: np.ndarray,
    values: np.ndarray,
    masks: np.ndarray,
    mean_curve: np.ndarray,
    basis: np.ndarray,
    ridge: float,
    train_column_counts: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    split_values = values[row_indices]
    split_masks = masks[row_indices]
    scores = infer_scores(split_values, split_masks, mean_curve, basis, ridge)
    reconstruction = mean_curve[None, :] + scores @ basis.T

    rows = []
    unsupported_columns = train_column_counts == 0
    for local_row, global_row in enumerate(row_indices):
        valid = split_masks[local_row]
        true_lnpi = split_values[local_row, valid]
        predicted_lnpi = normalize_valid(reconstruction[local_row, valid])
        true_probability = np.exp(true_lnpi)
        predicted_probability = np.exp(predicted_lnpi)
        absolute_probability_error = np.abs(predicted_probability - true_probability)
        valid_columns = np.where(valid)[0]
        unsupported_valid = unsupported_columns[valid]

        rows.append(
            {
                "split": split_name,
                "row_index": int(global_row),
                "n_valid_states": int(valid.sum()),
                "n_states_without_train_observations": int(unsupported_valid.sum()),
                "lnpi_mae": float(np.mean(np.abs(predicted_lnpi - true_lnpi))),
                "lnpi_rmse": float(np.sqrt(np.mean((predicted_lnpi - true_lnpi) ** 2))),
                "probability_mae": float(np.mean(absolute_probability_error)),
                "probability_rmse": float(
                    np.sqrt(np.mean((predicted_probability - true_probability) ** 2))
                ),
                "total_variation_distance": float(0.5 * np.sum(absolute_probability_error)),
                "unsupported_tail_lnpi_mae": (
                    float(
                        np.mean(
                            np.abs(predicted_lnpi[unsupported_valid] - true_lnpi[unsupported_valid])
                        )
                    )
                    if unsupported_valid.any()
                    else np.nan
                ),
                "maximum_valid_N": int(valid_columns[-1]),
            }
        )
    return pd.DataFrame(rows), scores


def summarize_metrics(rank: int, metrics: pd.DataFrame, history: pd.DataFrame) -> List[dict]:
    final_relative_change = float(history.iloc[-1]["relative_rmse_change"])
    rows = []
    for split_name, group in metrics.groupby("split", sort=False):
        row = {
            "rank": rank,
            "split": split_name,
            "n_mofs": group["row_index"].nunique(),
            "iterations": int(history["iteration"].max()),
            "final_observed_train_rmse": float(history.iloc[-1]["observed_train_rmse"]),
            "final_relative_rmse_change": final_relative_change,
        }
        for metric in (
            "lnpi_mae",
            "lnpi_rmse",
            "probability_mae",
            "probability_rmse",
            "total_variation_distance",
            "unsupported_tail_lnpi_mae",
        ):
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"median_{metric}"] = float(group[metric].median())
        rows.append(row)
    return rows


def parse_ranks(values: Iterable[int]) -> List[int]:
    ranks = sorted(set(int(value) for value in values))
    if not ranks or ranks[0] < 1:
        raise ValueError("All ranks must be positive integers.")
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit smooth weighted low-rank bases to masked absolute-N MPDs."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "1.data")
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15],
    )
    parser.add_argument("--smoothness", type=float, default=10.0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=HERE / "outputs/masked_low_rank")
    args = parser.parse_args()

    ranks = parse_ranks(args.ranks)
    values = np.asarray(np.load(args.data / "mpd_values.npz")["lnpi"], dtype=float)
    masks = np.asarray(np.load(args.data / "mpd_masks.npz")["valid_mask"], dtype=bool)
    index = pd.read_csv(args.data / "mof_ids.csv")
    mof_ids = index["mof_id"].astype(str).to_numpy()

    split_indices = {
        "train": np.where(index["split"].eq("train").to_numpy())[0],
        "val": np.where(index["split"].eq("val").to_numpy())[0],
        "test": np.where(index["split"].eq("test").to_numpy())[0],
    }
    train_indices = split_indices["train"]
    train_values = values[train_indices]
    train_masks = masks[train_indices]
    train_column_counts = train_masks.sum(axis=0)

    args.output.mkdir(parents=True, exist_ok=True)
    all_summary = []
    rank_selection_rows = []
    start_time = time.perf_counter()

    for rank in ranks:
        rank_dir = args.output / f"rank_{rank:02d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        mean_curve, train_scores, basis, history = fit_weighted_factorization(
            train_values,
            train_masks,
            rank=rank,
            smoothness=args.smoothness,
            ridge=args.ridge,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
        )

        metric_tables = []
        score_tables = []
        for split_name, rows in split_indices.items():
            metrics, scores = evaluate_split(
                split_name,
                rows,
                values,
                masks,
                mean_curve,
                basis,
                args.ridge,
                train_column_counts,
            )
            metrics.insert(2, "mof_id", mof_ids[metrics["row_index"].to_numpy(dtype=int)])
            metric_tables.append(metrics)
            score_df = pd.DataFrame(scores, columns=[f"LC{i + 1}" for i in range(rank)])
            score_df.insert(0, "mof_id", mof_ids[rows])
            score_df.insert(0, "split", split_name)
            score_tables.append(score_df)

        metrics_all = pd.concat(metric_tables, ignore_index=True)
        scores_all = pd.concat(score_tables, ignore_index=True)
        summary_rows = summarize_metrics(rank, metrics_all, history)
        all_summary.extend(summary_rows)
        validation_summary = next(row for row in summary_rows if row["split"] == "val")
        final_relative_change = float(history.iloc[-1]["relative_rmse_change"])
        converged = bool(
            np.isfinite(final_relative_change)
            and final_relative_change < args.tolerance
        )
        rank_selection_rows.append(
            {
                "rank": rank,
                "iterations": int(history.iloc[-1]["iteration"]),
                "converged": converged,
                "final_relative_rmse_change": final_relative_change,
                "validation_mean_total_variation_distance": validation_summary[
                    "mean_total_variation_distance"
                ],
                "validation_median_total_variation_distance": validation_summary[
                    "median_total_variation_distance"
                ],
                "validation_mean_lnpi_rmse": validation_summary["mean_lnpi_rmse"],
                "validation_median_lnpi_rmse": validation_summary["median_lnpi_rmse"],
            }
        )

        np.savez_compressed(
            rank_dir / "masked_low_rank_model.npz",
            mean_curve=mean_curve,
            basis=basis,
            train_column_observation_counts=train_column_counts,
            n_axis=np.arange(values.shape[1], dtype=np.int16),
        )
        history.to_csv(rank_dir / "fit_history.csv", index=False)
        metrics_all.to_csv(rank_dir / "reconstruction_metrics_by_mof.csv", index=False)
        scores_all.to_csv(rank_dir / "latent_coefficients.csv", index=False)
        print(
            f"rank={rank:2d} iterations={len(history):2d} "
            f"converged={converged} "
            f"train_rmse={history.iloc[-1]['observed_train_rmse']:.6f} "
            f"val_TV={validation_summary['mean_total_variation_distance']:.6f}"
        )

    summary = pd.DataFrame(all_summary)
    selection = pd.DataFrame(rank_selection_rows).sort_values("rank")
    selected_rank = int(
        selection.loc[
            selection["validation_mean_total_variation_distance"].idxmin(),
            "rank",
        ]
    )
    summary.to_csv(args.output / "reconstruction_vs_rank.csv", index=False)
    selection.to_csv(args.output / "rank_selection.csv", index=False)
    (args.output / "run_summary.json").write_text(
        json.dumps(
            {
                "method": "smooth_weighted_low_rank_factorization",
                "objective": "observed-entry squared error plus second-difference basis penalty",
                "training_only_fit": True,
                "ranks": ranks,
                "selected_rank_by_validation_mean_total_variation_distance": selected_rank,
                "smoothness": args.smoothness,
                "ridge": args.ridge,
                "max_iterations": args.max_iterations,
                "tolerance": args.tolerance,
                "n_train_mofs": len(train_indices),
                "train_column_observation_counts": train_column_counts.tolist(),
                "columns_without_train_observations": np.where(train_column_counts == 0)[0].tolist(),
                "runtime_seconds": time.perf_counter() - start_time,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Selected rank: {selected_rank}")
    print(f"Saved weighted low-rank study to {args.output}")


if __name__ == "__main__":
    main()

