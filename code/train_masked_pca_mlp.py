#!/usr/bin/env python3
"""Train an MLP to predict masked low-rank MPD coefficients from MOF descriptors."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def stable_logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def normalize_valid(values: np.ndarray) -> np.ndarray:
    return values - stable_logsumexp(values)


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    squared_error = 0.0
    n_values = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            squared_error += float(torch.sum((predictions - targets) ** 2).item())
            n_values += targets.numel()
    return squared_error / n_values


def predict(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(features, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            predictions.append(model(batch.to(device)).cpu().numpy())
    return np.vstack(predictions)


def reconstruct_and_evaluate(
    predicted_coefficients: np.ndarray,
    mean_curve: np.ndarray,
    basis: np.ndarray,
    true_values: np.ndarray,
    masks: np.ndarray,
    index: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    raw_reconstruction = mean_curve[None, :] + predicted_coefficients @ basis.T
    reconstructed = np.full_like(true_values, np.nan, dtype=float)
    rows = []

    for row_index in range(len(index)):
        valid = masks[row_index]
        predicted_lnpi = normalize_valid(raw_reconstruction[row_index, valid])
        true_lnpi = true_values[row_index, valid]
        reconstructed[row_index, valid] = predicted_lnpi

        predicted_probability = np.exp(predicted_lnpi)
        true_probability = np.exp(true_lnpi)
        probability_error = np.abs(predicted_probability - true_probability)
        rows.append(
            {
                "row_index": row_index,
                "mof_id": str(index.iloc[row_index]["mof_id"]),
                "split": str(index.iloc[row_index]["split"]),
                "Nmax": int(index.iloc[row_index]["Nmax"]),
                "n_valid_states": int(valid.sum()),
                "lnpi_mae": float(np.mean(np.abs(predicted_lnpi - true_lnpi))),
                "lnpi_rmse": float(
                    np.sqrt(np.mean((predicted_lnpi - true_lnpi) ** 2))
                ),
                "probability_mae": float(np.mean(probability_error)),
                "probability_rmse": float(np.sqrt(np.mean(probability_error**2))),
                "total_variation_distance": float(0.5 * np.sum(probability_error)),
                "normalization_error": float(abs(stable_logsumexp(predicted_lnpi))),
            }
        )

    return reconstructed, pd.DataFrame(rows)


def summarize_reconstruction(metrics: pd.DataFrame) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    metric_names = (
        "lnpi_mae",
        "lnpi_rmse",
        "probability_mae",
        "probability_rmse",
        "total_variation_distance",
    )
    for split_name, group in metrics.groupby("split", sort=False):
        split_summary: Dict[str, float | int] = {"n_mofs": int(len(group))}
        for metric in metric_names:
            split_summary[f"mean_{metric}"] = float(group[metric].mean())
            split_summary[f"median_{metric}"] = float(group[metric].median())
        result[str(split_name)] = split_summary
    return result


def save_scaler(path: Path, scaler: StandardScaler, names: Sequence[str]) -> None:
    np.savez_compressed(
        path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        names=np.asarray(names),
    )


def json_ready_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLP to predict rank-20 masked low-rank coefficients."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--low-rank-model",
        type=Path,
        default=HERE
        / "outputs/selected_rank20/rank_20/masked_low_rank_model.npz",
    )
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=HERE / "outputs/selected_rank20/rank_20/latent_coefficients.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "outputs/run_masked_pca_mlp_rank20",
    )
    parser.add_argument("--hidden", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    index = pd.read_csv(args.data / "mof_ids.csv")
    index["mof_id"] = index["mof_id"].astype(str)
    index["split"] = index["split"].replace({"val": "validation"})
    if index["mof_id"].duplicated().any():
        raise ValueError("mof_ids.csv contains duplicate MOF IDs.")

    feature_archive = np.load(args.data / "feature_values.npz", allow_pickle=False)
    features = np.asarray(feature_archive["features"], dtype=float)
    feature_names = [str(name) for name in feature_archive["feature_names"]]
    true_values = np.asarray(
        np.load(args.data / "mpd_values.npz", allow_pickle=False)["lnpi"],
        dtype=float,
    )
    masks = np.asarray(
        np.load(args.data / "mpd_masks.npz", allow_pickle=False)["valid_mask"],
        dtype=bool,
    )

    coefficient_table = pd.read_csv(args.coefficients)
    coefficient_table["mof_id"] = coefficient_table["mof_id"].astype(str)
    coefficient_columns = [
        column for column in coefficient_table.columns if column.startswith("LC")
    ]
    coefficient_columns.sort(key=lambda name: int(name[2:]))
    if not coefficient_columns:
        raise ValueError(f"No LC coefficient columns found in {args.coefficients}.")
    if coefficient_table["mof_id"].duplicated().any():
        raise ValueError("Latent coefficient table contains duplicate MOF IDs.")

    coefficient_table = coefficient_table.set_index("mof_id")
    missing_coefficients = sorted(set(index["mof_id"]) - set(coefficient_table.index))
    extra_coefficients = sorted(set(coefficient_table.index) - set(index["mof_id"]))
    if missing_coefficients or extra_coefficients:
        raise ValueError(
            "Coefficient and dataset MOF IDs differ. "
            f"Missing: {missing_coefficients[:5]}; extra: {extra_coefficients[:5]}"
        )
    aligned_coefficients = coefficient_table.loc[
        index["mof_id"], coefficient_columns
    ].to_numpy(dtype=float)
    aligned_coefficient_splits = coefficient_table.loc[
        index["mof_id"], "split"
    ].astype(str).replace({"val": "validation"}).to_numpy()
    if not np.array_equal(aligned_coefficient_splits, index["split"].astype(str)):
        raise ValueError("Coefficient split labels do not match mof_ids.csv.")

    model_archive = np.load(args.low_rank_model, allow_pickle=False)
    mean_curve = np.asarray(model_archive["mean_curve"], dtype=float)
    basis = np.asarray(model_archive["basis"], dtype=float)
    if basis.shape != (true_values.shape[1], len(coefficient_columns)):
        raise ValueError(
            f"Basis shape {basis.shape} is incompatible with MPDs "
            f"{true_values.shape} and {len(coefficient_columns)} coefficients."
        )
    if features.shape[0] != len(index) or true_values.shape != masks.shape:
        raise ValueError("Dataset array shapes are inconsistent with mof_ids.csv.")

    split_indices = {
        split: np.where(index["split"].eq(split).to_numpy())[0]
        for split in ("train", "validation", "test")
    }
    if any(len(rows) == 0 for rows in split_indices.values()):
        raise ValueError("Training, validation, and test splits must all be non-empty.")

    train_indices = split_indices["train"]
    validation_indices = split_indices["validation"]
    test_indices = split_indices["test"]

    feature_scaler = StandardScaler()
    coefficient_scaler = StandardScaler()
    scaled_features = np.empty_like(features, dtype=np.float32)
    scaled_coefficients = np.empty_like(aligned_coefficients, dtype=np.float32)
    feature_scaler.fit(features[train_indices])
    coefficient_scaler.fit(aligned_coefficients[train_indices])
    scaled_features[:] = feature_scaler.transform(features)
    scaled_coefficients[:] = coefficient_scaler.transform(aligned_coefficients)

    train_loader = make_loader(
        scaled_features[train_indices],
        scaled_coefficients[train_indices],
        args.batch_size,
        shuffle=True,
    )
    validation_loader = make_loader(
        scaled_features[validation_indices],
        scaled_coefficients[validation_indices],
        args.batch_size,
        shuffle=False,
    )
    test_loader = make_loader(
        scaled_features[test_indices],
        scaled_coefficients[test_indices],
        args.batch_size,
        shuffle=False,
    )

    device = torch.device(args.device)
    model = MLP(
        input_dim=features.shape[1],
        output_dim=aligned_coefficients.shape[1],
        hidden_dims=args.hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        squared_error = 0.0
        n_values = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)
            loss = nn.functional.mse_loss(predictions, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            squared_error += float(
                torch.sum((predictions.detach() - batch_targets) ** 2).item()
            )
            n_values += batch_targets.numel()

        train_loss = squared_error / n_values
        validation_loss = evaluate_loss(model, validation_loader, device)
        elapsed = time.perf_counter() - start_time

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.output / "best_model.pt")
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss_standardized_coefficient_mse": train_loss,
                "validation_loss_standardized_coefficient_mse": validation_loss,
                "elapsed_seconds": elapsed,
            }
        )
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:5d} train_mse={train_loss:.6f} "
                f"val_mse={validation_loss:.6f} elapsed_s={elapsed:.1f}"
            )
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; best epoch was {best_epoch}."
            )
            break

    model.load_state_dict(
        torch.load(
            args.output / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
    )
    model.to(device)
    test_loss = evaluate_loss(model, test_loader, device)

    predicted_scaled_coefficients = predict(
        model,
        scaled_features,
        device,
        args.batch_size,
    )
    predicted_coefficients = coefficient_scaler.inverse_transform(
        predicted_scaled_coefficients
    )
    reconstructed, reconstruction_metrics = reconstruct_and_evaluate(
        predicted_coefficients,
        mean_curve,
        basis,
        true_values,
        masks,
        index,
    )

    coefficient_output = index[["mof_id", "split"]].copy()
    for column_index, column in enumerate(coefficient_columns):
        coefficient_output[f"true_{column}"] = aligned_coefficients[:, column_index]
        coefficient_output[f"predicted_{column}"] = predicted_coefficients[
            :, column_index
        ]

    pd.DataFrame(history).to_csv(args.output / "training_log.csv", index=False)
    coefficient_output.to_csv(
        args.output / "latent_coefficient_predictions.csv",
        index=False,
    )
    reconstruction_metrics.to_csv(
        args.output / "reconstruction_metrics_by_mof.csv",
        index=False,
    )
    np.savez_compressed(
        args.output / "reconstructed_mpds.npz",
        predicted_lnpi=reconstructed,
        valid_mask=masks,
        n_axis=np.arange(true_values.shape[1], dtype=np.int16),
        mof_ids=index["mof_id"].to_numpy(dtype=str),
        splits=index["split"].to_numpy(dtype=str),
    )
    save_scaler(
        args.output / "feature_scaler.npz",
        feature_scaler,
        feature_names,
    )
    save_scaler(
        args.output / "coefficient_scaler.npz",
        coefficient_scaler,
        coefficient_columns,
    )

    completed_epochs = int(history[-1]["epoch"])
    total_time = time.perf_counter() - start_time
    reconstruction_summary = summarize_reconstruction(reconstruction_metrics)
    coefficient_mean_baseline_mse = {
        split: float(np.mean(scaled_coefficients[rows] ** 2))
        for split, rows in split_indices.items()
    }
    metrics = {
        "training_loss": "mse_standardized_masked_low_rank_coefficients",
        "rank": int(basis.shape[1]),
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "stopped_early": completed_epochs < args.epochs,
        "best_validation_loss_standardized_coefficient_mse": best_validation_loss,
        "test_loss_standardized_coefficient_mse": test_loss,
        "coefficient_mean_baseline_mse": coefficient_mean_baseline_mse,
        "test_coefficient_mse_skill_score_vs_mean": float(
            1.0 - test_loss / coefficient_mean_baseline_mse["test"]
        ),
        "training_time_seconds": total_time,
        "n_train_mofs": int(len(train_indices)),
        "n_validation_mofs": int(len(validation_indices)),
        "n_test_mofs": int(len(test_indices)),
        "maximum_reconstruction_normalization_error": float(
            reconstruction_metrics["normalization_error"].max()
        ),
        "reconstruction_metrics": reconstruction_summary,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "config.json").write_text(
        json.dumps(
            json_ready_args(args)
            | {
                "feature_names": feature_names,
                "coefficient_names": coefficient_columns,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Test standardized coefficient MSE: {test_loss:.6f}")
    print(
        "Test mean MPD total-variation distance: "
        f"{reconstruction_summary['test']['mean_total_variation_distance']:.6f}"
    )
    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    main()




