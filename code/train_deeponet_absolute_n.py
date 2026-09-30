#!/usr/bin/env python3
"""Train a mask-aware DeepONet on absolute integer-N MPDs."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


class DeepONet(nn.Module):
    def __init__(self, feature_dim: int, p: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p),
        )

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        branch_output = self.branch(features)
        trunk_output = self.trunk(coordinates)
        return branch_output @ trunk_output.T


def masked_log_softmax(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~masks, -torch.inf)
    return F.log_softmax(masked_logits, dim=1)


def per_mof_masked_mae(
    predicted_lnpi: torch.Tensor,
    true_lnpi: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    safe_prediction = torch.where(masks, predicted_lnpi, true_lnpi)
    absolute_error = torch.abs(safe_prediction - true_lnpi)
    return absolute_error.sum(dim=1) / masks.sum(dim=1)


def make_loader(
    features: np.ndarray,
    values: np.ndarray,
    masks: np.ndarray,
    row_indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(features[row_indices], dtype=torch.float32),
        torch.tensor(values[row_indices], dtype=torch.float32),
        torch.tensor(masks[row_indices], dtype=torch.bool),
        torch.tensor(row_indices, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    coordinates: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    n_mofs = 0
    for features, values, masks, _ in loader:
        features = features.to(device)
        values = values.to(device)
        masks = masks.to(device)
        prediction = masked_log_softmax(model(features, coordinates), masks)
        losses = per_mof_masked_mae(prediction, values, masks)
        loss_sum += float(losses.sum().item())
        n_mofs += len(features)
    return loss_sum / n_mofs


@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    coordinates: torch.Tensor,
    device: torch.device,
    output_shape: tuple[int, int],
) -> np.ndarray:
    predictions = np.full(output_shape, np.nan, dtype=np.float64)
    model.eval()
    for features, _, masks, row_indices in loader:
        features = features.to(device)
        masks_device = masks.to(device)
        batch_prediction = masked_log_softmax(
            model(features, coordinates),
            masks_device,
        ).cpu().numpy()
        batch_masks = masks.numpy()
        for local_row, global_row in enumerate(row_indices.numpy()):
            predictions[global_row, batch_masks[local_row]] = batch_prediction[
                local_row, batch_masks[local_row]
            ]
    return predictions


def reconstruction_metrics(
    predicted: np.ndarray,
    true_values: np.ndarray,
    masks: np.ndarray,
    index: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row_index in range(len(index)):
        valid = masks[row_index]
        predicted_lnpi = predicted[row_index, valid]
        true_lnpi = true_values[row_index, valid]
        predicted_probability = np.exp(predicted_lnpi)
        true_probability = np.exp(true_lnpi)
        probability_error = np.abs(predicted_probability - true_probability)
        maximum = float(np.max(predicted_lnpi))
        normalization_error = abs(
            maximum
            + float(np.log(np.sum(np.exp(predicted_lnpi - maximum))))
        )
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
                "normalization_error": normalization_error,
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> Dict[str, dict]:
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


def json_ready_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DeepONet on masked absolute integer-N MPDs."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "outputs/run_deeponet_absolute_n",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--p", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
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
    feature_archive = np.load(args.data / "feature_values.npz", allow_pickle=False)
    features = np.asarray(feature_archive["features"], dtype=float)
    feature_names = [str(name) for name in feature_archive["feature_names"]]
    value_archive = np.load(args.data / "mpd_values.npz", allow_pickle=False)
    values = np.asarray(value_archive["lnpi"], dtype=float)
    n_axis = np.asarray(value_archive["n_axis"], dtype=int)
    masks = np.asarray(
        np.load(args.data / "mpd_masks.npz", allow_pickle=False)["valid_mask"],
        dtype=bool,
    )

    if values.shape != masks.shape or values.shape[0] != len(index):
        raise ValueError("Prepared value, mask, and index shapes are inconsistent.")
    if features.shape != (len(index), len(feature_names)):
        raise ValueError("Prepared feature shape is inconsistent with the index.")
    if not np.array_equal(np.isfinite(values), masks):
        raise ValueError("Finite MPD values do not exactly match valid-mask positions.")
    if not np.array_equal(n_axis, np.arange(len(n_axis))):
        raise ValueError("The absolute-N coordinate must be contiguous from zero.")

    values_filled = np.where(masks, values, 0.0).astype(np.float32)
    split_indices = {
        split: np.where(index["split"].eq(split).to_numpy())[0]
        for split in ("train", "validation", "test")
    }
    if any(len(rows) == 0 for rows in split_indices.values()):
        raise ValueError("Training, validation, and test splits must all be non-empty.")

    feature_scaler = StandardScaler()
    feature_scaler.fit(features[split_indices["train"]])
    scaled_features = feature_scaler.transform(features).astype(np.float32)

    loaders = {
        split: make_loader(
            scaled_features,
            values_filled,
            masks,
            rows,
            args.batch_size,
            shuffle=split == "train",
        )
        for split, rows in split_indices.items()
    }
    all_loader = make_loader(
        scaled_features,
        values_filled,
        masks,
        np.arange(len(index)),
        args.batch_size,
        shuffle=False,
    )

    device = torch.device(args.device)
    global_nmax = int(n_axis[-1])
    coordinates = torch.tensor(
        n_axis / max(global_nmax, 1),
        dtype=torch.float32,
        device=device,
    ).view(-1, 1)
    model = DeepONet(
        feature_dim=features.shape[1],
        p=args.p,
        hidden_dim=args.hidden_dim,
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
    history: List[dict] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_mofs = 0
        for batch_features, batch_values, batch_masks, _ in loaders["train"]:
            batch_features = batch_features.to(device)
            batch_values = batch_values.to(device)
            batch_masks = batch_masks.to(device)
            prediction = masked_log_softmax(
                model(batch_features, coordinates),
                batch_masks,
            )
            per_mof_loss = per_mof_masked_mae(
                prediction,
                batch_values,
                batch_masks,
            )
            loss = per_mof_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(per_mof_loss.detach().sum().item())
            n_mofs += len(batch_features)

        train_loss = loss_sum / n_mofs
        validation_loss = evaluate(
            model,
            loaders["validation"],
            coordinates,
            device,
        )
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
                "train_loss_mean_per_mof_masked_lnpi_mae": train_loss,
                "validation_loss_mean_per_mof_masked_lnpi_mae": validation_loss,
                "elapsed_seconds": elapsed,
            }
        )
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:5d} train_mae={train_loss:.6f} "
                f"val_mae={validation_loss:.6f} elapsed_s={elapsed:.1f}"
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
    test_loss = evaluate(model, loaders["test"], coordinates, device)
    predictions = predict_all(
        model,
        all_loader,
        coordinates,
        device,
        values.shape,
    )
    metrics_by_mof = reconstruction_metrics(
        predictions,
        values,
        masks,
        index,
    )
    metric_summary = summarize_metrics(metrics_by_mof)

    pd.DataFrame(history).to_csv(args.output / "training_log.csv", index=False)
    metrics_by_mof.to_csv(
        args.output / "reconstruction_metrics_by_mof.csv",
        index=False,
    )
    np.savez_compressed(
        args.output / "predicted_mpds.npz",
        predicted_lnpi=predictions,
        valid_mask=masks,
        n_axis=n_axis.astype(np.int16),
        normalized_n_axis=(n_axis / max(global_nmax, 1)).astype(np.float32),
        mof_ids=index["mof_id"].to_numpy(dtype=str),
        splits=index["split"].to_numpy(dtype=str),
    )
    np.savez_compressed(
        args.output / "feature_scaler.npz",
        mean=feature_scaler.mean_,
        scale=feature_scaler.scale_,
        feature_names=np.asarray(feature_names),
    )
    (args.output / "splits.json").write_text(
        json.dumps(
            {
                "train_mofs": index.loc[
                    index["split"].eq("train"), "mof_id"
                ].tolist(),
                "val_mofs": index.loc[
                    index["split"].eq("validation"), "mof_id"
                ].tolist(),
                "test_mofs": index.loc[
                    index["split"].eq("test"), "mof_id"
                ].tolist(),
            },
            indent=2,
        )
        + "\n"
    )

    completed_epochs = int(history[-1]["epoch"])
    total_time = time.perf_counter() - start_time
    metrics = {
        "training_loss": "mean_per_mof_masked_normalized_lnpi_mae",
        "coordinate": "global_N_divided_by_205",
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "stopped_early": completed_epochs < args.epochs,
        "best_validation_loss_mean_per_mof_masked_lnpi_mae": best_validation_loss,
        "test_loss_mean_per_mof_masked_lnpi_mae": test_loss,
        "training_time_seconds": total_time,
        "n_train_mofs": int(len(split_indices["train"])),
        "n_validation_mofs": int(len(split_indices["validation"])),
        "n_test_mofs": int(len(split_indices["test"])),
        "maximum_reconstruction_normalization_error": float(
            metrics_by_mof["normalization_error"].max()
        ),
        "reconstruction_metrics": metric_summary,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "config.json").write_text(
        json.dumps(
            json_ready_args(args)
            | {
                "feature_names": feature_names,
                "global_Nmax": global_nmax,
                "coordinate_definition": "N/global_Nmax",
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Test masked lnPi MAE: {test_loss:.6f}")
    print(
        "Test mean MPD total-variation distance: "
        f"{metric_summary['test']['mean_total_variation_distance']:.6f}"
    )
    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    main()


