from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PhysioWindowDataset, WindowExample
from .metrics import (
    binary_brier_score,
    masked_correlation,
    masked_mae,
    masked_rmse,
    prediction_interval_coverage,
)
from .model import PhysioAtlasModel, masked_waveform_loss
from .splits import count_examples_by_domain, group_split
from .utils import make_json_safe


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    layers: int = 2
    seed: int = 42
    validation_fraction: float = 0.25
    device: str = "auto"
    use_auxiliary: bool = True
    use_geometry: bool = True
    architecture: str = "complex_link"
    causal: bool = False
    uncertainty_weight: float = 0.05
    observability_weight: float = 0.1
    split_strategy: str = "subject"
    held_out_groups: tuple[str, ...] = field(default_factory=tuple)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _model_inputs(batch: dict, device: torch.device, config: TrainConfig) -> dict:
    auxiliary = batch["auxiliary"].to(device)
    geometry = batch["geometry"].to(device)
    if not config.use_auxiliary:
        auxiliary = torch.zeros_like(auxiliary)
    if not config.use_geometry:
        geometry = torch.zeros_like(geometry)
    return {
        "rf": batch["rf"].to(device),
        "auxiliary": auxiliary,
        "geometry": geometry,
        "rf_linked": batch["rf_linked"].to(device),
        "link_geometry": batch["link_geometry"].to(device),
        "link_mask": batch["link_mask"].to(device),
        "rf_observed": batch["rf_observed"].to(device),
    }


def _evaluate(
    model: PhysioAtlasModel,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    deviations: list[np.ndarray] = []
    observability: list[np.ndarray] = []
    observability_targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(**_model_inputs(batch, device, config))
            predictions.append(output.waveform.cpu().numpy())
            deviations.append(torch.exp(0.5 * output.log_variance).cpu().numpy())
            observability.append(output.observability.cpu().numpy())
            targets.append(batch["target"].numpy())
            masks.append(batch["target_mask"].numpy())
            observability_targets.append(batch["observability_target"].numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    mask = np.concatenate(masks)
    std = np.concatenate(deviations)
    observation_probability = np.concatenate(observability)
    observation_target = np.concatenate(observability_targets)
    return {
        "rmse": masked_rmse(prediction, target, mask),
        "mae": masked_mae(prediction, target, mask),
        "correlation": masked_correlation(prediction, target, mask),
        "interval_90_coverage": prediction_interval_coverage(
            prediction, target, std, mask
        ),
        "mean_predicted_std": float(np.mean(std)),
        "observability_brier": binary_brier_score(
            observation_probability, observation_target
        ),
        "mean_observability": float(np.mean(observation_probability)),
    }


def train_model(
    examples: list[WindowExample],
    output_dir: Union[str, Path],
    config: TrainConfig,
) -> dict:
    set_deterministic(config.seed)
    train_examples, validation_examples, split = group_split(
        examples,
        strategy=config.split_strategy,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        held_out_groups=config.held_out_groups,
    )

    train_dataset = PhysioWindowDataset(train_examples)
    validation_dataset = PhysioWindowDataset(validation_examples)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    first = train_examples[0]
    model = PhysioAtlasModel(
        rf_features=first.rf.shape[1],
        auxiliary_features=first.auxiliary.shape[1],
        target_features=first.target.shape[1],
        geometry_features=first.geometry.shape[0],
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        bidirectional=not config.causal,
        architecture=config.architecture,
        linked_rf_channels=first.rf_linked.shape[2],
    )
    device = resolve_device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(config.epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(**_model_inputs(batch, device, config))
            loss = masked_waveform_loss(
                output,
                batch["target"].to(device),
                batch["target_mask"].to(device),
                batch["observability_target"].to(device),
                uncertainty_weight=config.uncertainty_weight,
                observability_weight=config.observability_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = _evaluate(model, validation_loader, device, config)
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics}
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "rf_features": first.rf.shape[1],
                "auxiliary_features": first.auxiliary.shape[1],
                "target_features": first.target.shape[1],
                "geometry_features": first.geometry.shape[0],
                "hidden_dim": config.hidden_dim,
                "layers": config.layers,
                "bidirectional": not config.causal,
                "architecture": config.architecture,
                "linked_rf_channels": first.rf_linked.shape[2],
            },
            "train_config": asdict(config),
        },
        checkpoint_path,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    result = {
        "schema_version": "physioatlas.run.v2",
        "status": "completed",
        "research_only": True,
        "clinical_claim_allowed": False,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "split_strategy": config.split_strategy,
        "train_groups": split["train_groups"],
        "validation_groups": split["validation_groups"],
        "train_subjects": sorted({e.subject_id for e in train_examples}),
        "validation_subjects": sorted({e.subject_id for e in validation_examples}),
        "n_train_windows": len(train_examples),
        "n_validation_windows": len(validation_examples),
        "domain_counts": {
            "train": count_examples_by_domain(train_examples),
            "validation": count_examples_by_domain(validation_examples),
        },
        "history": history,
        "final_metrics": history[-1],
        "checkpoint": str(checkpoint_path.name),
        "checkpoint_sha256": checkpoint_sha256,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "config": asdict(config),
    }
    result = make_json_safe(result)
    with (output / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result
