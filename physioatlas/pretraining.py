from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import PhysioWindowDataset, WindowExample
from .utils import make_json_safe


@dataclass(frozen=True)
class RFPretrainConfig:
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    mask_fraction: float = 0.25
    seed: int = 42
    device: str = "auto"


class RFMaskedFoundationEncoder(nn.Module):
    """Masked multi-link encoder for reusable RF representations.

    The objective reconstructs held-out canonical RF channels while aligning
    two independently masked views of the same window. Geometry is injected per
    link, and the exported embedding is independent of any physiological label.
    """

    def __init__(self, channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.value_encoder = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
        )

    def forward(
        self,
        values: torch.Tensor,
        geometry: torch.Tensor,
        link_mask: torch.Tensor,
        observed: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # channel_mask [B,T,L,C], True means hidden from the encoder.
        clean = torch.nan_to_num(values)
        normalized = clean - clean.mean(dim=1, keepdim=True)
        normalized = normalized / normalized.std(dim=1, keepdim=True).clamp_min(1e-4)
        encoded_input = normalized.masked_fill(channel_mask, 0.0)
        link_latent = self.value_encoder(encoded_input)
        geometry_latent = self.geometry_encoder(torch.nan_to_num(geometry))
        link_latent = link_latent + geometry_latent[:, None]
        valid = link_mask[:, None, :, None] * observed
        pooled = (link_latent * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        temporal, _ = self.temporal(pooled)
        temporal_links = temporal[:, :, None].expand(-1, -1, values.shape[2], -1)
        geometry_time = geometry_latent[:, None].expand(-1, values.shape[1], -1, -1)
        reconstruction = self.decoder(
            torch.cat([link_latent, temporal_links, geometry_time], dim=-1)
        )
        embedding = nn.functional.normalize(self.projection(temporal.mean(dim=1)), dim=-1)
        return reconstruction, embedding


def _random_channel_mask(
    shape: torch.Size,
    *,
    fraction: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if not 0.0 < fraction < 1.0:
        raise ValueError("mask_fraction must be between zero and one")
    random_values = torch.rand(shape, generator=generator, device="cpu")
    return (random_values < fraction).to(device)


def _view_loss(
    model: RFMaskedFoundationEncoder,
    batch: dict,
    mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = batch["rf_linked"].to(device)
    observed = batch["rf_observed"].to(device)
    link_mask = batch["link_mask"].to(device)
    reconstruction, embedding = model(
        values,
        batch["link_geometry"].to(device),
        link_mask,
        observed,
        mask,
    )
    valid = mask.float() * observed * link_mask[:, None, :, None]
    reconstruction_loss = ((reconstruction - values) ** 2 * valid).sum() / valid.sum().clamp_min(1.0)
    return reconstruction_loss, embedding


def train_rf_foundation(
    examples: list[WindowExample],
    output_dir: Union[str, Path],
    config: RFPretrainConfig,
) -> dict:
    if not examples:
        raise ValueError("RF pretraining requires examples")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available() else (
            "cpu" if config.device == "auto" else config.device
        )
    )
    dataset = PhysioWindowDataset(examples)
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    model = RFMaskedFoundationEncoder(
        channels=examples[0].rf_linked.shape[2],
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    mask_generator = torch.Generator().manual_seed(config.seed + 1000)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []
        reconstruction_losses: list[float] = []
        alignment_losses: list[float] = []
        for batch in loader:
            shape = batch["rf_linked"].shape
            mask_a = _random_channel_mask(
                shape,
                fraction=config.mask_fraction,
                generator=mask_generator,
                device=device,
            )
            mask_b = _random_channel_mask(
                shape,
                fraction=config.mask_fraction,
                generator=mask_generator,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            reconstruction_a, embedding_a = _view_loss(model, batch, mask_a, device)
            reconstruction_b, embedding_b = _view_loss(model, batch, mask_b, device)
            alignment = (1.0 - (embedding_a * embedding_b).sum(dim=-1)).mean()
            std = torch.cat([embedding_a, embedding_b], dim=0).std(dim=0)
            variance = torch.relu(0.1 - std).mean()
            loss = 0.5 * (reconstruction_a + reconstruction_b) + 0.1 * alignment + 0.05 * variance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            reconstruction_losses.append(float((0.5 * (reconstruction_a + reconstruction_b)).detach().cpu()))
            alignment_losses.append(float(alignment.detach().cpu()))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(epoch_losses)),
                "masked_reconstruction_mse": float(np.mean(reconstruction_losses)),
                "view_alignment_loss": float(np.mean(alignment_losses)),
            }
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "rf_foundation.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "channels": examples[0].rf_linked.shape[2],
                "hidden_dim": config.hidden_dim,
                "embedding_dim": 128,
            },
            "pretrain_config": asdict(config),
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = make_json_safe(
        {
            "schema_version": "physioatlas.rf-pretrain.v1",
            "status": "completed",
            "research_only": True,
            "label_free": True,
            "windows": len(examples),
            "subjects": sorted({example.subject_id for example in examples}),
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": digest,
            "history": history,
            "final_metrics": history[-1],
            "config": asdict(config),
        }
    )
    (output / "run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def load_rf_foundation(checkpoint_path: Union[str, Path]) -> RFMaskedFoundationEncoder:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = RFMaskedFoundationEncoder(
        channels=int(config["channels"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model
