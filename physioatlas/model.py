from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class PhysioAtlasOutput:
    waveform: torch.Tensor
    confidence: torch.Tensor
    observability: torch.Tensor
    log_variance: torch.Tensor
    latent: torch.Tensor


class FeatureNormalizer(nn.Module):
    """Per-window standardization with finite-value protection."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.nan_to_num(values)
        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True).clamp_min(1e-4)
        return (values - mean) / std


class ComplexLinkEncoder(nn.Module):
    """Encode canonical complex RF features per physical link.

    Input values have already been canonicalized by the dataset into explicit
    real, imaginary, magnitude, sine-phase, and cosine-phase channels. A shared
    encoder prevents link ordering from receiving its own private parameter
    block, while geometry-conditioned attention pools evidence across links.
    """

    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.value_encoder = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        link_geometry: torch.Tensor,
        link_mask: torch.Tensor,
        rf_observed: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # values [B,T,L,C], geometry [B,L,12], mask [B,L]
        values = torch.nan_to_num(values)
        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True).clamp_min(1e-4)
        encoded = self.value_encoder((values - mean) / std)
        geometry = self.geometry_encoder(torch.nan_to_num(link_geometry))
        geometry_time = geometry[:, None].expand(-1, values.shape[1], -1, -1)
        scores = self.score(torch.cat([encoded, geometry_time], dim=-1)).squeeze(-1)
        valid = link_mask[:, None, :].bool()
        if rf_observed is not None:
            valid = valid & (rf_observed.squeeze(-1) > 0)
        # Avoid all-masked softmax rows: mark one dummy link valid, then zero the
        # resulting pooled value with the observability gate below.
        any_valid = valid.any(dim=-1, keepdim=True)
        safe_valid = valid.clone()
        safe_valid[..., 0] |= ~any_valid.squeeze(-1)
        scores = scores.masked_fill(~safe_valid, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * safe_valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = (encoded * weights[..., None]).sum(dim=2)
        pooled = pooled * any_valid.float()
        link_observability = valid.float().mean(dim=-1, keepdim=True)
        return pooled, link_observability


class PhysioAtlasModel(nn.Module):
    """Geometry-aware waveform translator with calibrated abstention outputs."""

    def __init__(
        self,
        rf_features: int,
        auxiliary_features: int,
        target_features: int,
        geometry_features: int,
        *,
        hidden_dim: int = 64,
        layers: int = 2,
        bidirectional: bool = True,
        architecture: str = "compact",
        linked_rf_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if min(rf_features, auxiliary_features, target_features, geometry_features) <= 0:
            raise ValueError("feature dimensions must be positive")
        if architecture not in {"compact", "complex_link"}:
            raise ValueError(f"Unknown architecture {architecture!r}")
        self.architecture = architecture
        self.normalizer = FeatureNormalizer()
        branch_dim = max(4, hidden_dim // 2)
        if architecture == "compact":
            self.rf_encoder: nn.Module = nn.Sequential(
                nn.Linear(rf_features, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, branch_dim),
                nn.GELU(),
            )
            rf_output_dim = branch_dim
            self.link_encoder = None
        else:
            if not linked_rf_channels:
                raise ValueError("complex_link architecture requires linked_rf_channels")
            self.link_encoder = ComplexLinkEncoder(linked_rf_channels, hidden_dim)
            self.rf_encoder = nn.Identity()
            rf_output_dim = hidden_dim
        self.aux_encoder = nn.Sequential(
            nn.Linear(auxiliary_features, branch_dim),
            nn.GELU(),
            nn.LayerNorm(branch_dim),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_features, branch_dim),
            nn.GELU(),
            nn.LayerNorm(branch_dim),
        )
        self.temporal = nn.GRU(
            input_size=rf_output_dim + branch_dim * 2 + 1,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=0.1 if layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        temporal_dim = hidden_dim * (2 if bidirectional else 1)
        self.waveform_head = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, target_features)
        )
        self.observability_head = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, target_features),
            nn.Sigmoid(),
        )
        self.log_variance_head = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, target_features),
        )

    def forward(
        self,
        rf: torch.Tensor,
        auxiliary: torch.Tensor,
        geometry: torch.Tensor,
        rf_linked: Optional[torch.Tensor] = None,
        link_geometry: Optional[torch.Tensor] = None,
        link_mask: Optional[torch.Tensor] = None,
        rf_observed: Optional[torch.Tensor] = None,
    ) -> PhysioAtlasOutput:
        if self.architecture == "compact":
            rf_latent = self.rf_encoder(self.normalizer(rf))
            if rf_observed is None:
                link_observability = torch.ones(
                    (*rf.shape[:2], 1), device=rf.device, dtype=rf.dtype
                )
            else:
                link_observability = rf_observed.squeeze(-1).mean(dim=-1, keepdim=True)
        else:
            if rf_linked is None or link_geometry is None or link_mask is None:
                raise ValueError("complex_link forward requires linked RF and link geometry tensors")
            assert self.link_encoder is not None
            rf_latent, link_observability = self.link_encoder(
                rf_linked,
                link_geometry,
                link_mask,
                rf_observed,
            )
        aux_latent = self.aux_encoder(self.normalizer(auxiliary))
        geometry_latent = self.geometry_encoder(torch.nan_to_num(geometry))
        geometry_latent = geometry_latent[:, None, :].expand(-1, rf.shape[1], -1)
        fused = torch.cat(
            [rf_latent, aux_latent, geometry_latent, link_observability], dim=-1
        )
        temporal, _ = self.temporal(fused)
        observability = self.observability_head(temporal)
        log_variance = self.log_variance_head(temporal).clamp(-8.0, 5.0)
        return PhysioAtlasOutput(
            waveform=self.waveform_head(temporal),
            confidence=observability,
            observability=observability,
            log_variance=log_variance,
            latent=temporal,
        )


def masked_waveform_loss(
    output: PhysioAtlasOutput,
    target: torch.Tensor,
    mask: torch.Tensor,
    observability_target: Optional[torch.Tensor] = None,
    *,
    uncertainty_weight: float = 0.05,
    observability_weight: float = 0.1,
) -> torch.Tensor:
    mask = mask.expand_as(target)
    denominator = mask.sum().clamp_min(1.0)
    squared_error = (output.waveform - target) ** 2
    # Gaussian heteroscedastic NLL. The extra weight allows conservative use
    # while the uncertainty head is still being calibrated on real data.
    nll = 0.5 * (torch.exp(-output.log_variance) * squared_error + output.log_variance)
    nll_loss = (nll * mask).sum() / denominator
    absolute_error = ((output.waveform - target).abs() * mask).sum() / denominator
    observed_target = observability_target if observability_target is not None else mask
    observed_target = observed_target.expand_as(output.observability)
    observability_loss = nn.functional.binary_cross_entropy(
        output.observability.clamp(1e-5, 1.0 - 1e-5), observed_target
    )
    variance_regularizer = (output.log_variance.square() * mask).sum() / denominator
    return (
        nll_loss
        + 0.1 * absolute_error
        + observability_weight * observability_loss
        + uncertainty_weight * 0.01 * variance_regularizer
    )
