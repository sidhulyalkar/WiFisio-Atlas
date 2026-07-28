from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import torch


def _state_l2_norm(state: Mapping[str, torch.Tensor]) -> float:
    total = sum(float(tensor.detach().float().square().sum()) for tensor in state.values())
    return float(np.sqrt(total))


def aggregate_model_deltas(
    deltas: Sequence[Mapping[str, torch.Tensor]],
    *,
    weights: Optional[Sequence[float]] = None,
    clip_norm: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Aggregate clipped client deltas locally.

    This is a research scaffold, not a secure aggregation protocol. It provides
    deterministic clipping and optional Gaussian noise so federated experiments
    can be tested before networking or cryptographic aggregation is introduced.
    """
    if not deltas:
        raise ValueError("At least one model delta is required")
    if clip_norm <= 0 or noise_std < 0:
        raise ValueError("clip_norm must be positive and noise_std non-negative")
    keys = set(deltas[0])
    if any(set(delta) != keys for delta in deltas):
        raise ValueError("All model deltas must have identical keys")
    if weights is None:
        normalized = np.full(len(deltas), 1.0 / len(deltas), dtype=np.float64)
    else:
        if len(weights) != len(deltas) or any(weight < 0 for weight in weights):
            raise ValueError("weights must be non-negative and match the client count")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        normalized = np.asarray(weights, dtype=np.float64) / total

    aggregate = {key: torch.zeros_like(deltas[0][key]) for key in keys}
    for weight, delta in zip(normalized, deltas):
        norm = _state_l2_norm(delta)
        scale = min(1.0, clip_norm / max(norm, 1e-12))
        for key in keys:
            aggregate[key] = aggregate[key] + delta[key] * float(weight * scale)
    if noise_std:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for key, tensor in aggregate.items():
            noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
            aggregate[key] = tensor + noise.to(tensor.device) * noise_std
    return aggregate
