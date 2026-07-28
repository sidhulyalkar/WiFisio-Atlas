from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Union

import numpy as np

from .schema import RFRepresentation


@dataclass(frozen=True)
class ActiveSensingCandidate:
    candidate_id: str
    predicted_information_gain: float
    observability_gain: float
    novelty: float = 0.0
    energy_cost: float = 0.0
    switch_cost: float = 0.0


@dataclass(frozen=True)
class ActiveSensingDecision:
    candidate_id: str
    score: float
    ranked: list[dict[str, Union[float, str]]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def canonical_to_complex(values: np.ndarray, representation: Union[RFRepresentation, str]) -> np.ndarray:
    array = np.asarray(values)
    representation = RFRepresentation(representation)
    if np.iscomplexobj(array):
        return array.astype(np.complex64)
    if array.shape[-1] % 2:
        raise ValueError("Amplitude/phase and IQ representations require an even feature count")
    half = array.shape[-1] // 2
    first = array[..., :half].astype(np.float64)
    second = array[..., half:].astype(np.float64)
    if representation == RFRepresentation.AMPLITUDE_PHASE:
        return (first * np.exp(1j * second)).astype(np.complex64)
    if representation == RFRepresentation.IQ:
        return (first + 1j * second).astype(np.complex64)
    raise ValueError(f"Representation {representation.value!r} is not complex RF")


def reconstruct_cir(
    complex_csi: np.ndarray,
    *,
    n_delay_bins: Optional[int] = None,
    window: bool = True,
) -> np.ndarray:
    """Compute a deterministic delay-domain response from complex CSI.

    This is an IFFT baseline, not super-resolution tomography. It is useful for
    delay-spread features and software validation but cannot beat the bandwidth
    limit of the underlying radio measurements.
    """
    csi = np.asarray(complex_csi)
    if not np.iscomplexobj(csi) or csi.ndim < 1:
        raise ValueError("complex_csi must be a complex array")
    subcarriers = csi.shape[-1]
    if subcarriers < 2:
        raise ValueError("At least two subcarriers are required")
    bins = int(n_delay_bins or subcarriers)
    if bins < subcarriers:
        raise ValueError("n_delay_bins cannot be smaller than the subcarrier count")
    tapered = csi * np.hanning(subcarriers) if window else csi
    return np.fft.ifft(tapered, n=bins, axis=-1).astype(np.complex64)


def delay_spread_features(cir: np.ndarray, *, delay_resolution_s: float) -> dict[str, float]:
    response = np.asarray(cir)
    if not np.iscomplexobj(response) or response.shape[-1] < 2:
        raise ValueError("cir must be a complex response with at least two delay bins")
    power = np.abs(response) ** 2
    mean_power = power.reshape(-1, power.shape[-1]).mean(axis=0)
    total = float(mean_power.sum())
    if total <= 0 or not np.isfinite(total):
        return {"rms_delay_spread_s": 0.0, "dominant_tap_ratio": 0.0, "tap_count": 0.0}
    probability = mean_power / total
    delays = np.arange(mean_power.size, dtype=np.float64) * delay_resolution_s
    centroid = float(np.sum(delays * probability))
    rms = float(np.sqrt(np.sum(((delays - centroid) ** 2) * probability)))
    threshold = float(mean_power.max() * 0.1)
    return {
        "rms_delay_spread_s": rms,
        "dominant_tap_ratio": float(mean_power.max() / total),
        "tap_count": float(np.count_nonzero(mean_power >= threshold)),
    }


def delay_doppler_map(
    complex_csi: np.ndarray,
    *,
    n_delay_bins: Optional[int] = None,
    n_doppler_bins: Optional[int] = None,
) -> np.ndarray:
    """Build a magnitude delay-Doppler map from [time, subcarrier] CSI."""
    csi = np.asarray(complex_csi)
    if csi.ndim != 2 or not np.iscomplexobj(csi):
        raise ValueError("complex_csi must have shape [time, subcarrier]")
    cir = reconstruct_cir(csi, n_delay_bins=n_delay_bins)
    doppler_bins = int(n_doppler_bins or csi.shape[0])
    doppler = np.fft.fftshift(np.fft.fft(cir, n=doppler_bins, axis=0), axes=0)
    return np.abs(doppler).astype(np.float32)


def select_active_measurement(
    candidates: Iterable[ActiveSensingCandidate],
    *,
    exploration_weight: float = 0.1,
    energy_weight: float = 0.05,
    switch_weight: float = 0.05,
) -> ActiveSensingDecision:
    """Rank the next link, channel, or beam by expected research utility."""
    ranked: list[dict[str, Union[float, str]]] = []
    for candidate in candidates:
        numeric = (
            candidate.predicted_information_gain,
            candidate.observability_gain,
            candidate.novelty,
            candidate.energy_cost,
            candidate.switch_cost,
        )
        if not all(np.isfinite(value) for value in numeric):
            continue
        score = (
            candidate.predicted_information_gain
            + candidate.observability_gain
            + exploration_weight * candidate.novelty
            - energy_weight * candidate.energy_cost
            - switch_weight * candidate.switch_cost
        )
        ranked.append({"candidate_id": candidate.candidate_id, "score": float(score)})
    if not ranked:
        raise ValueError("No finite active-sensing candidate was supplied")
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["candidate_id"])))
    return ActiveSensingDecision(
        candidate_id=str(ranked[0]["candidate_id"]),
        score=float(ranked[0]["score"]),
        ranked=ranked,
    )
