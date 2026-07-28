from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import geometry_summary, link_feature_matrix
from .io import LoadedSession, iter_manifests, load_session
from .schema import Modality, RFRepresentation, StreamReference
from .synchronization import align_streams


@dataclass(frozen=True)
class WindowExample:
    rf: np.ndarray
    rf_linked: np.ndarray
    rf_observed: np.ndarray
    auxiliary: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    observability_target: np.ndarray
    geometry: np.ndarray
    link_geometry: np.ndarray
    link_mask: np.ndarray
    subject_id: str
    session_id: str
    environment_id: str = "unknown"
    posture: Optional[str] = None
    condition: Optional[str] = None
    hardware_domain: str = "unknown"


def subject_split(
    subject_ids: Sequence[str], *, validation_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    unique = sorted(set(subject_ids))
    if len(unique) < 2:
        raise ValueError("Subject-held-out evaluation needs at least two subjects")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(unique))
    n_val = max(1, int(round(len(unique) * validation_fraction)))
    n_val = min(n_val, len(unique) - 1)
    validation = set(shuffled[:n_val])
    return set(shuffled[n_val:]), validation


def _canonical_rf(values: np.ndarray, representation: RFRepresentation) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        values = values.reshape(values.shape[0], -1)
    if representation in {RFRepresentation.AMPLITUDE_PHASE, RFRepresentation.IQ}:
        if values.shape[1] % 2:
            raise ValueError(
                f"RF representation {representation.value} requires an even feature count"
            )
        half = values.shape[1] // 2
        first, second = values[:, :half], values[:, half:]
        if representation == RFRepresentation.AMPLITUDE_PHASE:
            amplitude, phase = first, second
            real = amplitude * np.cos(phase)
            imag = amplitude * np.sin(phase)
        else:
            real, imag = first, second
            amplitude = np.sqrt(real * real + imag * imag)
            phase = np.arctan2(imag, real)
        return np.concatenate(
            [real, imag, np.log1p(np.abs(amplitude)), np.sin(phase), np.cos(phase)],
            axis=1,
        ).astype(np.float32)
    # Generic real features receive an explicit zero-imaginary channel. This is
    # a declared fallback, not an attempt to infer missing phase.
    zeros = np.zeros_like(values)
    return np.concatenate(
        [values, zeros, np.log1p(np.abs(values)), zeros, np.ones_like(values)],
        axis=1,
    ).astype(np.float32)


def _rf_references(session: LoadedSession) -> list[StreamReference]:
    return [
        reference
        for reference in session.manifest.streams
        if reference.modality in {Modality.WIFI_CSI, Modality.MMWAVE, Modality.UWB}
    ]


def _linked_rf(
    session: LoadedSession,
    aligned,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    references = _rf_references(session)
    if not references:
        raise ValueError(f"Session {session.manifest.session_id} contains no RF stream")
    geometry_ids, geometry_features = link_feature_matrix(session.geometry)
    geometry_by_id = {link_id: geometry_features[index] for index, link_id in enumerate(geometry_ids)}
    link_values: list[np.ndarray] = []
    link_observed: list[np.ndarray] = []
    link_quality: list[np.ndarray] = []
    link_geometry: list[np.ndarray] = []

    for reference in references:
        key = f"{reference.modality.value}:{reference.sensor_id}"
        canonical = _canonical_rf(aligned.values[key], reference.representation)
        declared_ids = list(reference.metadata.get("link_ids", []))
        if not declared_ids and len(references) == 1 and geometry_ids:
            declared_ids = geometry_ids
        if not declared_ids:
            declared_ids = [str(reference.metadata.get("link_id", key))]
        if canonical.shape[1] % len(declared_ids):
            raise ValueError(
                f"RF stream {key} has {canonical.shape[1]} canonical features that cannot be "
                f"divided across {len(declared_ids)} declared links"
            )
        channels = canonical.shape[1] // len(declared_ids)
        for index, link_id in enumerate(declared_ids):
            link_values.append(canonical[:, index * channels : (index + 1) * channels])
            link_observed.append(aligned.observed[key].astype(np.float32)[:, None])
            link_quality.append(aligned.quality[key].astype(np.float32)[:, None])
            link_geometry.append(geometry_by_id.get(link_id, np.zeros(12, dtype=np.float32)))

    max_channels = max(value.shape[1] for value in link_values)
    time_steps = link_values[0].shape[0]
    linked = np.zeros((time_steps, len(link_values), max_channels), dtype=np.float32)
    observed = np.zeros((time_steps, len(link_values), 1), dtype=np.float32)
    quality = np.zeros((time_steps, len(link_values), 1), dtype=np.float32)
    for index, value in enumerate(link_values):
        linked[:, index, : value.shape[1]] = value
        observed[:, index] = link_observed[index]
        quality[:, index] = link_quality[index]
    return linked, observed, quality, np.stack(link_geometry).astype(np.float32)


def session_windows(
    session: LoadedSession,
    *,
    sample_rate_hz: float,
    window_seconds: float,
    stride_seconds: float,
    target_key: str = "ecg",
    max_gap_s: float = 0.25,
) -> list[WindowExample]:
    aligned = align_streams(
        session.streams,
        sample_rate_hz=sample_rate_hz,
        max_gap_s=max_gap_s,
    )
    rf_linked, rf_observed, rf_quality, link_geometry = _linked_rf(session, aligned)
    rf = rf_linked.reshape(rf_linked.shape[0], -1).astype(np.float32)
    aux_parts = [
        value
        for key, value in aligned.values.items()
        if key.startswith("pose:")
        or (key.startswith("resp_belt:") and target_key != "resp_belt")
        or (key.startswith("ppg:") and target_key != "ppg")
    ]
    auxiliary = (
        np.concatenate(aux_parts, axis=1).astype(np.float32)
        if aux_parts
        else np.zeros((rf.shape[0], 1), dtype=np.float32)
    )
    target_candidates = [
        (key, value)
        for key, value in aligned.values.items()
        if key.startswith(target_key + ":")
    ]
    if not target_candidates:
        raise ValueError(
            f"Session {session.manifest.session_id} has no target stream {target_key!r}"
        )
    target_stream_key, target = target_candidates[0]
    target = target.astype(np.float32)
    target_mask = aligned.observed[target_stream_key].astype(np.float32)[:, None]
    rf_available = (rf_observed * (rf_quality >= 0.2)).max(axis=1)
    observability_target = (target_mask * rf_available).astype(np.float32)

    window = max(2, int(round(window_seconds * sample_rate_hz)))
    stride = max(1, int(round(stride_seconds * sample_rate_hz)))
    geometry = geometry_summary(session.geometry)
    hardware = sorted(
        {
            sensor.hardware or "unknown"
            for sensor in session.geometry.sensors
            if sensor.modality in {Modality.WIFI_CSI, Modality.MMWAVE, Modality.UWB}
        }
    )
    hardware_domain = "+".join(hardware) if hardware else "unknown"
    examples: list[WindowExample] = []
    for start in range(0, rf.shape[0] - window + 1, stride):
        stop = start + window
        examples.append(
            WindowExample(
                rf=rf[start:stop],
                rf_linked=rf_linked[start:stop],
                rf_observed=rf_observed[start:stop],
                auxiliary=auxiliary[start:stop],
                target=target[start:stop],
                target_mask=target_mask[start:stop],
                observability_target=observability_target[start:stop],
                geometry=geometry,
                link_geometry=link_geometry,
                link_mask=np.ones(link_geometry.shape[0], dtype=np.float32),
                subject_id=session.manifest.subject_id,
                session_id=session.manifest.session_id,
                environment_id=session.manifest.environment_id,
                posture=session.manifest.posture,
                condition=session.manifest.condition,
                hardware_domain=hardware_domain,
            )
        )
    return examples


def _pad_examples(examples: list[WindowExample]) -> list[WindowExample]:
    max_links = max(example.rf_linked.shape[1] for example in examples)
    max_channels = max(example.rf_linked.shape[2] for example in examples)
    padded: list[WindowExample] = []
    for example in examples:
        time_steps, links, channels = example.rf_linked.shape
        linked = np.zeros((time_steps, max_links, max_channels), dtype=np.float32)
        linked[:, :links, :channels] = example.rf_linked
        observed = np.zeros((time_steps, max_links, 1), dtype=np.float32)
        observed[:, :links] = example.rf_observed
        link_geometry = np.zeros((max_links, 12), dtype=np.float32)
        link_geometry[:links] = example.link_geometry
        link_mask = np.zeros(max_links, dtype=np.float32)
        link_mask[:links] = example.link_mask
        padded.append(
            replace(
                example,
                rf=linked.reshape(time_steps, -1),
                rf_linked=linked,
                rf_observed=observed,
                link_geometry=link_geometry,
                link_mask=link_mask,
            )
        )
    return padded


class PhysioWindowDataset(Dataset):
    def __init__(self, examples: list[WindowExample]):
        if not examples:
            raise ValueError("Dataset requires at least one window")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Union[torch.Tensor, str]]:
        example = self.examples[index]
        return {
            "rf": torch.from_numpy(example.rf),
            "rf_linked": torch.from_numpy(example.rf_linked),
            "rf_observed": torch.from_numpy(example.rf_observed),
            "auxiliary": torch.from_numpy(example.auxiliary),
            "target": torch.from_numpy(example.target),
            "target_mask": torch.from_numpy(example.target_mask),
            "observability_target": torch.from_numpy(example.observability_target),
            "geometry": torch.from_numpy(example.geometry),
            "link_geometry": torch.from_numpy(example.link_geometry),
            "link_mask": torch.from_numpy(example.link_mask),
            "subject_id": example.subject_id,
            "session_id": example.session_id,
            "environment_id": example.environment_id,
            "posture": example.posture or "unknown",
            "condition": example.condition or "unknown",
            "hardware_domain": example.hardware_domain,
        }


def load_windows_from_root(
    root: Union[str, Path],
    *,
    sample_rate_hz: float,
    window_seconds: float,
    stride_seconds: float,
    target_key: str,
    max_gap_s: float = 0.25,
) -> list[WindowExample]:
    examples: list[WindowExample] = []
    for manifest_path in iter_manifests(root):
        examples.extend(
            session_windows(
                load_session(manifest_path),
                sample_rate_hz=sample_rate_hz,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
                target_key=target_key,
                max_gap_s=max_gap_s,
            )
        )
    if not examples:
        raise ValueError(f"No usable session windows found below {root}")
    return _pad_examples(examples)
