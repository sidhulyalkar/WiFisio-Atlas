from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from scipy import signal
from scipy.optimize import nnls

from .household import extract_identity_embedding
from .household_csi_schema import (
    CsiProcessingConfig,
    CsiProcessingReport,
    HouseholdCsiCalibration,
    LinkCsiCalibration,
    ZoneCsiCalibration,
)
from .household_csi_input import (
    LinkTrace,
    link_empty_calibration,
    load_raw_csi_jsonl,
    trace_for_link,
)
from .io import SignalStream, dump_json
from .utils import make_json_safe


def _condition_number(matrix: np.ndarray) -> Optional[float]:
    if min(matrix.shape) < 2:
        return None
    value = float(np.linalg.cond(matrix))
    return value if np.isfinite(value) else None


def _apply_sequence_synchronization(
    links: list[LinkCsiCalibration],
    capture_path: Union[str, Path],
) -> tuple[list[LinkCsiCalibration], bool]:
    capture, _ = load_raw_csi_jsonl(capture_path)
    reference_id = links[0].link_id
    if reference_id not in capture:
        raise ValueError(f"Synchronization capture is missing reference link {reference_id!r}")
    reference = {
        frame.sequence: frame.timestamp_s
        for frame in capture[reference_id]
        if frame.sequence is not None
    }
    updated = []
    verified = bool(reference)
    for link in links:
        frames = capture.get(link.link_id, [])
        candidate = {
            frame.sequence: frame.timestamp_s for frame in frames if frame.sequence is not None
        }
        shared = sorted(set(reference) & set(candidate))
        if len(shared) < 10:
            updated.append(link)
            verified = False
            continue
        offsets = np.asarray([candidate[item] - reference[item] for item in shared])
        median = float(np.median(offsets))
        jitter = float(np.median(np.abs(offsets - median)) * 1.4826)
        updated.append(
            link.model_copy(
                update={
                    "clock_offset_s": median,
                    "clock_offset_mad_s": jitter,
                    "synchronization_pairs": len(shared),
                }
            )
        )
    return updated, verified


def build_household_csi_calibration(
    empty_capture: Union[str, Path],
    zone_captures: dict[str, tuple[list[float], Union[str, Path]]],
    output_path: Union[str, Path],
    *,
    environment_id: str,
    sample_rate_hz: float = 20.0,
    minimum_links: int = 3,
    synchronization_capture: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    empty, empty_report = load_raw_csi_jsonl(empty_capture)
    if len(empty) < minimum_links:
        raise ValueError(f"At least {minimum_links} empty-room links are required")
    link_order = sorted(empty)
    links = [link_empty_calibration(link_id, empty[link_id]) for link_id in link_order]
    synchronization_verified = False
    if synchronization_capture is not None:
        links, synchronization_verified = _apply_sequence_synchronization(
            links, synchronization_capture
        )
    link_calibration = {item.link_id: item for item in links}
    zones = []
    for zone_id, (center_m, capture_path) in sorted(zone_captures.items()):
        capture, _ = load_raw_csi_jsonl(capture_path)
        missing = set(link_order) - set(capture)
        if missing:
            raise ValueError(f"Zone {zone_id!r} is missing links: {sorted(missing)}")
        signature = []
        samples = []
        for link_id in link_order:
            trace = trace_for_link(capture[link_id], link_calibration[link_id])
            samples.append(trace.energy.size)
            baseline = link_calibration[link_id].empty_energy_median
            signature.append(max(float(np.median(trace.energy)) - baseline, 1e-4))
        vector = np.asarray(signature)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        zones.append(
            ZoneCsiCalibration(
                zone_id=zone_id,
                center_m=center_m,
                link_signature=vector.tolist(),
                samples=min(samples),
                source_capture=str(Path(capture_path).resolve()),
            )
        )
    signature_matrix = np.asarray([item.link_signature for item in zones], dtype=float).T
    calibration = HouseholdCsiCalibration(
        environment_id=environment_id,
        sample_rate_hz=sample_rate_hz,
        link_order=link_order,
        links=links,
        zones=zones,
        minimum_links=minimum_links,
        synchronization_verified=synchronization_verified,
        zone_signature_condition_number=_condition_number(signature_matrix),
        calibration_notes=[
            "Each zone signature requires a single consented stationary participant.",
            "Each link remains on one fixed band/channel; simultaneous receivers enable multi-band fusion.",
            (
                "Receiver clocks were aligned from shared packet sequence IDs."
                if synchronization_verified
                else "Receiver clock synchronization is unverified; provide a shared-packet sync capture."
            ),
            "Zone positions are calibrated anchors, not continuous RF imaging.",
            "Recalibrate after moving the router, receivers, or major furniture.",
        ],
    )
    dump_json(output_path, calibration.model_dump(mode="json"))
    return make_json_safe(
        {
            "schema_version": calibration.schema_version,
            "output": str(Path(output_path).resolve()),
            "environment_id": environment_id,
            "links": link_order,
            "zones": [item.zone_id for item in zones],
            "condition_number": calibration.zone_signature_condition_number,
            "empty_capture": empty_report,
            "synchronization_verified": synchronization_verified,
            "research_only": True,
        }
    )


def load_household_csi_calibration(path: Union[str, Path]) -> HouseholdCsiCalibration:
    return HouseholdCsiCalibration.model_validate(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _regularize_trace(
    trace: LinkTrace,
    timeline: np.ndarray,
    maximum_gap_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if trace.timestamps_s.size < 2:
        shape = timeline.shape
        return np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape)
    indices = np.searchsorted(trace.timestamps_s, timeline)
    indices = np.clip(indices, 1, trace.timestamps_s.size - 1)
    left = indices - 1
    right = indices
    choose_right = (
        np.abs(trace.timestamps_s[right] - timeline)
        < np.abs(timeline - trace.timestamps_s[left])
    )
    nearest = np.where(choose_right, right, left)
    distance = np.abs(trace.timestamps_s[nearest] - timeline)
    valid = distance <= maximum_gap_s
    signed = np.where(valid, trace.signed_signal[nearest], np.nan)
    energy = np.where(valid, trace.energy[nearest], np.nan)
    quality = np.where(valid, trace.quality[nearest], 0.0)
    return signed, energy, quality


def _dominant_rate(
    values: np.ndarray,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
) -> tuple[Optional[float], float]:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if finite.mean() < 0.8 or finite.sum() < max(16, int(sample_rate_hz * 6)):
        return None, 0.0
    if not np.all(finite):
        array = np.interp(np.arange(array.size), np.flatnonzero(finite), array[finite])
    array = signal.detrend(array)
    frequencies, spectrum = signal.welch(
        array,
        fs=sample_rate_hz,
        nperseg=min(array.size, max(16, int(sample_rate_hz * 30))),
    )
    selected = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    if not np.any(selected):
        return None, 0.0
    band = spectrum[selected]
    index = int(np.argmax(band))
    peak = float(band[index])
    total = float(np.sum(band))
    confidence = float(np.clip(4.0 * peak / max(total, 1e-12), 0.0, 1.0))
    if peak <= 1e-10 or confidence < 0.25:
        return None, confidence
    return float(frequencies[selected][index] * 60.0), confidence


def _identity_embedding(
    timestamps_s: np.ndarray,
    component: np.ndarray,
    signature: np.ndarray,
    sample_rate_hz: float,
) -> Optional[list[float]]:
    if component.size < max(16, int(sample_rate_hz * 8)):
        return None
    values = component[:, None] * signature[None, :]
    stream = SignalStream(timestamps_s=timestamps_s, values=values.astype(np.float32))
    return extract_identity_embedding(stream, sample_rate_hz=sample_rate_hz).tolist()


def process_mixed_household_csi(
    input_path: Union[str, Path],
    calibration_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    config: Optional[CsiProcessingConfig] = None,
) -> dict[str, Any]:
    config = config or CsiProcessingConfig()
    calibration = load_household_csi_calibration(calibration_path)
    grouped, load_report = load_raw_csi_jsonl(input_path)
    observed_links = [item for item in calibration.link_order if item in grouped]
    if len(observed_links) < calibration.minimum_links:
        raise ValueError(
            f"Observed {len(observed_links)} calibrated links; "
            f"{calibration.minimum_links} are required"
        )
    link_map = {item.link_id: item for item in calibration.links}
    observed_bands = sorted(
        {
            link_map[item].frequency_band
            for item in observed_links
            if link_map[item].frequency_band != "unknown"
        }
    )
    missing_bands = set(config.require_frequency_bands) - set(observed_bands)
    if missing_bands:
        raise ValueError(
            "Required frequency bands are absent from the synchronized capture: "
            f"{sorted(missing_bands)}"
        )
    traces = {
        link_id: trace_for_link(grouped[link_id], link_map[link_id])
        for link_id in observed_links
    }
    traces = {
        link_id: LinkTrace(
            timestamps_s=trace.timestamps_s - link_map[link_id].clock_offset_s,
            signed_signal=trace.signed_signal,
            energy=trace.energy,
            quality=trace.quality,
        )
        for link_id, trace in traces.items()
    }
    start = max(trace.timestamps_s[0] for trace in traces.values())
    end = min(trace.timestamps_s[-1] for trace in traces.values())
    if end - start < config.vital_window_s:
        raise ValueError("Mixed CSI capture is shorter than the configured vital window")
    timeline = np.arange(start, end + 0.5 / calibration.sample_rate_hz, 1.0 / calibration.sample_rate_hz)
    signed_columns, energy_columns, quality_columns = [], [], []
    for link_id in observed_links:
        signed, energy, quality = _regularize_trace(
            traces[link_id], timeline, config.maximum_gap_s
        )
        signed_columns.append(signed)
        energy_columns.append(energy)
        quality_columns.append(quality)
    signed_matrix = np.stack(signed_columns, axis=1)
    energy_matrix = np.stack(energy_columns, axis=1)
    quality_matrix = np.stack(quality_columns, axis=1)
    link_indices = [calibration.link_order.index(item) for item in observed_links]
    zone_matrix = np.asarray(
        [[zone.link_signature[index] for zone in calibration.zones] for index in link_indices],
        dtype=float,
    )
    empty_floor = np.asarray(
        [link_map[item].empty_energy_median for item in observed_links], dtype=float
    )
    localization_samples = max(2, int(config.localization_window_s * calibration.sample_rate_hz))
    vital_samples = max(16, int(config.vital_window_s * calibration.sample_rate_hz))
    stride_samples = max(1, int(config.stride_s * calibration.sample_rate_hz))
    observations: list[dict[str, Any]] = []
    windows = 0
    not_observable = 0
    for stop in range(vital_samples, timeline.size + 1, stride_samples):
        windows += 1
        local = energy_matrix[max(0, stop - localization_samples) : stop]
        energy_vector = np.nanmedian(local, axis=0) - empty_floor
        energy_vector = np.maximum(np.nan_to_num(energy_vector, nan=0.0), 0.0)
        activation, residual = nnls(zone_matrix, energy_vector)
        active = np.flatnonzero(
            activation >= calibration.occupancy_activation_threshold
        )
        if active.size > calibration.maximum_occupants:
            active = active[np.argsort(activation[active])[-calibration.maximum_occupants :]]
        availability = float(np.mean(np.isfinite(signed_matrix[stop - vital_samples : stop])))
        hardware_quality = float(
            np.mean(quality_matrix[stop - vital_samples : stop])
        )
        location_fit = float(
            np.exp(-residual / max(np.linalg.norm(energy_vector), 1e-6))
        )
        if active.size == 0 or availability < config.minimum_observability:
            not_observable += 1
            continue
        active_matrix = zone_matrix[:, active]
        separation_condition = (
            float(np.linalg.cond(active_matrix)) if active.size > 1 else 1.0
        )
        if not np.isfinite(separation_condition):
            separation_condition = float("inf")
        separation_quality = float(
            np.clip(
                1.0
                - max(0.0, separation_condition - 1.0)
                / max(config.maximum_separation_condition - 1.0, 1e-6),
                0.0,
                1.0,
            )
        )
        window_signal = signed_matrix[stop - vital_samples : stop]
        finite_counts = np.sum(np.isfinite(window_signal), axis=1)
        valid_rows = finite_counts >= calibration.minimum_links
        filled = np.nan_to_num(window_signal, nan=0.0)
        gram = active_matrix.T @ active_matrix + config.ridge * np.eye(active.size)
        demixed = np.full((vital_samples, active.size), np.nan)
        demixed[valid_rows] = (
            np.linalg.solve(gram, active_matrix.T @ filled[valid_rows].T).T
        )
        for target_column, zone_index in enumerate(active.tolist()):
            component = demixed[:, target_column]
            observability = float(
                np.clip(
                    availability
                    * hardware_quality
                    * location_fit
                    * separation_quality,
                    0.0,
                    1.0,
                )
            )
            respiration, respiration_confidence = _dominant_rate(
                component, calibration.sample_rate_hz, (0.08, 0.60)
            )
            allow_heart = (
                config.enable_heart_rate
                and (active.size == 1 or config.allow_multi_person_heart_rate)
            ) and separation_condition <= config.maximum_separation_condition
            heart, heart_confidence = (
                _dominant_rate(component, calibration.sample_rate_hz, (0.70, 2.50))
                if allow_heart and observability >= config.minimum_observability
                else (None, 0.0)
            )
            if heart is not None and respiration is not None:
                harmonics = [respiration * multiplier for multiplier in range(2, 9)]
                if min(abs(heart - harmonic) for harmonic in harmonics) < 3.0:
                    heart, heart_confidence = None, 0.0
            high_frequency = _dominant_rate(
                component, calibration.sample_rate_hz, (0.60, 3.00)
            )[1]
            zone = calibration.zones[zone_index]
            embedding = _identity_embedding(
                timeline[stop - vital_samples : stop]
                - timeline[stop - vital_samples],
                component,
                active_matrix[:, target_column],
                calibration.sample_rate_hz,
            )
            respiration_status = (
                "estimated"
                if respiration is not None
                and observability >= config.minimum_observability
                else "not_observable"
            )
            heart_status = (
                "estimated"
                if heart is not None and observability >= config.minimum_observability
                else "experimental_disabled"
                if not config.enable_heart_rate
                else "not_observable"
            )
            observation = {
                "schema_version": "physioatlas.household-observation.v2",
                "timestamp_s": float(timeline[stop - 1]),
                "position_m": zone.center_m,
                "position_covariance_m2": [
                    calibration.position_uncertainty_m**2,
                    0.0,
                    0.0,
                    calibration.position_uncertainty_m**2,
                ],
                "zone_id": zone.zone_id,
                "embeddings": {} if embedding is None else {"wifi_csi": embedding},
                "identity_embedding_versions": {
                    "wifi_csi": "physioatlas.household-csi-identity.v1"
                },
                "signal_quality": hardware_quality,
                "observability": observability,
                "respiratory_rate_bpm": (
                    respiration if respiration_status == "estimated" else None
                ),
                "heart_rate_bpm": heart if heart_status == "estimated" else None,
                "motion_index": high_frequency,
                "measurement_status": {
                    "position": "calibrated_zone_estimate",
                    "respiration": respiration_status,
                    "heart_rate": heart_status,
                    "identity": "embedding_only_not_identified",
                },
                "confidence": {
                    "localization_fit": location_fit,
                    "separation_quality": separation_quality,
                    "respiration": respiration_confidence,
                    "heart_rate": heart_confidence,
                },
                "provenance": {
                    "source_kind": "calibrated_multilink_wifi_csi",
                    "calibration_schema": calibration.schema_version,
                    "environment_id": calibration.environment_id,
                    "link_ids": observed_links,
                    "frequency_bands": observed_bands,
                    "multi_band_fusion": len(observed_bands) > 1,
                    "synchronization_verified": calibration.synchronization_verified,
                    "clock_offsets_s": {
                        item: link_map[item].clock_offset_s for item in observed_links
                    },
                    "window_start_s": float(timeline[stop - vital_samples]),
                    "window_end_s": float(timeline[stop - 1]),
                    "separation_condition_number": separation_condition,
                    "claim_class": "predicted",
                    "research_only": True,
                },
            }
            observations.append(make_json_safe(observation))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation, sort_keys=True, allow_nan=False) + "\n")
    report = CsiProcessingReport(
        input_path=str(Path(input_path).resolve()),
        calibration_path=str(Path(calibration_path).resolve()),
        output_path=str(destination.resolve()),
        accepted_frames=load_report["accepted_frames"],
        rejected_frames=load_report["rejected_frames"],
        links_observed=observed_links,
        frequency_bands_observed=observed_bands,
        synchronization_verified=calibration.synchronization_verified,
        windows_evaluated=windows,
        observations_emitted=len(observations),
        not_observable_windows=not_observable,
    )
    return report.model_dump(mode="json")
