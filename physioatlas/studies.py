from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import signal
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from .io import LoadedSession, SignalStream, dump_json, iter_manifests, load_session
from .metrics import masked_correlation
from .physiology import estimate_delay
from .schema import Modality
from .utils import make_json_safe


class PriorityStudy(str, Enum):
    RESPIRATION_SLEEP = "respiration_sleep"
    CARDIAC_MECHANICS = "cardiac_mechanics"
    PULSE_PROPAGATION = "pulse_propagation"
    MOBILITY_GAIT = "mobility_gait"
    ORGAN_CORRELATED_MOTION = "organ_correlated_motion"


class StudyProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.study-protocol.v1"
    study_id: str
    target: PriorityStudy
    dataset_root: str
    output_dir: str
    sample_rate_hz: float = Field(default=20.0, gt=0)
    minimum_sessions: int = Field(default=2, ge=1)
    require_reference: bool = True
    environment_holdout: bool = False
    research_only: bool = True
    notes: str = ""


STUDY_CATALOG: dict[PriorityStudy, dict[str, Any]] = {
    PriorityStudy.RESPIRATION_SLEEP: {
        "primary_outputs": ["respiratory_rate_bpm", "waveform_correlation", "pause_fraction"],
        "reference": "respiratory belt or polysomnography respiratory channels",
        "claim_boundary": "Respiratory mechanics research; not sleep-apnea diagnosis.",
    },
    PriorityStudy.CARDIAC_MECHANICS: {
        "primary_outputs": ["heart_rate_bpm", "waveform_correlation", "mechanical_timing_error_s"],
        "reference": "ECG and preferably seismocardiography or mmWave",
        "claim_boundary": "Mechanical cardiac timing research; not arrhythmia diagnosis.",
    },
    PriorityStudy.PULSE_PROPAGATION: {
        "primary_outputs": ["ecg_to_ppg_delay_s", "delay_correlation", "delay_repeatability_s"],
        "reference": "synchronized ECG and multi-site PPG",
        "claim_boundary": "Pulse-arrival timing research; not blood-pressure measurement.",
    },
    PriorityStudy.MOBILITY_GAIT: {
        "primary_outputs": ["motion_index", "cadence_per_min", "transition_count"],
        "reference": "pose/depth or instrumented gait reference",
        "claim_boundary": "Mobility research; not fall-risk or frailty diagnosis.",
    },
    PriorityStudy.ORGAN_CORRELATED_MOTION: {
        "primary_outputs": ["held_out_correlation", "shift_null_correlation", "mae"],
        "reference": "synchronized ultrasound displacement",
        "claim_boundary": "Regional motion correlation only; no organ image or pathology claim.",
    },
}


def list_priority_studies() -> list[dict[str, Any]]:
    return [
        {"target": target.value, **metadata, "research_only": True}
        for target, metadata in STUDY_CATALOG.items()
    ]


def load_study_protocol(path: Union[str, Path]) -> StudyProtocol:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8")) if source.suffix == ".json" else __import__("yaml").safe_load(source.read_text(encoding="utf-8"))
    return StudyProtocol.model_validate(payload)


def _stream_by_modality(session: LoadedSession, modalities: set[Modality]) -> Optional[SignalStream]:
    for key, stream in session.streams.items():
        if Modality(key.split(":", 1)[0]) in modalities:
            return stream
    return None


def _resample(stream: SignalStream, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(stream.timestamps_s, dtype=np.float64)
    values = np.asarray(stream.values, dtype=np.float64).reshape(timestamps.size, -1)
    start, stop = float(timestamps[0]), float(timestamps[-1])
    grid = np.arange(start, stop, 1.0 / sample_rate_hz)
    aligned = np.column_stack([np.interp(grid, timestamps, values[:, index]) for index in range(values.shape[1])])
    return grid, aligned


def _standardize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (values - mean) / std


def _rf_component(stream: SignalStream, sample_rate_hz: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    timestamps, values = _resample(stream, sample_rate_hz)
    values = _standardize(values)
    frequencies, power = signal.welch(values, fs=sample_rate_hz, axis=0, nperseg=min(len(timestamps), max(16, int(sample_rate_hz * 10))))
    selected = (frequencies >= band[0]) & (frequencies <= band[1])
    band_power = np.trapezoid(power[selected], frequencies[selected], axis=0) if np.any(selected) else np.zeros(values.shape[1])
    channel = int(np.argmax(band_power))
    sos = signal.butter(3, band, btype="bandpass", fs=sample_rate_hz, output="sos")
    filtered = signal.sosfiltfilt(sos, values[:, channel]) if values.shape[0] > 24 else signal.sosfilt(sos, values[:, channel])
    return timestamps, filtered


def _reference_component(stream: SignalStream, sample_rate_hz: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    timestamps, values = _resample(stream, sample_rate_hz)
    signal_values = _standardize(values[:, :1])[:, 0]
    sos = signal.butter(3, band, btype="bandpass", fs=sample_rate_hz, output="sos")
    filtered = signal.sosfiltfilt(sos, signal_values) if signal_values.size > 24 else signal.sosfilt(sos, signal_values)
    return timestamps, filtered


def _dominant_rate(values: np.ndarray, sample_rate_hz: float, band: tuple[float, float]) -> tuple[float, float]:
    frequencies, power = signal.welch(values, fs=sample_rate_hz, nperseg=min(values.size, max(16, int(sample_rate_hz * 12))))
    selected = (frequencies >= band[0]) & (frequencies <= band[1])
    if not np.any(selected):
        return float("nan"), 0.0
    spectrum = power[selected]
    index = int(np.argmax(spectrum))
    frequency = frequencies[selected][index]
    confidence = float(spectrum[index] / max(np.sum(spectrum), 1e-12))
    return float(frequency * 60.0), float(np.clip(confidence * 4.0, 0.0, 1.0))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    size = min(left.size, right.size)
    if size < 4 or np.std(left[:size]) < 1e-10 or np.std(right[:size]) < 1e-10:
        return float("nan")
    return float(np.corrcoef(left[:size], right[:size])[0, 1])


def _shift_null_correlation(prediction: np.ndarray, reference: np.ndarray) -> float:
    size = min(prediction.size, reference.size)
    if size < 8:
        return float("nan")
    shift = max(2, size // 3)
    shifted = np.roll(reference[:size], shift)
    return max(_correlation(prediction[:size], shifted), _correlation(-prediction[:size], shifted))


def analyze_respiration_session(session: LoadedSession, *, sample_rate_hz: float = 20.0) -> dict[str, Any]:
    rf = _stream_by_modality(session, {Modality.WIFI_CSI, Modality.MMWAVE})
    reference = _stream_by_modality(session, {Modality.RESP_BELT})
    if rf is None:
        raise ValueError("Respiration study requires an RF stream")
    _, rf_waveform = _rf_component(rf, sample_rate_hz, (0.08, 0.60))
    rf_rate, rate_confidence = _dominant_rate(rf_waveform, sample_rate_hz, (0.08, 0.60))
    result: dict[str, Any] = {
        "session_id": session.manifest.session_id,
        "subject_id": session.manifest.subject_id,
        "environment_id": session.manifest.environment_id,
        "respiratory_rate_bpm": rf_rate,
        "rate_confidence": rate_confidence,
        "pause_fraction": float(np.mean(np.abs(rf_waveform) < 0.15 * max(np.std(rf_waveform), 1e-8))),
        "reference_available": reference is not None,
    }
    if reference is not None:
        _, reference_waveform = _reference_component(reference, sample_rate_hz, (0.08, 0.60))
        reference_rate, _ = _dominant_rate(reference_waveform, sample_rate_hz, (0.08, 0.60))
        waveform_correlation = max(_correlation(rf_waveform, reference_waveform), _correlation(-rf_waveform, reference_waveform))
        shift_null = _shift_null_correlation(rf_waveform, reference_waveform)
        result.update(
            {
                "reference_rate_bpm": reference_rate,
                "rate_absolute_error_bpm": abs(rf_rate - reference_rate),
                "waveform_correlation": waveform_correlation,
                "shift_null_correlation": shift_null,
                "correlation_above_shift_null": waveform_correlation - shift_null,
            }
        )
    return make_json_safe(result)


def analyze_cardiac_session(session: LoadedSession, *, sample_rate_hz: float = 20.0) -> dict[str, Any]:
    rf = _stream_by_modality(session, {Modality.MMWAVE, Modality.WIFI_CSI})
    reference = _stream_by_modality(session, {Modality.ECG, Modality.PPG})
    if rf is None:
        raise ValueError("Cardiac study requires an RF stream")
    _, rf_waveform = _rf_component(rf, sample_rate_hz, (0.70, min(3.0, 0.45 * sample_rate_hz)))
    rf_rate, rate_confidence = _dominant_rate(rf_waveform, sample_rate_hz, (0.70, min(3.0, 0.45 * sample_rate_hz)))
    result: dict[str, Any] = {
        "session_id": session.manifest.session_id,
        "subject_id": session.manifest.subject_id,
        "environment_id": session.manifest.environment_id,
        "heart_rate_bpm": rf_rate,
        "rate_confidence": rate_confidence,
        "reference_available": reference is not None,
    }
    if reference is not None:
        _, reference_waveform = _reference_component(reference, sample_rate_hz, (0.70, min(3.0, 0.45 * sample_rate_hz)))
        reference_rate, _ = _dominant_rate(reference_waveform, sample_rate_hz, (0.70, min(3.0, 0.45 * sample_rate_hz)))
        delay = estimate_delay(reference_waveform, rf_waveform, sample_rate_hz=sample_rate_hz, min_delay_s=-0.5, max_delay_s=0.8)
        waveform_correlation = max(_correlation(rf_waveform, reference_waveform), _correlation(-rf_waveform, reference_waveform))
        shift_null = _shift_null_correlation(rf_waveform, reference_waveform)
        result.update(
            {
                "reference_rate_bpm": reference_rate,
                "rate_absolute_error_bpm": abs(rf_rate - reference_rate),
                "waveform_correlation": waveform_correlation,
                "shift_null_correlation": shift_null,
                "correlation_above_shift_null": waveform_correlation - shift_null,
                "mechanical_timing_delay_s": delay.delay_s,
                "timing_correlation": delay.correlation,
            }
        )
    return make_json_safe(result)


def _event_peak_candidates(waveform: np.ndarray, sample_rate_hz: float) -> list[np.ndarray]:
    values = np.asarray(waveform, dtype=np.float64)
    values = (values - np.median(values)) / max(np.std(values), 1e-8)
    distance = max(1, int(0.30 * sample_rate_hz))
    candidates = []
    for oriented in (values, -values):
        peaks, _ = signal.find_peaks(
            oriented,
            distance=distance,
            prominence=max(0.20, 0.20 * float(np.std(oriented))),
        )
        candidates.append(peaks.astype(np.int64))
    return candidates


def _matched_event_delays(
    source_peaks: np.ndarray,
    target_peaks: np.ndarray,
    *,
    sample_rate_hz: float,
    min_delay_s: float,
    max_delay_s: float,
) -> np.ndarray:
    delays = []
    lower = max(1, int(round(min_delay_s * sample_rate_hz)))
    upper = max(lower, int(round(max_delay_s * sample_rate_hz)))
    target = np.asarray(target_peaks, dtype=np.int64)
    for peak in np.asarray(source_peaks, dtype=np.int64):
        candidates = target[(target >= peak + lower) & (target <= peak + upper)]
        if candidates.size:
            delays.append((int(candidates[0]) - int(peak)) / sample_rate_hz)
    return np.asarray(delays, dtype=np.float64)


def _robust_mad(values: np.ndarray) -> Optional[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return None
    median = float(np.median(array))
    return float(1.4826 * np.median(np.abs(array - median)))


def analyze_pulse_session(session: LoadedSession, *, sample_rate_hz: float = 50.0) -> dict[str, Any]:
    ecg = _stream_by_modality(session, {Modality.ECG})
    ppg = _stream_by_modality(session, {Modality.PPG})
    if ecg is None or ppg is None:
        raise ValueError("Pulse propagation study requires synchronized ECG and PPG")
    _, source = _reference_component(ecg, sample_rate_hz, (0.65, min(4.0, 0.45 * sample_rate_hz)))
    _, target = _reference_component(ppg, sample_rate_hz, (0.65, min(4.0, 0.45 * sample_rate_hz)))
    estimate = estimate_delay(source, target, sample_rate_hz=sample_rate_hz, min_delay_s=0.02, max_delay_s=0.6)

    best_event_solution = None
    for source_peaks in _event_peak_candidates(source, sample_rate_hz):
        for target_peaks in _event_peak_candidates(target, sample_rate_hz):
            candidate_delays = _matched_event_delays(
                source_peaks,
                target_peaks,
                sample_rate_hz=sample_rate_hz,
                min_delay_s=0.02,
                max_delay_s=0.6,
            )
            if candidate_delays.size < 3:
                continue
            candidate_median = float(np.median(candidate_delays))
            candidate_mad = _robust_mad(candidate_delays)
            candidate_fraction = float(candidate_delays.size / max(source_peaks.size, 1))
            # ECG/PPG polarity may differ across devices. Choose the polarity
            # pair whose event delay agrees with the independent full-waveform
            # lag estimate, then prefer coverage and repeatability.
            score = (
                -abs(candidate_median - estimate.delay_s),
                candidate_fraction,
                -(candidate_mad if candidate_mad is not None else np.inf),
            )
            if best_event_solution is None or score > best_event_solution[0]:
                best_event_solution = (score, source_peaks, target_peaks, candidate_delays)
    if best_event_solution is None:
        raise ValueError("Pulse propagation requires at least three matched ECG-to-PPG events")
    _, source_peaks, target_peaks, delays = best_event_solution

    # A time reversal is a weak null for periodic physiology because a sinusoid
    # can still align after lag search. Instead, jitter individual target events
    # while preserving their overall count and recording duration. This tests
    # whether one stable causal delay explains the sequence.
    seed = int.from_bytes(
        hashlib.sha256(session.manifest.session_id.encode("utf-8")).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(seed)
    jitter_samples = np.rint(rng.uniform(-0.22, 0.22, size=target_peaks.size) * sample_rate_hz).astype(np.int64)
    null_target_peaks = np.sort(np.clip(target_peaks + jitter_samples, 0, max(len(target) - 1, 0)))
    null_delays = _matched_event_delays(
        source_peaks,
        null_target_peaks,
        sample_rate_hz=sample_rate_hz,
        min_delay_s=0.02,
        max_delay_s=0.6,
    )
    delay_median = float(np.median(delays))
    delay_mad = _robust_mad(delays)
    null_mad = _robust_mad(null_delays)
    matched_fraction = float(delays.size / max(source_peaks.size, 1))
    null_matched_fraction = float(null_delays.size / max(source_peaks.size, 1))
    repeatability_gain = (
        float(null_mad - delay_mad)
        if delay_mad is not None and null_mad is not None
        else None
    )
    return make_json_safe(
        {
            "session_id": session.manifest.session_id,
            "subject_id": session.manifest.subject_id,
            "environment_id": session.manifest.environment_id,
            "ecg_to_ppg_delay_s": delay_median,
            "cross_correlation_delay_s": estimate.delay_s,
            "delay_correlation": estimate.correlation,
            "source_events": int(source_peaks.size),
            "matched_events": int(delays.size),
            "matched_fraction": matched_fraction,
            "delay_repeatability_mad_s": delay_mad,
            "jitter_null_repeatability_mad_s": null_mad,
            "jitter_null_matched_fraction": null_matched_fraction,
            "repeatability_gain_over_jitter_null_s": repeatability_gain,
            "within_physiological_search_bounds": bool(0.02 <= delay_median <= 0.6),
            "null_definition": "independent deterministic per-event timing jitter",
            "research_only": True,
        }
    )


def analyze_mobility_session(session: LoadedSession, *, sample_rate_hz: float = 20.0) -> dict[str, Any]:
    source = _stream_by_modality(session, {Modality.POSE, Modality.MMWAVE, Modality.WIFI_CSI})
    if source is None:
        raise ValueError("Mobility study requires pose, mmWave, or WiFi CSI")
    timestamps, values = _resample(source, sample_rate_hz)
    values = _standardize(values)
    velocity = np.diff(values, axis=0, prepend=values[[0]]) * sample_rate_hz
    energy = np.sqrt(np.mean(velocity * velocity, axis=1))
    smoothed = signal.savgol_filter(energy, min(len(energy) // 2 * 2 - 1, 21), 2) if len(energy) >= 7 else energy
    peaks, _ = signal.find_peaks(smoothed, distance=max(1, int(0.25 * sample_rate_hz)), prominence=max(np.std(smoothed) * 0.25, 1e-8))
    duration_minutes = max((timestamps[-1] - timestamps[0]) / 60.0, 1e-6)
    activity_threshold = np.median(smoothed) + np.std(smoothed)
    transitions = int(np.sum(np.diff((smoothed > activity_threshold).astype(np.int8)) == 1))
    return make_json_safe(
        {
            "session_id": session.manifest.session_id,
            "subject_id": session.manifest.subject_id,
            "environment_id": session.manifest.environment_id,
            "motion_index": float(np.median(energy)),
            "cadence_per_min": float(len(peaks) / duration_minutes),
            "periodicity_confidence": float(
                np.clip(
                    (np.max(smoothed) - np.median(smoothed))
                    / max(3.0 * np.std(smoothed), 1e-8),
                    0.0,
                    1.0,
                )
            ),
            "transition_count": transitions,
            "active_fraction": float(np.mean(smoothed > activity_threshold)),
            "reference_modality": next(key.split(":", 1)[0] for key, stream in session.streams.items() if stream is source),
            "research_only": True,
        }
    )


def analyze_organ_motion_session(session: LoadedSession, *, sample_rate_hz: float = 20.0) -> dict[str, Any]:
    rf = _stream_by_modality(session, {Modality.WIFI_CSI, Modality.MMWAVE})
    ultrasound = _stream_by_modality(session, {Modality.ULTRASOUND})
    if rf is None or ultrasound is None:
        raise ValueError("Organ-correlated motion study requires RF and ultrasound displacement")
    _, rf_values = _resample(rf, sample_rate_hz)
    _, target_values = _resample(ultrasound, sample_rate_hz)
    size = min(rf_values.shape[0], target_values.shape[0])
    x = _standardize(rf_values[:size])
    y = _standardize(target_values[:size, :1])[:, 0]
    split = max(8, int(size * 0.65))
    if size - split < 4:
        raise ValueError("Organ-motion study needs a longer synchronized recording")
    model = Ridge(alpha=10.0).fit(x[:split], y[:split])
    prediction = model.predict(x[split:])
    correlation = _correlation(prediction, y[split:])
    shift = max(1, int(0.25 * (size - split)))
    shifted_target = np.roll(y[split:], shift)
    null_correlation = _correlation(prediction, shifted_target)
    return make_json_safe(
        {
            "session_id": session.manifest.session_id,
            "subject_id": session.manifest.subject_id,
            "environment_id": session.manifest.environment_id,
            "held_out_correlation": correlation,
            "shift_null_correlation": null_correlation,
            "correlation_above_shift_null": correlation - null_correlation,
            "mae": float(mean_absolute_error(y[split:], prediction)),
            "train_samples": split,
            "held_out_samples": size - split,
            "claim_boundary": STUDY_CATALOG[PriorityStudy.ORGAN_CORRELATED_MOTION]["claim_boundary"],
            "research_only": True,
        }
    )


_ANALYZERS = {
    PriorityStudy.RESPIRATION_SLEEP: analyze_respiration_session,
    PriorityStudy.CARDIAC_MECHANICS: analyze_cardiac_session,
    PriorityStudy.PULSE_PROPAGATION: analyze_pulse_session,
    PriorityStudy.MOBILITY_GAIT: analyze_mobility_session,
    PriorityStudy.ORGAN_CORRELATED_MOTION: analyze_organ_motion_session,
}


def _aggregate_numeric(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    output = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=np.float64)
        if values.size:
            output[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "n": int(values.size),
            }
    return output


def _scientific_gate(target: PriorityStudy, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_evaluable", "passed": False, "reasons": ["no successful sessions"]}
    def median(key: str) -> Optional[float]:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(row[key])]
        return float(np.median(values)) if values else None
    reasons = []
    passed: Optional[bool]
    if target == PriorityStudy.RESPIRATION_SLEEP:
        error = median("rate_absolute_error_bpm")
        gain = median("correlation_above_shift_null")
        passed = error is not None and gain is not None and error <= 3.0 and gain > 0.05
        reasons = [f"median rate error={error}", f"median correlation above shift null={gain}"]
    elif target == PriorityStudy.CARDIAC_MECHANICS:
        error = median("rate_absolute_error_bpm")
        gain = median("correlation_above_shift_null")
        passed = error is not None and gain is not None and error <= 8.0 and gain > 0.03
        reasons = [f"median rate error={error}", f"median correlation above shift null={gain}"]
    elif target == PriorityStudy.PULSE_PROPAGATION:
        gain = median("repeatability_gain_over_jitter_null_s")
        repeatability = median("delay_repeatability_mad_s")
        matched = median("matched_fraction")
        delays = [row.get("within_physiological_search_bounds") for row in rows]
        passed = (
            gain is not None
            and repeatability is not None
            and matched is not None
            and gain > 0.02
            and repeatability <= 0.05
            and matched >= 0.70
            and all(value is True for value in delays)
        )
        reasons = [
            f"median repeatability gain over jitter null={gain}",
            f"median delay MAD={repeatability}",
            f"median matched fraction={matched}",
            "all median delays within declared search bounds",
        ]
    elif target == PriorityStudy.ORGAN_CORRELATED_MOTION:
        gain = median("correlation_above_shift_null")
        passed = gain is not None and gain > 0.05
        reasons = [f"median held-out correlation above shifted null={gain}"]
    else:
        passed = None
        reasons = ["mobility requires an external gait/motion reference before a scientific gate can be evaluated"]
    return {
        "status": "passed" if passed is True else ("failed" if passed is False else "not_evaluable"),
        "passed": passed,
        "reasons": reasons,
        "exploratory_thresholds": True,
    }


def run_priority_study(protocol: StudyProtocol) -> dict[str, Any]:
    analyzer = _ANALYZERS[protocol.target]
    rows = []
    failures = []
    for manifest_path in iter_manifests(protocol.dataset_root):
        try:
            rows.append(analyzer(load_session(manifest_path), sample_rate_hz=protocol.sample_rate_hz))
        except Exception as exc:
            failures.append({"manifest": str(manifest_path), "error": str(exc)})
    output = Path(protocol.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = make_json_safe(
        {
            "schema_version": "physioatlas.priority-study-report.v1",
            "study_id": protocol.study_id,
            "target": protocol.target.value,
            "protocol": protocol.model_dump(mode="json"),
            "catalog": STUDY_CATALOG[protocol.target],
            "sessions_analyzed": len(rows),
            "sessions_failed": len(failures),
            "session_results": rows,
            "aggregate": _aggregate_numeric(rows),
            "scientific_gate": _scientific_gate(protocol.target, rows),
            "failures": failures,
            "infrastructure_valid": len(rows) >= protocol.minimum_sessions,
            "valid": len(rows) >= protocol.minimum_sessions,
            "research_only": True,
        }
    )
    dump_json(output / "study-report.json", report)
    return report


def run_all_priority_studies(
    dataset_root: Union[str, Path],
    output_root: Union[str, Path],
    *,
    sample_rate_hz: float = 20.0,
    minimum_sessions: int = 2,
) -> dict[str, Any]:
    reports = []
    for target in PriorityStudy:
        protocol = StudyProtocol(
            study_id=f"priority-{target.value}",
            target=target,
            dataset_root=str(Path(dataset_root).resolve()),
            output_dir=str(Path(output_root).resolve() / target.value),
            sample_rate_hz=50.0 if target == PriorityStudy.PULSE_PROPAGATION else sample_rate_hz,
            minimum_sessions=minimum_sessions,
        )
        reports.append(run_priority_study(protocol))
    summary = {
        "schema_version": "physioatlas.priority-study-suite.v1",
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "studies": reports,
        "valid": all(report["valid"] for report in reports),
        "research_only": True,
    }
    dump_json(Path(output_root) / "priority-study-suite.json", summary)
    return summary
