from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import signal

from .io import LoadedSession, SignalStream, dump_json, iter_manifests, load_session
from .privacy import load_consent, validate_consent
from .schema import Modality
from .utils import make_json_safe

IDENTITY_MODALITIES = (Modality.WIFI_CSI, Modality.MMWAVE)


class BiologicalSummary(BaseModel):
    """Research-only baseline summary, never a diagnostic record."""

    model_config = ConfigDict(extra="forbid")

    respiratory_rate_bpm: Optional[float] = None
    respiratory_rate_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    respiratory_rate_std_bpm: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    heart_rate_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    heart_rate_std_bpm: Optional[float] = None
    motion_index: Optional[float] = None
    motion_index_std: Optional[float] = None
    sessions: int = Field(default=0, ge=0)
    research_only: bool = True


class ModalityIdentityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: Modality
    centroid: list[float]
    within_distance_q95: float = Field(ge=0.0)
    acceptance_threshold: float = Field(gt=0.0)
    samples: int = Field(gt=0)
    embedding_version: str = "physioatlas.rf-identity.v1"


class HouseholdMemberProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    member_id: str = Field(min_length=1)
    display_alias: str = Field(min_length=1)
    consent_id: str
    enrolled_at_utc: str
    profiles: list[ModalityIdentityProfile]
    biological_summary: BiologicalSummary
    minimum_modalities: int = Field(default=1, ge=1)
    active: bool = True


class HouseholdRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.household-registry.v1"
    household_id: str = Field(min_length=1)
    created_at_utc: str
    members: list[HouseholdMemberProfile] = Field(default_factory=list)
    open_set_margin: float = Field(default=0.20, ge=0.0)
    require_consent: bool = True
    local_only: bool = True
    raw_biometrics_stored: bool = False
    derived_identity_templates_stored: bool = True
    enrollment_session_ids: list[str] = Field(default_factory=list)
    research_only: bool = True

    @model_validator(mode="after")
    def unique_members(self) -> "HouseholdRegistry":
        ids = [item.member_id for item in self.members]
        if len(ids) != len(set(ids)):
            raise ValueError("Household member IDs must be unique")
        return self


@dataclass(frozen=True)
class IdentityDecision:
    member_id: Optional[str]
    display_alias: Optional[str]
    confidence: float
    accepted: bool
    reason: str
    modality_scores: dict[str, float]
    distance: Optional[float]
    second_best_margin: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(
            {
                "member_id": self.member_id,
                "display_alias": self.display_alias,
                "confidence": self.confidence,
                "accepted": self.accepted,
                "reason": self.reason,
                "modality_scores": self.modality_scores,
                "distance": self.distance,
                "second_best_margin": self.second_best_margin,
                "research_only": True,
            }
        )


def _safe_standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    median = np.median(array, axis=0, keepdims=True)
    scale = np.median(np.abs(array - median), axis=0, keepdims=True) * 1.4826
    scale[scale < 1e-6] = np.std(array, axis=0, keepdims=True)[scale < 1e-6]
    scale[scale < 1e-6] = 1.0
    return np.clip((array - median) / scale, -8.0, 8.0)


def _band_power(values: np.ndarray, sample_rate_hz: float, low: float, high: float) -> np.ndarray:
    if values.shape[0] < 8 or sample_rate_hz <= 0:
        return np.zeros(values.shape[1], dtype=np.float64)
    frequencies, power = signal.welch(
        values,
        fs=sample_rate_hz,
        axis=0,
        nperseg=min(values.shape[0], max(8, int(sample_rate_hz * 8))),
    )
    selected = (frequencies >= low) & (frequencies <= min(high, sample_rate_hz / 2.0))
    if not np.any(selected):
        return np.zeros(values.shape[1], dtype=np.float64)
    return np.trapezoid(power[selected], frequencies[selected], axis=0)


def extract_identity_embedding(
    stream: SignalStream,
    *,
    sample_rate_hz: Optional[float] = None,
    max_channels: int = 24,
) -> np.ndarray:
    """Create a fixed-size, session-level RF signature.

    The signature describes stable channel statistics and spectral structure. It
    is intentionally unsuitable for unconstrained identification: deployment
    must use the consent-gated open-set classifier below and retain unknowns.
    """
    stream.validate()
    values = _safe_standardize(np.asarray(stream.values).reshape(stream.timestamps_s.size, -1))
    if sample_rate_hz is None:
        differences = np.diff(stream.timestamps_s)
        sample_rate_hz = 1.0 / float(np.median(differences)) if differences.size else 1.0
    variances = np.var(values, axis=0)
    indices = np.argsort(variances)[::-1][: min(max_channels, values.shape[1])]
    selected = values[:, indices]

    channel_stats = np.stack(
        [
            np.mean(selected, axis=0),
            np.std(selected, axis=0),
            np.quantile(selected, 0.10, axis=0),
            np.quantile(selected, 0.25, axis=0),
            np.quantile(selected, 0.75, axis=0),
            np.quantile(selected, 0.90, axis=0),
            _band_power(selected, sample_rate_hz, 0.08, 0.55),
            _band_power(selected, sample_rate_hz, 0.70, 2.50),
            _band_power(selected, sample_rate_hz, 2.50, min(8.0, sample_rate_hz / 2.0)),
        ],
        axis=0,
    )
    pooled: list[float] = []
    for row in channel_stats:
        pooled.extend(
            [
                float(np.mean(row)),
                float(np.std(row)),
                float(np.quantile(row, 0.10)),
                float(np.quantile(row, 0.50)),
                float(np.quantile(row, 0.90)),
            ]
        )

    if selected.shape[1] > 1 and selected.shape[0] > 2:
        covariance = np.cov(selected, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance))[::-1]
    else:
        eigenvalues = np.asarray([float(np.var(selected))])
    eigenvalues = eigenvalues[:8]
    eigenvalues = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    pooled.extend(np.pad(eigenvalues, (0, 8 - eigenvalues.size)).tolist())

    # Preserve channel-distribution and narrowband structure that robust
    # within-channel standardization intentionally removes. These are normalized
    # summaries, not raw CSI samples.
    raw = np.asarray(stream.values, dtype=np.float64).reshape(stream.timestamps_s.size, -1)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)[:, indices]
    raw_std = np.std(raw, axis=0)
    raw_std /= max(float(np.linalg.norm(raw_std)), 1e-12)
    pooled.extend(np.pad(raw_std[:16], (0, max(0, 16 - raw_std[:16].size))).tolist())
    if selected.shape[0] >= 16:
        frequencies, spectral = signal.welch(
            selected, fs=sample_rate_hz, axis=0,
            nperseg=min(selected.shape[0], max(16, int(sample_rate_hz * 12))),
        )
        average_spectrum = np.mean(spectral, axis=1)
        valid_frequency = frequencies <= min(8.0, sample_rate_hz / 2.0)
        target_frequency = np.linspace(0.05, min(8.0, sample_rate_hz / 2.0), 20)
        sampled = np.interp(target_frequency, frequencies[valid_frequency], average_spectrum[valid_frequency])
        sampled /= max(float(np.linalg.norm(sampled)), 1e-12)
        pooled.extend(sampled.tolist())
    else:
        pooled.extend([0.0] * 20)

    if stream.quality is None:
        pooled.extend([1.0, 0.0])
    else:
        quality = np.asarray(stream.quality, dtype=np.float64).reshape(-1)
        pooled.extend([float(np.mean(quality)), float(np.mean(quality <= 0.0))])
    embedding = np.asarray(pooled, dtype=np.float64)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 1e-12 else embedding


def _dominant_rate(
    stream: SignalStream,
    *,
    low_hz: float,
    high_hz: float,
) -> tuple[Optional[float], float]:
    values = np.asarray(stream.values, dtype=np.float64).reshape(stream.timestamps_s.size, -1)
    values = _safe_standardize(values)
    if values.shape[0] < 16:
        return None, 0.0
    sample_rate_hz = 1.0 / float(np.median(np.diff(stream.timestamps_s)))
    powers = _band_power(values, sample_rate_hz, low_hz, high_hz)
    channel = int(np.argmax(powers))
    frequencies, spectrum = signal.welch(
        values[:, channel],
        fs=sample_rate_hz,
        nperseg=min(values.shape[0], max(16, int(sample_rate_hz * 12))),
    )
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(mask):
        return None, 0.0
    band = spectrum[mask]
    selected_frequencies = frequencies[mask]
    index = int(np.argmax(band))
    peak = float(band[index])
    confidence = peak / max(float(np.sum(band)), 1e-12)
    return float(selected_frequencies[index] * 60.0), float(np.clip(confidence * 4.0, 0.0, 1.0))


def summarize_biology(session: LoadedSession) -> BiologicalSummary:
    heart_stream: Optional[SignalStream] = None
    respiratory_stream: Optional[SignalStream] = None
    rf_stream: Optional[SignalStream] = None
    for key, stream in session.streams.items():
        modality = Modality(key.split(":", 1)[0])
        if modality in {Modality.ECG, Modality.PPG} and heart_stream is None:
            heart_stream = stream
        if modality == Modality.RESP_BELT and respiratory_stream is None:
            respiratory_stream = stream
        if modality in IDENTITY_MODALITIES and rf_stream is None:
            rf_stream = stream
    heart_rate, heart_confidence = _dominant_rate(
        heart_stream or rf_stream,
        low_hz=0.65,
        high_hz=3.0,
    ) if (heart_stream or rf_stream) is not None else (None, 0.0)
    respiratory_rate, respiratory_confidence = _dominant_rate(
        respiratory_stream or rf_stream,
        low_hz=0.08,
        high_hz=0.65,
    ) if (respiratory_stream or rf_stream) is not None else (None, 0.0)
    motion_index = None
    if rf_stream is not None:
        standardized = _safe_standardize(np.asarray(rf_stream.values).reshape(rf_stream.timestamps_s.size, -1))
        motion_index = float(np.median(np.sqrt(np.mean(np.diff(standardized, axis=0) ** 2, axis=1))))
    return BiologicalSummary(
        respiratory_rate_bpm=respiratory_rate,
        respiratory_rate_confidence=respiratory_confidence,
        heart_rate_bpm=heart_rate,
        heart_rate_confidence=heart_confidence,
        motion_index=motion_index,
        sessions=1,
    )


def _session_embeddings(session: LoadedSession) -> dict[Modality, np.ndarray]:
    embeddings: dict[Modality, list[np.ndarray]] = {}
    sensor_rates = {sensor.sensor_id: sensor.sample_rate_hz for sensor in session.geometry.sensors}
    for key, stream in session.streams.items():
        modality_text, sensor_id = key.split(":", 1)
        modality = Modality(modality_text)
        if modality not in IDENTITY_MODALITIES:
            continue
        embeddings.setdefault(modality, []).append(
            extract_identity_embedding(stream, sample_rate_hz=sensor_rates.get(sensor_id))
        )
    return {
        modality: np.mean(np.stack(items), axis=0) / max(np.linalg.norm(np.mean(np.stack(items), axis=0)), 1e-12)
        for modality, items in embeddings.items()
    }


def _average_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(item) for item in values if item is not None and np.isfinite(item)]
    return float(np.mean(finite)) if finite else None


def _std_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(item) for item in values if item is not None and np.isfinite(item)]
    return float(np.std(finite)) if finite else None


def enroll_household(
    dataset_root: Union[str, Path],
    registry_path: Union[str, Path],
    *,
    household_id: str = "household-local",
    aliases: Optional[dict[str, str]] = None,
    require_consent: bool = True,
    minimum_sessions: int = 2,
) -> dict[str, Any]:
    aliases = aliases or {}
    grouped: dict[str, list[LoadedSession]] = {}
    consent_ids: dict[str, str] = {}
    rejected: list[str] = []
    for manifest_path in iter_manifests(dataset_root):
        session = load_session(manifest_path)
        subject_id = session.manifest.subject_id
        if require_consent:
            if not session.manifest.consent_path:
                rejected.append(f"{session.manifest.session_id}: missing consent")
                continue
            policy = load_consent(session.manifest.consent_path)
            report = validate_consent(
                policy,
                subject_id=subject_id,
                modalities=[Modality.WIFI_CSI],
                targets=[],
                use="identity_enrollment",
            )
            if not report["valid"] or not getattr(policy, "allow_identity_enrollment", False):
                rejected.extend(
                    [f"{session.manifest.session_id}: {error}" for error in report["errors"]]
                    or [f"{session.manifest.session_id}: identity enrollment not allowed"]
                )
                continue
            consent_ids[subject_id] = policy.consent_id
        else:
            consent_ids[subject_id] = "consent-not-required"
        grouped.setdefault(subject_id, []).append(session)

    members: list[HouseholdMemberProfile] = []
    for subject_id, sessions in sorted(grouped.items()):
        if len(sessions) < minimum_sessions:
            rejected.append(
                f"{subject_id}: needs at least {minimum_sessions} sessions, found {len(sessions)}"
            )
            continue
        modality_samples: dict[Modality, list[np.ndarray]] = {}
        summaries = []
        for session in sessions:
            for modality, embedding in _session_embeddings(session).items():
                modality_samples.setdefault(modality, []).append(embedding)
            summaries.append(summarize_biology(session))
        modality_profiles: list[ModalityIdentityProfile] = []
        for modality, samples in sorted(modality_samples.items(), key=lambda item: item[0].value):
            matrix = np.stack(samples)
            centroid = np.mean(matrix, axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            distances = 1.0 - np.clip(matrix @ centroid, -1.0, 1.0)
            q95 = float(np.quantile(distances, 0.95)) if distances.size > 1 else 0.02
            threshold = float(np.clip(q95 + 0.08, 0.08, 0.45))
            modality_profiles.append(
                ModalityIdentityProfile(
                    modality=modality,
                    centroid=centroid.tolist(),
                    within_distance_q95=q95,
                    acceptance_threshold=threshold,
                    samples=len(samples),
                )
            )
        if not modality_profiles:
            rejected.append(f"{subject_id}: no WiFi CSI or mmWave streams")
            continue
        members.append(
            HouseholdMemberProfile(
                member_id=subject_id,
                display_alias=aliases.get(subject_id, subject_id),
                consent_id=consent_ids[subject_id],
                enrolled_at_utc=datetime.now(timezone.utc).isoformat(),
                profiles=modality_profiles,
                biological_summary=BiologicalSummary(
                    respiratory_rate_bpm=_average_optional(
                        item.respiratory_rate_bpm for item in summaries
                    ),
                    respiratory_rate_confidence=float(
                        np.mean([item.respiratory_rate_confidence for item in summaries])
                    ),
                    respiratory_rate_std_bpm=_std_optional(
                        item.respiratory_rate_bpm for item in summaries
                    ),
                    heart_rate_bpm=_average_optional(item.heart_rate_bpm for item in summaries),
                    heart_rate_confidence=float(
                        np.mean([item.heart_rate_confidence for item in summaries])
                    ),
                    heart_rate_std_bpm=_std_optional(item.heart_rate_bpm for item in summaries),
                    motion_index=_average_optional(item.motion_index for item in summaries),
                    motion_index_std=_std_optional(item.motion_index for item in summaries),
                    sessions=len(summaries),
                ),
            )
        )
    registry = HouseholdRegistry(
        household_id=household_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        members=members,
        require_consent=require_consent,
        enrollment_session_ids=sorted(
            session.manifest.session_id for sessions in grouped.values() for session in sessions
        ),
    )
    dump_json(registry_path, registry.model_dump(mode="json"))
    return {
        "schema_version": "physioatlas.household-enrollment.v1",
        "registry": str(Path(registry_path).resolve()),
        "members_enrolled": len(members),
        "members": [member.member_id for member in members],
        "rejected": rejected,
        "ready": len(members) >= 2,
        "privacy_boundary": "Only explicitly consented household members are enrolled; all other observations remain unknown.",
        "biological_distinctiveness": biological_distinctiveness(registry),
    }


def load_household_registry(path: Union[str, Path]) -> HouseholdRegistry:
    return HouseholdRegistry.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def classify_household_identity(
    registry: HouseholdRegistry,
    embeddings: dict[Modality, np.ndarray],
    embedding_versions: Optional[dict[Modality, str]] = None,
) -> IdentityDecision:
    if not embeddings:
        return IdentityDecision(None, None, 0.0, False, "no_identity_modalities", {}, None, None)
    candidates: list[tuple[float, HouseholdMemberProfile, dict[str, float], int, bool]] = []
    for member in registry.members:
        if not member.active:
            continue
        distances: dict[str, float] = {}
        threshold_passes = 0
        for profile in member.profiles:
            candidate = embeddings.get(profile.modality)
            if candidate is None:
                continue
            if (
                embedding_versions is not None
                and profile.modality in embedding_versions
                and embedding_versions[profile.modality] != profile.embedding_version
            ):
                continue
            vector = np.asarray(candidate, dtype=np.float64)
            vector /= max(np.linalg.norm(vector), 1e-12)
            centroid = np.asarray(profile.centroid, dtype=np.float64)
            distance = float(1.0 - np.clip(np.dot(vector, centroid), -1.0, 1.0))
            distances[profile.modality.value] = distance
            if distance <= profile.acceptance_threshold:
                threshold_passes += 1
        if not distances:
            continue
        fused = float(np.mean(list(distances.values())))
        enough = threshold_passes >= min(member.minimum_modalities, len(distances))
        candidates.append((fused, member, distances, threshold_passes, enough))
    if not candidates:
        return IdentityDecision(None, None, 0.0, False, "no_compatible_profiles", {}, None, None)
    candidates.sort(key=lambda item: item[0])
    best_distance, best_member, scores, _, enough = candidates[0]
    second_distance = candidates[1][0] if len(candidates) > 1 else 1.0
    margin = float(second_distance - best_distance)
    margin_ratio = float(margin / max(best_distance, 1e-6))
    profile_thresholds = [
        profile.acceptance_threshold
        for profile in best_member.profiles
        if profile.modality.value in scores
    ]
    threshold = float(np.mean(profile_thresholds)) if profile_thresholds else 0.0
    accepted = bool(enough and best_distance <= threshold and margin_ratio >= registry.open_set_margin)
    confidence = float(
        np.clip(0.55 * (1.0 - best_distance / max(threshold, 1e-6)) + 0.45 * min(margin_ratio, 2.0) / 2.0, 0.0, 1.0)
    )
    reason = "accepted" if accepted else (
        "ambiguous_household_match" if margin_ratio < registry.open_set_margin else "outside_enrollment_distribution"
    )
    return IdentityDecision(
        best_member.member_id if accepted else None,
        best_member.display_alias if accepted else None,
        confidence,
        accepted,
        reason,
        scores,
        best_distance,
        margin,
    )


def evaluate_household_registry(
    dataset_root: Union[str, Path],
    registry_path: Union[str, Path],
) -> dict[str, Any]:
    registry = load_household_registry(registry_path)
    confusion: dict[str, dict[str, int]] = {}
    rows = []
    accepted_correct = 0
    accepted_total = 0
    total = 0
    for manifest_path in iter_manifests(dataset_root):
        session = load_session(manifest_path)
        embeddings = _session_embeddings(session)
        decision = classify_household_identity(registry, embeddings)
        truth = session.manifest.subject_id
        predicted = decision.member_id or "unknown"
        confusion.setdefault(truth, {})[predicted] = confusion.setdefault(truth, {}).get(predicted, 0) + 1
        total += 1
        if decision.accepted:
            accepted_total += 1
            accepted_correct += int(predicted == truth)
        rows.append(
            {
                "session_id": session.manifest.session_id,
                "truth": truth,
                "prediction": predicted,
                **decision.to_dict(),
            }
        )
    accuracy = accepted_correct / accepted_total if accepted_total else 0.0
    coverage = accepted_total / total if total else 0.0
    evaluated_session_ids = [row["session_id"] for row in rows]
    overlapping_sessions = sorted(set(evaluated_session_ids) & set(registry.enrollment_session_ids))
    return make_json_safe(
        {
            "schema_version": "physioatlas.household-evaluation.v1",
            "sessions": total,
            "accepted": accepted_total,
            "accepted_accuracy": accuracy,
            "coverage": coverage,
            "overall_correct_fraction": accepted_correct / total if total else 0.0,
            "confusion": confusion,
            "predictions": rows,
            "overlapping_enrollment_sessions": overlapping_sessions,
            "split_is_disjoint": not overlapping_sessions,
            "valid": total > 0 and not overlapping_sessions and accepted_accuracy_gate(accuracy, coverage),
            "warning": None if not overlapping_sessions else "Evaluation overlaps enrollment sessions and cannot support a generalization claim.",
        }
    )


def accepted_accuracy_gate(accuracy: float, coverage: float) -> bool:
    return bool(accuracy >= 0.80 and coverage >= 0.50)


def biological_distinctiveness(registry: HouseholdRegistry) -> dict[str, Any]:
    """Summarize baseline separability without treating it as identity evidence."""
    features = []
    member_ids = []
    for member in registry.members:
        summary = member.biological_summary
        row = [
            summary.respiratory_rate_bpm,
            summary.heart_rate_bpm,
            summary.motion_index,
        ]
        if any(value is None or not np.isfinite(value) for value in row):
            continue
        features.append([float(value) for value in row])
        member_ids.append(member.member_id)
    if len(features) < 2:
        return {
            "available": False,
            "pairs": [],
            "warning": "At least two complete baselines are required.",
            "research_only": True,
        }
    matrix = np.asarray(features, dtype=np.float64)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - np.mean(matrix, axis=0)) / scale
    pairs = []
    for left in range(len(member_ids)):
        for right in range(left + 1, len(member_ids)):
            pairs.append(
                {
                    "member_a": member_ids[left],
                    "member_b": member_ids[right],
                    "normalized_baseline_distance": float(
                        np.linalg.norm(normalized[left] - normalized[right])
                    ),
                    "respiratory_rate_difference_bpm": abs(matrix[left, 0] - matrix[right, 0]),
                    "heart_rate_difference_bpm": abs(matrix[left, 1] - matrix[right, 1]),
                    "motion_index_difference": abs(matrix[left, 2] - matrix[right, 2]),
                }
            )
    return make_json_safe(
        {
            "available": True,
            "features": ["respiratory_rate_bpm", "heart_rate_bpm", "motion_index"],
            "pairs": pairs,
            "minimum_pair_distance": min(item["normalized_baseline_distance"] for item in pairs),
            "warning": "Biological baselines can overlap and must not be used as sole identity evidence.",
            "research_only": True,
        }
    )
