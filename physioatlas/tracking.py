from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .household import HouseholdRegistry, classify_household_identity
from .schema import Modality
from .utils import make_json_safe


@dataclass(frozen=True)
class TrackObservation:
    timestamp_s: float
    position_m: np.ndarray
    embeddings: dict[Modality, np.ndarray]
    embedding_versions: dict[Modality, str] = field(default_factory=dict)
    signal_quality: float = 1.0
    respiratory_rate_bpm: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    motion_index: Optional[float] = None
    observability: float = 1.0

    def validate(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64).reshape(-1)
        if position.size not in {2, 3} or not np.all(np.isfinite(position)):
            raise ValueError("Track observation position must contain two or three finite coordinates")
        if not np.isfinite(self.timestamp_s):
            raise ValueError("Track observation timestamp must be finite")
        if not 0.0 <= self.signal_quality <= 1.0:
            raise ValueError("signal_quality must be within [0, 1]")
        if not 0.0 <= self.observability <= 1.0:
            raise ValueError("observability must be within [0, 1]")


@dataclass
class ActiveTrack:
    track_id: str
    last_timestamp_s: float
    position_m: np.ndarray
    velocity_mps: np.ndarray
    embeddings: dict[Modality, np.ndarray] = field(default_factory=dict)
    embedding_versions: dict[Modality, str] = field(default_factory=dict)
    observations: int = 1
    missed_updates: int = 0
    member_id: Optional[str] = None
    display_alias: Optional[str] = None
    identity_confidence: float = 0.0
    identity_reason: str = "not_enough_observations"
    respiratory_rate_bpm: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    motion_index: Optional[float] = None
    signal_quality: float = 1.0
    observability: float = 1.0

    def predict_position(self, timestamp_s: float) -> np.ndarray:
        delta = max(0.0, timestamp_s - self.last_timestamp_s)
        return self.position_m + self.velocity_mps * delta

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(
            {
                "track_id": self.track_id,
                "member_id": self.member_id,
                "display_alias": self.display_alias,
                "identity_confidence": self.identity_confidence,
                "identity_reason": self.identity_reason,
                "identity_embedding_versions": {
                    key.value: value for key, value in self.embedding_versions.items()
                },
                "position_m": self.position_m.tolist(),
                "velocity_mps": self.velocity_mps.tolist(),
                "observations": self.observations,
                "missed_updates": self.missed_updates,
                "respiratory_rate_bpm": self.respiratory_rate_bpm,
                "heart_rate_bpm": self.heart_rate_bpm,
                "motion_index": self.motion_index,
                "signal_quality": self.signal_quality,
                "observability": self.observability,
                "research_only": True,
            }
        )


class MultiPersonTracker:
    """Position/embedding tracker with open-set household identity assignment.

    Anonymous track IDs are the primary identifiers. A household profile is
    attached only after repeated observations and a consent-gated registry match.
    """

    def __init__(
        self,
        registry: Optional[HouseholdRegistry] = None,
        *,
        position_gate_m: float = 1.5,
        embedding_gate: float = 0.5,
        max_missed_updates: int = 8,
        identity_after_observations: int = 4,
        smoothing: float = 0.25,
    ) -> None:
        self.registry = registry
        self.position_gate_m = position_gate_m
        self.embedding_gate = embedding_gate
        self.max_missed_updates = max_missed_updates
        self.identity_after_observations = identity_after_observations
        self.smoothing = smoothing
        self._tracks: dict[str, ActiveTrack] = {}
        self._next_track = 1

    @property
    def tracks(self) -> list[ActiveTrack]:
        return sorted(self._tracks.values(), key=lambda item: item.track_id)

    @staticmethod
    def _embedding_distance(
        left: dict[Modality, np.ndarray],
        right: dict[Modality, np.ndarray],
        left_versions: Optional[dict[Modality, str]] = None,
        right_versions: Optional[dict[Modality, str]] = None,
    ) -> float:
        distances = []
        for modality in sorted(set(left) & set(right), key=lambda item: item.value):
            left_version = (left_versions or {}).get(modality)
            right_version = (right_versions or {}).get(modality)
            if left_version is not None and right_version is not None and left_version != right_version:
                continue
            a = np.asarray(left[modality], dtype=np.float64)
            b = np.asarray(right[modality], dtype=np.float64)
            a /= max(np.linalg.norm(a), 1e-12)
            b /= max(np.linalg.norm(b), 1e-12)
            distances.append(1.0 - float(np.clip(np.dot(a, b), -1.0, 1.0)))
        return float(np.mean(distances)) if distances else 0.5

    def _association_cost(self, track: ActiveTrack, observation: TrackObservation) -> float:
        predicted = track.predict_position(observation.timestamp_s)
        position = np.asarray(observation.position_m, dtype=np.float64).reshape(-1)
        if predicted.size != position.size:
            if predicted.size == 2:
                predicted = np.pad(predicted, (0, 1))
            if position.size == 2:
                position = np.pad(position, (0, 1))
        position_distance = float(np.linalg.norm(predicted - position))
        if position_distance > self.position_gate_m:
            return 1e6
        embedding_distance = self._embedding_distance(
            track.embeddings,
            observation.embeddings,
            track.embedding_versions,
            observation.embedding_versions,
        )
        if track.embeddings and observation.embeddings and embedding_distance > self.embedding_gate:
            return 1e6
        quality_penalty = 0.25 * (1.0 - observation.signal_quality)
        return position_distance / max(self.position_gate_m, 1e-6) + embedding_distance + quality_penalty

    @staticmethod
    def _smooth_optional(current: Optional[float], new: Optional[float], alpha: float) -> Optional[float]:
        if new is None or not np.isfinite(new):
            return current
        return float(new) if current is None else float((1.0 - alpha) * current + alpha * new)

    def _update_track(self, track: ActiveTrack, observation: TrackObservation) -> None:
        observation.validate()
        new_position = np.asarray(observation.position_m, dtype=np.float64).reshape(-1)
        if track.position_m.size != new_position.size:
            dimension = max(track.position_m.size, new_position.size)
            track.position_m = np.pad(track.position_m, (0, dimension - track.position_m.size))
            track.velocity_mps = np.pad(track.velocity_mps, (0, dimension - track.velocity_mps.size))
            new_position = np.pad(new_position, (0, dimension - new_position.size))
        delta = max(observation.timestamp_s - track.last_timestamp_s, 1e-6)
        instantaneous_velocity = (new_position - track.position_m) / delta
        alpha = self.smoothing
        track.velocity_mps = (1.0 - alpha) * track.velocity_mps + alpha * instantaneous_velocity
        track.position_m = (1.0 - alpha) * track.position_m + alpha * new_position
        for modality, embedding in observation.embeddings.items():
            vector = np.asarray(embedding, dtype=np.float64)
            vector /= max(np.linalg.norm(vector), 1e-12)
            incoming_version = observation.embedding_versions.get(modality)
            existing_version = track.embedding_versions.get(modality)
            if (
                incoming_version is not None
                and existing_version is not None
                and incoming_version != existing_version
            ):
                continue
            if modality in track.embeddings:
                combined = (1.0 - alpha) * track.embeddings[modality] + alpha * vector
                combined /= max(np.linalg.norm(combined), 1e-12)
                track.embeddings[modality] = combined
            else:
                track.embeddings[modality] = vector
            if incoming_version is not None:
                track.embedding_versions[modality] = incoming_version
        track.last_timestamp_s = observation.timestamp_s
        track.observations += 1
        track.missed_updates = 0
        track.signal_quality = float((1.0 - alpha) * track.signal_quality + alpha * observation.signal_quality)
        track.observability = float((1.0 - alpha) * track.observability + alpha * observation.observability)
        track.respiratory_rate_bpm = self._smooth_optional(
            track.respiratory_rate_bpm, observation.respiratory_rate_bpm, alpha
        )
        track.heart_rate_bpm = self._smooth_optional(
            track.heart_rate_bpm, observation.heart_rate_bpm, alpha
        )
        track.motion_index = self._smooth_optional(track.motion_index, observation.motion_index, alpha)
        if self.registry is not None and track.observations >= self.identity_after_observations:
            decision = classify_household_identity(
                self.registry, track.embeddings, track.embedding_versions
            )
            track.member_id = decision.member_id
            track.display_alias = decision.display_alias
            track.identity_confidence = decision.confidence
            track.identity_reason = decision.reason

    def _create_track(self, observation: TrackObservation) -> ActiveTrack:
        observation.validate()
        track_id = f"unknown-track-{self._next_track:03d}"
        self._next_track += 1
        position = np.asarray(observation.position_m, dtype=np.float64).reshape(-1)
        embeddings = {}
        for modality, embedding in observation.embeddings.items():
            vector = np.asarray(embedding, dtype=np.float64)
            vector /= max(np.linalg.norm(vector), 1e-12)
            embeddings[modality] = vector
        track = ActiveTrack(
            track_id=track_id,
            last_timestamp_s=observation.timestamp_s,
            position_m=position,
            velocity_mps=np.zeros_like(position),
            embeddings=embeddings,
            embedding_versions=dict(observation.embedding_versions),
            respiratory_rate_bpm=observation.respiratory_rate_bpm,
            heart_rate_bpm=observation.heart_rate_bpm,
            motion_index=observation.motion_index,
            signal_quality=observation.signal_quality,
            observability=observation.observability,
        )
        self._tracks[track_id] = track
        return track

    def update(self, observations: list[TrackObservation]) -> list[ActiveTrack]:
        for observation in observations:
            observation.validate()
        tracks = self.tracks
        matched_tracks: set[str] = set()
        matched_observations: set[int] = set()
        if tracks and observations:
            cost = np.asarray(
                [[self._association_cost(track, observation) for observation in observations] for track in tracks],
                dtype=np.float64,
            )
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows.tolist(), columns.tolist()):
                if cost[row, column] >= 1e5:
                    continue
                track = tracks[row]
                self._update_track(track, observations[column])
                matched_tracks.add(track.track_id)
                matched_observations.add(column)
        for track in tracks:
            if track.track_id not in matched_tracks:
                track.missed_updates += 1
        for index, observation in enumerate(observations):
            if index not in matched_observations:
                self._create_track(observation)
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed_updates > self.max_missed_updates
        ]
        for track_id in expired:
            del self._tracks[track_id]
        return self.tracks

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "physioatlas.tracker-state.v1",
            "tracks": [track.to_dict() for track in self.tracks],
            "anonymous_tracks": sum(track.member_id is None for track in self.tracks),
            "identified_tracks": sum(track.member_id is not None for track in self.tracks),
            "research_only": True,
            "privacy_boundary": "Unenrolled or ambiguous people retain anonymous track IDs.",
        }
