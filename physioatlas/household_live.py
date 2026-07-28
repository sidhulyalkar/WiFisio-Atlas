from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .household import HouseholdRegistry, biological_distinctiveness, load_household_registry
from .research_hub import ResearchHubStore
from .schema import Modality
from .tracking import MultiPersonTracker, TrackObservation
from .utils import make_json_safe


def _member_embedding(member, modality: Modality) -> Optional[np.ndarray]:
    for profile in member.profiles:
        if profile.modality == modality:
            return np.asarray(profile.centroid, dtype=np.float64)
    return None


def calibration_matrix(registry: HouseholdRegistry) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for member in registry.members:
        modalities = {}
        for profile in member.profiles:
            calibration_score = float(
                np.clip(
                    1.0 - profile.within_distance_q95 / max(profile.acceptance_threshold, 1e-6),
                    0.0,
                    1.0,
                )
            )
            modalities[profile.modality.value] = {
                "samples": profile.samples,
                "within_distance_q95": profile.within_distance_q95,
                "acceptance_threshold": profile.acceptance_threshold,
                "calibration_score": calibration_score,
                "status": "calibrated" if profile.samples >= 3 and calibration_score >= 0.5 else "needs_more_data",
            }
        biology = member.biological_summary
        physiology_score_components = [
            min(1.0, biology.sessions / 3.0),
            biology.respiratory_rate_confidence,
            biology.heart_rate_confidence,
        ]
        if biology.respiratory_rate_bpm and biology.respiratory_rate_std_bpm is not None:
            physiology_score_components.append(
                max(0.0, 1.0 - biology.respiratory_rate_std_bpm / max(biology.respiratory_rate_bpm, 1e-6))
            )
        if biology.heart_rate_bpm and biology.heart_rate_std_bpm is not None:
            physiology_score_components.append(
                max(0.0, 1.0 - biology.heart_rate_std_bpm / max(biology.heart_rate_bpm, 1e-6))
            )
        physiology_score = float(np.clip(np.mean(physiology_score_components), 0.0, 1.0))
        output[member.member_id] = {
            "display_alias": member.display_alias,
            "modalities": modalities,
            "biology": biology.model_dump(mode="json"),
            "physiology_calibration_score": physiology_score,
            "physiology_status": "calibrated" if biology.sessions >= 3 and physiology_score >= 0.65 else "needs_more_data",
        }
    return output


def _rolling_track_series(
    previous: dict[str, Any], tracks: list, timestamp_s: float, *, maximum: int = 300
) -> dict[str, list[dict[str, Any]]]:
    series = {key: list(values) for key, values in dict(previous or {}).items()}
    active_ids = {track.track_id for track in tracks}
    for track in tracks:
        values = series.setdefault(track.track_id, [])
        values.append(
            {
                "timestamp_s": timestamp_s,
                "heart_rate_bpm": track.heart_rate_bpm,
                "respiratory_rate_bpm": track.respiratory_rate_bpm,
                "motion_index": track.motion_index,
                "observability": track.observability,
                "signal_quality": track.signal_quality,
            }
        )
        series[track.track_id] = values[-maximum:]
    # Preserve recently disappeared tracks for inspection but cap total keys.
    ordered = sorted(series, key=lambda key: (key not in active_ids, key))
    return {key: series[key] for key in ordered[-64:]}


def simulate_household_live(
    registry_path: Union[str, Path],
    state_path: Union[str, Path],
    *,
    steps: int = 40,
    seed: int = 41,
    interval_s: float = 0.0,
    include_unknown: bool = True,
) -> dict[str, Any]:
    registry = load_household_registry(registry_path)
    if len(registry.members) < 2:
        raise ValueError("Household live simulation requires at least two enrolled members")
    rng = np.random.default_rng(seed)
    tracker = MultiPersonTracker(registry, identity_after_observations=3)
    store = ResearchHubStore(state_path)
    base_positions = {
        member.member_id: np.asarray([1.0 + index * 1.5, 1.0 + (index % 2) * 1.2])
        for index, member in enumerate(registry.members)
    }
    history = []
    for step in range(steps):
        observations = []
        for index, member in enumerate(registry.members):
            embeddings = {}
            for modality in (Modality.WIFI_CSI, Modality.MMWAVE):
                centroid = _member_embedding(member, modality)
                if centroid is None:
                    continue
                noisy = centroid + rng.normal(0, 0.006, size=centroid.shape)
                noisy /= max(np.linalg.norm(noisy), 1e-12)
                embeddings[modality] = noisy
            biology = member.biological_summary
            position = base_positions[member.member_id] + np.asarray(
                [0.18 * np.sin(0.22 * step + index), 0.12 * np.cos(0.17 * step + index)]
            )
            observations.append(
                TrackObservation(
                    timestamp_s=float(step),
                    position_m=position,
                    embeddings=embeddings,
                    respiratory_rate_bpm=(biology.respiratory_rate_bpm or 14.0) + rng.normal(0, 0.18),
                    heart_rate_bpm=(biology.heart_rate_bpm or 70.0) + rng.normal(0, 0.45),
                    motion_index=max(0.0, (biology.motion_index or 0.2) + rng.normal(0, 0.01)),
                    signal_quality=float(np.clip(0.92 + rng.normal(0, 0.025), 0.0, 1.0)),
                    observability=float(np.clip(0.90 + rng.normal(0, 0.04), 0.0, 1.0)),
                )
            )
        if include_unknown and step >= max(4, steps // 4):
            dimension = len(registry.members[0].profiles[0].centroid)
            unknown = rng.normal(size=dimension)
            unknown /= max(np.linalg.norm(unknown), 1e-12)
            observations.append(
                TrackObservation(
                    timestamp_s=float(step),
                    position_m=np.asarray([0.5 + 0.04 * step, 3.0]),
                    embeddings={Modality.WIFI_CSI: unknown},
                    respiratory_rate_bpm=16.5,
                    heart_rate_bpm=82.0,
                    motion_index=0.28,
                    signal_quality=0.83,
                    observability=0.78,
                )
            )
        tracks = tracker.update(observations)
        snapshot = tracker.snapshot()
        state = {
            "schema_version": "physioatlas.research-hub-state.v1",
            "status": "live_simulation",
            "session_id": "household-synthetic-live",
            "step": step,
            "household": {
                "household_id": registry.household_id,
                "members": [
                    {
                        "member_id": member.member_id,
                        "display_alias": member.display_alias,
                        "baseline": member.biological_summary.model_dump(mode="json"),
                    }
                    for member in registry.members
                ],
                "tracks": [track.to_dict() for track in tracks],
                "track_series": _rolling_track_series(
                    store.read().get("household", {}).get("track_series", {}), tracks, float(step)
                ),
                "anonymous_tracks": snapshot["anonymous_tracks"],
                "identified_tracks": snapshot["identified_tracks"],
                "biological_distinctiveness": biological_distinctiveness(registry),
            },
            "modalities": {
                "wifi_csi": {"status": "simulated", "quality": 0.90},
                "mmwave": {"status": "simulated", "quality": 0.88},
                "reference_sensors": {"status": "synthetic"},
            },
            "calibration": calibration_matrix(registry),
            "studies": store.read().get("studies", []),
            "experiments": store.read().get("experiments", []),
            "events": store.read().get("events", []),
            "privacy": {
                "local_only": True,
                "consent_required": True,
                "unknown_people_anonymous": True,
                "identity_mode": "open_set_enrolled_household_only",
            },
            "research_only": True,
        }
        store.publish(state)
        history.append(snapshot)
        if interval_s > 0:
            time.sleep(interval_s)
    final = store.read()
    all_tracks = final.get("household", {}).get("tracks", [])
    result = {
        "schema_version": "physioatlas.household-live-simulation.v1",
        "state_path": str(Path(state_path).resolve()),
        "steps": steps,
        "enrolled_members": len(registry.members),
        "identified_tracks": sum(item.get("member_id") is not None for item in all_tracks),
        "anonymous_tracks": sum(item.get("member_id") is None for item in all_tracks),
        "unknown_remained_anonymous": any(item.get("member_id") is None for item in all_tracks) if include_unknown else True,
        "valid": sum(item.get("member_id") is not None for item in all_tracks) >= len(registry.members)
        and (not include_unknown or any(item.get("member_id") is None for item in all_tracks)),
        "research_only": True,
    }
    store.append_event("simulation_completed", result)
    return make_json_safe(result)


def observation_from_payload(payload: dict[str, Any]) -> TrackObservation:
    if not isinstance(payload, dict):
        raise ValueError("Household observation must be a JSON object")
    embeddings = {}
    raw_versions = payload.get(
        "identity_embedding_versions", payload.get("identity_embedding_version", {})
    )
    if isinstance(raw_versions, str):
        embedding_versions = {
            Modality(key): raw_versions for key in dict(payload.get("embeddings", {}))
        }
    elif isinstance(raw_versions, dict):
        embedding_versions = {Modality(key): str(value) for key, value in raw_versions.items()}
    else:
        raise ValueError("identity embedding versions must be a string or modality mapping")
    for key, value in dict(payload.get("embeddings", {})).items():
        modality = Modality(key)
        if modality not in {Modality.WIFI_CSI, Modality.MMWAVE}:
            raise ValueError(f"Unsupported identity embedding modality: {key}")
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"Embedding {key} must contain finite values")
        embeddings[modality] = vector
    return TrackObservation(
        timestamp_s=float(payload["timestamp_s"]),
        position_m=np.asarray(payload["position_m"], dtype=np.float64),
        embeddings=embeddings,
        embedding_versions=embedding_versions,
        signal_quality=float(payload.get("signal_quality", 1.0)),
        respiratory_rate_bpm=payload.get("respiratory_rate_bpm"),
        heart_rate_bpm=payload.get("heart_rate_bpm"),
        motion_index=payload.get("motion_index"),
        observability=float(payload.get("observability", 1.0)),
    )


def _publish_tracker_state(
    store: ResearchHubStore,
    registry: HouseholdRegistry,
    tracker: MultiPersonTracker,
    *,
    status: str,
    session_id: str,
    modalities: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    previous = store.read()
    state = {
        "schema_version": "physioatlas.research-hub-state.v1",
        "status": status,
        "session_id": session_id,
        "household": {
            "household_id": registry.household_id,
            "members": [
                {
                    "member_id": member.member_id,
                    "display_alias": member.display_alias,
                    "baseline": member.biological_summary.model_dump(mode="json"),
                }
                for member in registry.members
            ],
            "tracks": [track.to_dict() for track in tracker.tracks],
            "track_series": _rolling_track_series(
                previous.get("household", {}).get("track_series", {}),
                tracker.tracks,
                max((track.last_timestamp_s for track in tracker.tracks), default=0.0),
            ),
            "anonymous_tracks": sum(track.member_id is None for track in tracker.tracks),
            "identified_tracks": sum(track.member_id is not None for track in tracker.tracks),
            "biological_distinctiveness": biological_distinctiveness(registry),
        },
        "modalities": modalities or previous.get("modalities", {}),
        "calibration": calibration_matrix(registry),
        "studies": previous.get("studies", []),
        "experiments": previous.get("experiments", []),
        "events": previous.get("events", []),
        "privacy": {
            "local_only": True,
            "consent_required": True,
            "unknown_people_anonymous": True,
            "identity_mode": "open_set_enrolled_household_only",
        },
        "research_only": True,
    }
    store.publish(state)
    return state


def run_household_jsonl(
    input_path: Union[str, Path],
    registry_path: Union[str, Path],
    state_path: Union[str, Path],
    *,
    session_id: str = "household-jsonl-live",
) -> dict[str, Any]:
    """Consume localized person observations produced by RuView/InnerLoop.

    This bridge intentionally expects already-separated person observations.
    Raw single-link CSI does not uniquely identify which person produced each
    component, so upstream localization or multi-link separation is required.
    """
    registry = load_household_registry(registry_path)
    tracker = MultiPersonTracker(registry)
    store = ResearchHubStore(state_path)
    grouped: dict[float, list[TrackObservation]] = {}
    rejected = 0
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                observation = observation_from_payload(__import__("json").loads(line))
                grouped.setdefault(observation.timestamp_s, []).append(observation)
            except Exception:
                rejected += 1
    for timestamp in sorted(grouped):
        tracker.update(grouped[timestamp])
        _publish_tracker_state(
            store,
            registry,
            tracker,
            status="jsonl_replay",
            session_id=session_id,
            modalities={"localized_observations": {"status": "replay", "quality": 1.0}},
        )
    result = {
        "schema_version": "physioatlas.household-jsonl-report.v1",
        "input": str(Path(input_path).resolve()),
        "state": str(Path(state_path).resolve()),
        "timestamps": len(grouped),
        "accepted_observations": sum(len(items) for items in grouped.values()),
        "rejected_observations": rejected,
        "tracks": len(tracker.tracks),
        "identified": sum(track.member_id is not None for track in tracker.tracks),
        "anonymous": sum(track.member_id is None for track in tracker.tracks),
        "valid": bool(grouped) and len(tracker.tracks) > 0,
        "status": "passed" if grouped and len(tracker.tracks) > 0 else "failed",
        "research_only": True,
    }
    store.append_event("jsonl_replay_completed", result)
    return make_json_safe(result)


def run_household_udp(
    registry_path: Union[str, Path],
    state_path: Union[str, Path],
    *,
    host: str = "127.0.0.1",
    port: int = 8790,
    duration_s: float = 30.0,
    session_id: str = "household-udp-live",
) -> dict[str, Any]:
    import json
    import socket

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    registry = load_household_registry(registry_path)
    tracker = MultiPersonTracker(registry)
    store = ResearchHubStore(state_path)
    accepted = 0
    rejected = 0
    started = time.monotonic()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.25)
    try:
        while time.monotonic() - started < duration_s:
            try:
                raw, _ = sock.recvfrom(1_000_000)
            except socket.timeout:
                continue
            try:
                payload = json.loads(raw)
                items = payload if isinstance(payload, list) else [payload]
                observations = [observation_from_payload(item) for item in items]
                tracker.update(observations)
                accepted += len(observations)
                _publish_tracker_state(
                    store,
                    registry,
                    tracker,
                    status="udp_live",
                    session_id=session_id,
                    modalities={"localized_observations": {"status": "live", "quality": 1.0}},
                )
            except Exception as exc:
                rejected += 1
                store.append_event("udp_observation_rejected", {"error": str(exc)})
    finally:
        sock.close()
    result = {
        "schema_version": "physioatlas.household-udp-report.v1",
        "host": host,
        "port": port,
        "duration_s": duration_s,
        "accepted_observations": accepted,
        "rejected_datagrams": rejected,
        "tracks": len(tracker.tracks),
        "valid": accepted > 0 and len(tracker.tracks) > 0,
        "status": "passed" if accepted > 0 and len(tracker.tracks) > 0 else "failed",
        "research_only": True,
    }
    store.append_event("udp_live_completed", result)
    return make_json_safe(result)
