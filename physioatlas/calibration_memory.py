from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class CalibrationProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.calibration-profile.v1"
    profile_id: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    hardware_ids: list[str] = Field(default_factory=list)
    geometry_fingerprint: list[float] = Field(default_factory=list)
    empty_room_features: list[float] = Field(default_factory=list)
    clock_calibrations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    geometry_calibrations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    uncertainty_profile: dict[str, float] = Field(default_factory=dict)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


def profile_sha256(profile: CalibrationProfile) -> str:
    payload = profile.model_dump(mode="json")
    payload.pop("created_at_utc", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_profile(path: Union[str, Path], profile: CalibrationProfile) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = profile.model_dump(mode="json")
    payload["sha256"] = profile_sha256(profile)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_profile(path: Union[str, Path]) -> CalibrationProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = payload.pop("sha256", None)
    profile = CalibrationProfile.model_validate(payload)
    if expected is not None and expected != profile_sha256(profile):
        raise ValueError("Calibration profile hash mismatch")
    return profile


def select_calibration_profile(
    profiles: list[CalibrationProfile],
    *,
    environment_id: Optional[str] = None,
    hardware_ids: Optional[list[str]] = None,
    geometry_fingerprint: Optional[np.ndarray] = None,
) -> tuple[CalibrationProfile, dict[str, float]]:
    if not profiles:
        raise ValueError("At least one calibration profile is required")
    requested_hardware = set(hardware_ids or [])
    requested_geometry = None if geometry_fingerprint is None else np.asarray(geometry_fingerprint, dtype=float).reshape(-1)
    scored: list[tuple[float, CalibrationProfile, dict[str, float]]] = []
    for profile in profiles:
        environment_penalty = 0.0 if environment_id is None or profile.environment_id == environment_id else 1.0
        profile_hardware = set(profile.hardware_ids)
        hardware_penalty = 0.0
        if requested_hardware:
            union = requested_hardware | profile_hardware
            hardware_penalty = 1.0 - len(requested_hardware & profile_hardware) / max(1, len(union))
        geometry_penalty = 0.0
        if requested_geometry is not None:
            candidate = np.asarray(profile.geometry_fingerprint, dtype=float).reshape(-1)
            if candidate.shape != requested_geometry.shape or candidate.size == 0:
                geometry_penalty = 1.0
            else:
                geometry_penalty = float(np.linalg.norm(candidate - requested_geometry) / np.sqrt(candidate.size))
        score = environment_penalty + hardware_penalty + geometry_penalty
        detail = {
            "total_penalty": float(score),
            "environment_penalty": float(environment_penalty),
            "hardware_penalty": float(hardware_penalty),
            "geometry_penalty": float(geometry_penalty),
        }
        scored.append((score, profile, detail))
    scored.sort(key=lambda item: (item[0], item[1].profile_id))
    return scored[0][1], scored[0][2]
