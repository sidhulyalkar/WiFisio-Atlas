from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Union

import numpy as np

from .io import iter_manifests, load_session
from .physiology import load_physiology_graph
from .privacy import audit_dataset_consent
from .schema import Modality, RFRepresentation


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    resolved_root = root.resolve()
    for path in sorted({item.resolve() for item in files}):
        try:
            identity = path.relative_to(resolved_root).as_posix()
        except ValueError:
            identity = path.name
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stream_stats(timestamps: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    duration = float(timestamps[-1] - timestamps[0]) if timestamps.size >= 2 else 0.0
    delta = np.diff(timestamps)
    median_rate = float(1.0 / np.median(delta)) if delta.size and np.median(delta) > 0 else 0.0
    flattened = values.reshape(values.shape[0], -1)
    return {
        "samples": int(values.shape[0]),
        "features": int(flattened.shape[1]),
        "duration_seconds": duration,
        "median_sample_rate_hz": median_rate,
        "mean": float(np.mean(flattened)),
        "std": float(np.std(flattened)),
        "minimum": float(np.min(flattened)),
        "maximum": float(np.max(flattened)),
    }


def validate_dataset(
    root: Union[str, Path],
    *,
    require_consent: bool = False,
    max_clock_drift_ppm: float = 250.0,
    max_geometry_rmse_m: float = 0.10,
) -> dict:
    root_path = Path(root).resolve()
    manifests = list(iter_manifests(root_path))
    errors: list[str] = []
    warnings: list[str] = []
    subjects: set[str] = set()
    environments: set[str] = set()
    postures: set[str] = set()
    conditions: set[str] = set()
    hardware_domains: set[str] = set()
    session_ids: set[str] = set()
    sessions = 0
    modalities: set[str] = set()
    files_for_fingerprint: list[Path] = []
    session_reports: list[dict[str, Any]] = []

    for manifest_path in manifests:
        try:
            session = load_session(manifest_path)
            manifest = session.manifest
            sessions += 1
            subjects.add(manifest.subject_id)
            environments.add(manifest.environment_id)
            postures.add(manifest.posture or "unknown")
            conditions.add(manifest.condition or "unknown")
            hardware = sorted({sensor.hardware or "unknown" for sensor in session.geometry.sensors})
            hardware_domains.update(hardware)
            if manifest.session_id in session_ids:
                errors.append(f"{manifest_path}: duplicate session_id {manifest.session_id!r}")
            session_ids.add(manifest.session_id)

            stream_modalities = {reference.modality.value for reference in manifest.streams}
            modalities.update(stream_modalities)
            target_modalities = {target.source_modality.value for target in manifest.targets}
            missing_target_sources = target_modalities - stream_modalities
            if missing_target_sources:
                errors.append(
                    f"{manifest_path}: target source modalities missing streams: {sorted(missing_target_sources)}"
                )
            if not ({"wifi_csi", "mmwave", "uwb"} & stream_modalities):
                warnings.append(f"{manifest_path}: no RF stream; RF-to-physiology training is unavailable")

            manifest_files = [manifest_path, Path(manifest.geometry_path)]
            for optional_path in (
                manifest.consent_path,
                manifest.physiology_graph_path,
                manifest.calibration_path,
                manifest.privacy_policy_path,
            ):
                if optional_path:
                    optional = Path(optional_path)
                    if not optional.exists():
                        errors.append(f"{manifest_path}: referenced file does not exist: {optional}")
                    else:
                        manifest_files.append(optional)
            for reference in manifest.streams:
                manifest_files.append(Path(reference.path))
            files_for_fingerprint.extend(manifest_files)

            references = {
                f"{reference.modality.value}:{reference.sensor_id}": reference
                for reference in manifest.streams
            }
            stream_report: dict[str, Any] = {}
            for key, stream in session.streams.items():
                reference = references[key]
                if not np.all(np.isfinite(stream.values)):
                    errors.append(f"{manifest_path}: {key} contains non-finite values")
                if stream.timestamps_s.size < 2:
                    errors.append(f"{manifest_path}: {key} has fewer than two samples")
                    continue
                stats = _stream_stats(stream.timestamps_s, np.asarray(stream.values))
                stream_report[key] = {**stats, "representation": reference.representation.value}
                if stats["std"] < 1e-8:
                    warnings.append(f"{manifest_path}: {key} is approximately constant")
                if reference.representation in {
                    RFRepresentation.AMPLITUDE_PHASE,
                    RFRepresentation.IQ,
                } and stats["features"] % 2:
                    errors.append(
                        f"{manifest_path}: {key} declares {reference.representation.value} with an odd feature count"
                    )
                if stream.quality is not None:
                    if not np.all(np.isfinite(stream.quality)):
                        errors.append(f"{manifest_path}: {key} quality contains non-finite values")
                    if np.any((stream.quality < 0) | (stream.quality > 1)):
                        errors.append(f"{manifest_path}: {key} quality is outside [0, 1]")
                    low_quality = float(np.mean(stream.quality < 0.2))
                    stream_report[key]["low_quality_fraction"] = low_quality
                    if low_quality > 0.5:
                        warnings.append(f"{manifest_path}: {key} is low quality for more than half the session")

            for target in manifest.targets:
                if target.clinical_claim_allowed:
                    warnings.append(
                        f"{manifest_path}: target {target.name} permits a clinical claim; review required"
                    )
                if not target.supervision_required:
                    warnings.append(
                        f"{manifest_path}: target {target.name} does not require supervision; attribution risk"
                    )

            is_synthetic = bool(manifest.metadata.get("synthetic", False))
            if not is_synthetic and not manifest.consent_path and not manifest.consent_reference:
                message = f"{manifest_path}: non-synthetic session has no consent record"
                (errors if require_consent else warnings).append(message)

            positioned = {
                sensor.sensor_id
                for sensor in session.geometry.sensors
                if sensor.position_m is not None
            }
            for link in session.geometry.rf_links:
                if link.tx_sensor_id not in positioned or link.rx_sensor_id not in positioned:
                    warnings.append(
                        f"{manifest_path}: RF link {link.link_id} lacks complete sensor positions"
                    )
            if (
                session.geometry.calibration_uncertainty_m is not None
                and session.geometry.calibration_uncertainty_m > max_geometry_rmse_m
            ):
                errors.append(
                    f"{manifest_path}: geometry uncertainty {session.geometry.calibration_uncertainty_m:.4f} m exceeds {max_geometry_rmse_m:.4f} m"
                )

            graph_summary = None
            if manifest.physiology_graph_path:
                graph = load_physiology_graph(manifest.physiology_graph_path)
                graph_summary = {
                    "graph_id": graph.graph_id,
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                }
                target_node_ids = {target.graph_node_id for target in manifest.targets if target.graph_node_id}
                graph_node_ids = {node.node_id for node in graph.nodes}
                unknown_target_nodes = sorted(target_node_ids - graph_node_ids)
                if unknown_target_nodes:
                    errors.append(
                        f"{manifest_path}: targets reference unknown physiology nodes {unknown_target_nodes}"
                    )

            calibration_summary = None
            if manifest.calibration_path:
                calibration = json.loads(Path(manifest.calibration_path).read_text(encoding="utf-8"))
                clocks = calibration.get("clocks", {})
                excessive = {
                    name: float(values.get("drift_ppm", 0.0))
                    for name, values in clocks.items()
                    if abs(float(values.get("drift_ppm", 0.0))) > max_clock_drift_ppm
                }
                if excessive:
                    errors.append(f"{manifest_path}: excessive clock drift {excessive}")
                geometry_rmse = float(calibration.get("geometry_rmse_m", 0.0))
                if geometry_rmse > max_geometry_rmse_m:
                    errors.append(
                        f"{manifest_path}: calibration geometry RMSE {geometry_rmse:.4f} m exceeds threshold"
                    )
                calibration_summary = {
                    "clock_domains": sorted(clocks),
                    "geometry_rmse_m": geometry_rmse,
                }

            session_reports.append(
                {
                    "manifest": str(manifest_path),
                    "session_id": manifest.session_id,
                    "subject_id": manifest.subject_id,
                    "environment_id": manifest.environment_id,
                    "posture": manifest.posture,
                    "condition": manifest.condition,
                    "synthetic": is_synthetic,
                    "modalities": sorted(stream_modalities),
                    "streams": stream_report,
                    "physiology_graph": graph_summary,
                    "calibration": calibration_summary,
                }
            )
        except Exception as exc:  # validation should report all sessions
            errors.append(f"{manifest_path}: {exc}")

    if not manifests:
        errors.append(f"No manifest.json files found below {root_path}")
    if len(subjects) < 2:
        warnings.append("Fewer than two subjects: subject-held-out evaluation is impossible")
    if len(environments) < 2:
        warnings.append("Fewer than two environments: environment-held-out evaluation is impossible")

    consent_report = audit_dataset_consent(root_path, require_consent=require_consent)
    if require_consent and not consent_report["valid"]:
        errors.extend(consent_report["errors"])

    fingerprint = None
    if files_for_fingerprint and not errors:
        try:
            fingerprint = _dataset_fingerprint(files_for_fingerprint, root_path)
        except Exception as exc:
            errors.append(f"Could not fingerprint dataset: {exc}")

    return {
        "schema_version": "physioatlas.validation.v3",
        "valid": not errors,
        "root": str(root_path),
        "sessions": sessions,
        "subjects": len(subjects),
        "subject_ids": sorted(subjects),
        "environments": len(environments),
        "environment_ids": sorted(environments),
        "postures": sorted(postures),
        "conditions": sorted(conditions),
        "hardware_domains": sorted(hardware_domains),
        "modalities": sorted(modalities),
        "modality_session_counts": dict(
            sorted(Counter(modality for report in session_reports for modality in report["modalities"]).items())
        ),
        "dataset_sha256": fingerprint,
        "consent": consent_report,
        "session_reports": session_reports,
        "errors": errors,
        "warnings": warnings,
    }
