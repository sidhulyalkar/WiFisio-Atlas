from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .calibration_reflector import (
    DeterministicSimulatedReflector,
    SimulatedReflectorDynamics,
    analyze_reflector_telemetry,
)
from .calibration_reflector_schema import (
    ReflectorCalibrationHealth,
    ReflectorDeviceIdentity,
    ReflectorHealthThresholds,
    ReflectorInstallation,
    ReflectorSafetyLimits,
    ReflectorTelemetrySample,
    ReflectorTrajectoryPlan,
)
from .household_bfi import (
    build_household_bfi_calibration,
    process_household_bfi,
)
from .reflector_rf_atlas import (
    build_reflector_rf_atlas,
    compare_reflector_atlases,
)
from .reflector_rf_atlas_schema import ReflectorRfAtlas, ReflectorRfSample
from .rf_fusion import (
    build_bfi_zone_prototype_model,
    process_fusion_windows_jsonl,
)


def add_rf_research_parsers(subparsers: argparse._SubParsersAction) -> None:
    bfi_calibrate = subparsers.add_parser(
        "household-bfi-calibrate",
        help="Build a robust empty-room calibration for fixed BFI streams",
    )
    bfi_calibrate.add_argument("--empty", required=True)
    bfi_calibrate.add_argument("--output", required=True)
    bfi_calibrate.add_argument("--environment-id", required=True)

    bfi_process = subparsers.add_parser(
        "household-bfi-process",
        help="Normalize decoded beamforming feedback for later evidence fusion",
    )
    bfi_process.add_argument("--input", required=True)
    bfi_process.add_argument("--calibration", required=True)
    bfi_process.add_argument("--output", required=True)

    bfi_zones = subparsers.add_parser(
        "household-bfi-zone-calibrate",
        help="Fit BFI single-target zone prototypes from normalized captures",
    )
    bfi_zones.add_argument(
        "--zones",
        required=True,
        help="JSON object mapping zone IDs to normalized BFI JSONL captures",
    )
    bfi_zones.add_argument("--output", required=True)

    fusion = subparsers.add_parser(
        "household-csi-bfi-fuse",
        help="Fuse calibrated CSI/BFI zone evidence with explicit abstention",
    )
    fusion.add_argument(
        "--input",
        required=True,
        help="JSONL windows containing an evidence list",
    )
    fusion.add_argument("--config", required=True)
    fusion.add_argument("--output", required=True)

    reflector_simulate = subparsers.add_parser(
        "reflector-simulate-calibration",
        help="Run the explicitly synthetic, safety-gated reflector driver",
    )
    reflector_simulate.add_argument("--config", required=True)
    reflector_simulate.add_argument("--telemetry", required=True)
    reflector_simulate.add_argument("--health", required=True)

    reflector_analyze = subparsers.add_parser(
        "reflector-analyze-telemetry",
        help="Validate reflector telemetry before accepting an RF calibration",
    )
    reflector_analyze.add_argument("--input", required=True)
    reflector_analyze.add_argument("--output", required=True)
    reflector_analyze.add_argument("--thresholds", help="Optional JSON threshold object")

    reflector_atlas = subparsers.add_parser(
        "reflector-build-rf-atlas",
        help="Build a motor-off waypoint RF atlas from aligned CSI/BFI samples",
    )
    reflector_atlas.add_argument("--telemetry", required=True)
    reflector_atlas.add_argument("--health", required=True)
    reflector_atlas.add_argument("--rf-samples", required=True)
    reflector_atlas.add_argument("--atlas-id", required=True)
    reflector_atlas.add_argument("--environment-id", required=True)
    reflector_atlas.add_argument("--output", required=True)

    reflector_compare = subparsers.add_parser(
        "reflector-compare-atlas",
        help="Compare a new reflector transfer atlas with a baseline",
    )
    reflector_compare.add_argument("--baseline", required=True)
    reflector_compare.add_argument("--candidate", required=True)
    reflector_compare.add_argument("--maximum-drift-score", type=float, default=0.35)
    reflector_compare.add_argument("--output", required=True)


def _zone_paths(manifest_path: str) -> dict[str, Path]:
    source = Path(manifest_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("BFI zone manifest must be a non-empty JSON object")
    result = {}
    for zone_id, value in payload.items():
        path = Path(str(value))
        if not path.is_absolute():
            path = source.parent / path
        result[str(zone_id)] = path
    return result


def handle_rf_research_command(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.command == "household-bfi-calibrate":
        result = build_household_bfi_calibration(
            args.empty, args.output, environment_id=args.environment_id
        )
        return result.model_dump(mode="json")
    if args.command == "household-bfi-process":
        result = process_household_bfi(args.input, args.calibration, args.output)
        return result.model_dump(mode="json")
    if args.command == "household-bfi-zone-calibrate":
        result = build_bfi_zone_prototype_model(_zone_paths(args.zones), args.output)
        return result.model_dump(mode="json")
    if args.command == "household-csi-bfi-fuse":
        return process_fusion_windows_jsonl(args.input, args.config, args.output)
    if args.command == "reflector-simulate-calibration":
        payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        driver = DeterministicSimulatedReflector(
            identity=ReflectorDeviceIdentity.model_validate(payload["identity"]),
            installation=ReflectorInstallation.model_validate(payload["installation"]),
            limits=ReflectorSafetyLimits.model_validate(payload["safety_limits"]),
            dynamics=SimulatedReflectorDynamics(**payload.get("simulation", {})),
        )
        plan = ReflectorTrajectoryPlan.model_validate(payload["trajectory"])
        thresholds = ReflectorHealthThresholds.model_validate(
            payload.get("health_thresholds", {})
        )
        driver.arm()
        telemetry = driver.execute(plan)
        health = analyze_reflector_telemetry(telemetry, thresholds)
        telemetry_path = Path(args.telemetry)
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with telemetry_path.open("w", encoding="utf-8") as handle:
            for sample in telemetry:
                handle.write(sample.model_dump_json() + "\n")
        health_path = Path(args.health)
        health_path.parent.mkdir(parents=True, exist_ok=True)
        health_path.write_text(health.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return health.model_dump(mode="json")
    if args.command == "reflector-analyze-telemetry":
        telemetry = [
            ReflectorTelemetrySample.model_validate_json(line)
            for line in Path(args.input).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        threshold_payload = (
            json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
            if args.thresholds
            else {}
        )
        health = analyze_reflector_telemetry(
            telemetry, ReflectorHealthThresholds.model_validate(threshold_payload)
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(health.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return health.model_dump(mode="json")
    if args.command == "reflector-build-rf-atlas":
        telemetry = [
            ReflectorTelemetrySample.model_validate_json(line)
            for line in Path(args.telemetry).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        health = ReflectorCalibrationHealth.model_validate_json(
            Path(args.health).read_text(encoding="utf-8")
        )
        if {item.plan_id for item in telemetry} != {health.plan_id}:
            raise ValueError("declared reflector health does not match telemetry")
        rf_samples = [
            ReflectorRfSample.model_validate_json(line)
            for line in Path(args.rf_samples).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        atlas = build_reflector_rf_atlas(
            telemetry,
            rf_samples,
            health,
            atlas_id=args.atlas_id,
            environment_id=args.environment_id,
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(atlas.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return atlas.model_dump(mode="json")
    if args.command == "reflector-compare-atlas":
        baseline = ReflectorRfAtlas.model_validate_json(
            Path(args.baseline).read_text(encoding="utf-8")
        )
        candidate = ReflectorRfAtlas.model_validate_json(
            Path(args.candidate).read_text(encoding="utf-8")
        )
        report = compare_reflector_atlases(
            baseline,
            candidate,
            maximum_drift_score=args.maximum_drift_score,
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return report.model_dump(mode="json")
    return None
