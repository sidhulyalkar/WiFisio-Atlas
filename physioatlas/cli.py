from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

from .acquisition import AcquisitionConfig, DEFAULT_REGISTRY, run_acquisition
from .adapters import csv_to_npz, ruview_jsonl_to_npz
from .calibration import estimate_clock_calibration, estimate_rigid_transform, save_calibration
from .calibration_memory import load_profile, select_calibration_profile
from .dashboard import DashboardServer, LiveStateStore
from .dataset import load_windows_from_root
from .experiment import run_experiment
from .household import enroll_household, evaluate_household_registry
from .household_csi import build_household_csi_calibration, process_mixed_household_csi
from .household_csi_schema import CsiProcessingConfig
from .household_live import run_household_jsonl, run_household_udp, simulate_household_live
from .household_protocol import initialize_household_workspace
from .household_verification import run_household_fault_test, run_household_smoke_test
from .physiology import analyze_dataset_propagation
from .pretraining import RFPretrainConfig, train_rf_foundation
from .privacy import audit_dataset_consent, create_privacy_export
from .research_hub import ResearchHubServer, ResearchHubStore
from .research_hub_worker import ResearchHubActionWorker
from .studies import PriorityStudy, StudyProtocol, list_priority_studies, load_study_protocol, run_all_priority_studies, run_priority_study
from .rf_field import (
    ActiveSensingCandidate,
    canonical_to_complex,
    delay_doppler_map,
    delay_spread_features,
    reconstruct_cir,
    select_active_measurement,
)
from .schema import Modality, RFRepresentation
from .synthetic import create_synthetic_cohort
from .training import TrainConfig, train_model
from .ultrasound import ultrasound_npz_to_displacement
from .uncertainty import conformal_interval, fit_split_conformal, interval_coverage
from .utils import make_json_safe
from .validation import validate_dataset
from .verification import doctor, run_fault_injection_test, run_smoke_test, verify_run


def _emit(payload: Union[dict, list]) -> None:
    print(json.dumps(make_json_safe(payload), indent=2, sort_keys=True, allow_nan=False))


def _load_vector(path: str) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return np.asarray(np.load(source, allow_pickle=False), dtype=float).reshape(-1)
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            key = "events_s" if "events_s" in archive else archive.files[0]
            return np.asarray(archive[key], dtype=float).reshape(-1)
    return np.loadtxt(source, delimiter=",", ndmin=1).reshape(-1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physioatlas",
        description="Research-only geometry-aware physiological RF toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser("synthesize", help="Create a deterministic synthetic cohort")
    synth.add_argument("--output", required=True)
    synth.add_argument("--subjects", type=int, default=4)
    synth.add_argument("--sessions-per-subject", type=int, default=1)
    synth.add_argument("--duration-seconds", type=float, default=30.0)
    synth.add_argument("--sample-rate-hz", type=float, default=20.0)
    synth.add_argument("--seed", type=int, default=7)

    validate = subparsers.add_parser("validate", help="Validate manifests and stream files")
    validate.add_argument("--data", required=True)
    validate.add_argument("--require-consent", action="store_true")

    adapters = subparsers.add_parser("list-adapters", help="List acquisition adapters")

    acquire = subparsers.add_parser("acquire", help="Run a declarative acquisition configuration")
    acquire.add_argument("config")

    convert_csi = subparsers.add_parser(
        "convert-ruview", help="Convert RuView CSI JSONL/RVCSI to PhysioAtlas NPZ"
    )
    convert_csi.add_argument("--input", required=True)
    convert_csi.add_argument("--output", required=True)

    convert_csv = subparsers.add_parser(
        "convert-csv", help="Convert a timestamped physiological CSV to NPZ"
    )
    convert_csv.add_argument("--input", required=True)
    convert_csv.add_argument("--output", required=True)
    convert_csv.add_argument("--timestamp-column", required=True)
    convert_csv.add_argument("--value-columns", nargs="+", required=True)
    convert_csv.add_argument("--timestamp-scale", type=float, default=1.0)

    ultrasound = subparsers.add_parser(
        "extract-ultrasound", help="Extract a displacement trace from ultrasound frame NPZ"
    )
    ultrasound.add_argument("--input", required=True)
    ultrasound.add_argument("--output", required=True)
    ultrasound.add_argument("--axial-spacing-m", type=float, default=1.0)
    ultrasound.add_argument("--lateral-spacing-m", type=float, default=1.0)

    clock = subparsers.add_parser("calibrate-clock", help="Fit an affine clock calibration")
    clock.add_argument("--source-events", required=True)
    clock.add_argument("--reference-events", required=True)
    clock.add_argument("--source-clock", default="sensor")
    clock.add_argument("--reference-clock", default="host")
    clock.add_argument("--output", required=True)

    geometry = subparsers.add_parser("calibrate-geometry", help="Fit a rigid 3-D transform")
    geometry.add_argument("--source-points", required=True, help="CSV/NPY with N x 3 points")
    geometry.add_argument("--target-points", required=True, help="CSV/NPY with N x 3 points")
    geometry.add_argument("--source-frame", default="sensor")
    geometry.add_argument("--target-frame", default="room_meters")
    geometry.add_argument("--output", required=True)

    propagation = subparsers.add_parser(
        "analyze-propagation", help="Estimate declared physiological graph delays"
    )
    propagation.add_argument("--data", required=True)
    propagation.add_argument("--sample-rate-hz", type=float, default=50.0)

    privacy = subparsers.add_parser("privacy-audit", help="Validate dataset consent scopes")
    privacy.add_argument("--data", required=True)
    privacy.add_argument("--require-consent", action="store_true")

    privacy_export = subparsers.add_parser(
        "privacy-export", help="Create a pseudonymized dataset export"
    )
    privacy_export.add_argument("--data", required=True)
    privacy_export.add_argument("--output", required=True)
    privacy_export.add_argument("--secret", required=True)
    privacy_export.add_argument(
        "--drop-modality", action="append", default=[], choices=[item.value for item in Modality]
    )

    dashboard = subparsers.add_parser("serve-dashboard", help="Serve the local research monitor")
    dashboard.add_argument("--state", default="outputs/physioatlas/live-state.json")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)

    publish = subparsers.add_parser("publish-dashboard", help="Publish a JSON state snapshot")
    publish.add_argument("--state", default="outputs/physioatlas/live-state.json")
    publish.add_argument("--json", required=True, help="JSON object or path to a JSON file")

    pretrain = subparsers.add_parser("pretrain-rf", help="Run label-free masked RF pretraining")
    pretrain.add_argument("--data", required=True)
    pretrain.add_argument("--output", required=True)
    pretrain.add_argument("--target-for-windowing", default="ecg")
    pretrain.add_argument("--sample-rate-hz", type=float, default=20.0)
    pretrain.add_argument("--window-seconds", type=float, default=8.0)
    pretrain.add_argument("--stride-seconds", type=float, default=4.0)
    pretrain.add_argument("--max-gap-s", type=float, default=0.25)
    pretrain.add_argument("--epochs", type=int, default=5)
    pretrain.add_argument("--batch-size", type=int, default=8)
    pretrain.add_argument("--learning-rate", type=float, default=1e-3)
    pretrain.add_argument("--hidden-dim", type=int, default=64)
    pretrain.add_argument("--mask-fraction", type=float, default=0.25)
    pretrain.add_argument("--seed", type=int, default=42)
    pretrain.add_argument("--device", default="auto")

    rf_field = subparsers.add_parser(
        "analyze-rf-field", help="Compute bounded CIR and delay-Doppler RF features"
    )
    rf_field.add_argument("--input", required=True, help="NPZ containing a values array")
    rf_field.add_argument(
        "--representation",
        required=True,
        choices=[RFRepresentation.AMPLITUDE_PHASE.value, RFRepresentation.IQ.value],
    )
    rf_field.add_argument("--bandwidth-hz", type=float, required=True)
    rf_field.add_argument("--output")

    active = subparsers.add_parser(
        "select-active-sensing", help="Rank candidate links, channels, or beams"
    )
    active.add_argument("--candidates", required=True, help="JSON file containing a list")

    conformal = subparsers.add_parser(
        "fit-conformal", help="Fit a split-conformal residual interval"
    )
    conformal.add_argument("--truth", required=True)
    conformal.add_argument("--prediction", required=True)
    conformal.add_argument("--alpha", type=float, default=0.1)
    conformal.add_argument("--output")

    profile = subparsers.add_parser(
        "select-calibration-profile", help="Select the closest versioned calibration profile"
    )
    profile.add_argument("--profile", action="append", required=True)
    profile.add_argument("--environment-id")
    profile.add_argument("--hardware-id", action="append", default=[])
    profile.add_argument("--geometry-fingerprint")

    train = subparsers.add_parser("train", help="Run held-out waveform training")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--target", default="ecg")
    train.add_argument("--sample-rate-hz", type=float, default=20.0)
    train.add_argument("--window-seconds", type=float, default=8.0)
    train.add_argument("--stride-seconds", type=float, default=4.0)
    train.add_argument("--max-gap-s", type=float, default=0.25)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--validation-fraction", type=float, default=0.25)
    train.add_argument("--split-strategy", choices=["subject", "session", "environment", "posture", "condition", "hardware"], default="subject")
    train.add_argument("--held-out-group", action="append", default=[])
    train.add_argument("--architecture", choices=["compact", "complex_link"], default="complex_link")
    train.add_argument("--causal", action="store_true")
    train.add_argument("--device", default="auto")
    train.add_argument("--no-auxiliary", action="store_true")
    train.add_argument("--no-geometry", action="store_true")

    run_config = subparsers.add_parser(
        "run-config", help="Run a declarative experiment with baselines"
    )
    run_config.add_argument("config")
    run_config.add_argument("--output")

    inspect = subparsers.add_parser("inspect-run", help="Print a machine-readable run record")
    inspect.add_argument("run_json")

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check environment, repository, config, and optional dataset readiness"
    )
    doctor_parser.add_argument("--repo", default=".")
    doctor_parser.add_argument("--data")
    doctor_parser.add_argument("--config")

    smoke = subparsers.add_parser(
        "smoke-test", help="Run deterministic end-to-end integration checks"
    )
    smoke.add_argument("--output", default="outputs/physioatlas/innerloop-smoke")
    smoke.add_argument("--seed", type=int, default=17)
    smoke.add_argument("--subjects", type=int, default=3)
    smoke.add_argument("--duration-seconds", type=float, default=8.0)
    smoke.add_argument("--sample-rate-hz", type=float, default=10.0)

    verify = subparsers.add_parser(
        "verify-run", help="Verify a completed experiment and replay its checkpoint"
    )
    verify.add_argument("path")

    fault = subparsers.add_parser(
        "fault-test", help="Inject known failures and prove validation fails closed"
    )
    fault.add_argument("--output", default="outputs/physioatlas/fault-test")
    fault.add_argument("--seed", type=int, default=29)

    household_enroll = subparsers.add_parser(
        "household-enroll", help="Build a consent-gated open-set household registry"
    )
    household_enroll.add_argument("--data", required=True)
    household_enroll.add_argument("--registry", required=True)
    household_enroll.add_argument("--household-id", default="household-local")
    household_enroll.add_argument("--aliases", help="Optional JSON mapping subject_id to display alias")
    household_enroll.add_argument("--minimum-sessions", type=int, default=2)
    household_enroll.add_argument("--allow-without-consent", action="store_true")

    household_evaluate = subparsers.add_parser(
        "household-evaluate", help="Evaluate enrolled household identity gates"
    )
    household_evaluate.add_argument("--data", required=True)
    household_evaluate.add_argument("--registry", required=True)

    household_simulate = subparsers.add_parser(
        "household-simulate-live", help="Publish a deterministic multi-person live-feed simulation"
    )
    household_simulate.add_argument("--registry", required=True)
    household_simulate.add_argument("--state", default="outputs/physioatlas/research-hub-state.json")
    household_simulate.add_argument("--steps", type=int, default=40)
    household_simulate.add_argument("--seed", type=int, default=41)
    household_simulate.add_argument("--interval-s", type=float, default=0.0)
    household_simulate.add_argument("--no-unknown", action="store_true")

    household_smoke = subparsers.add_parser(
        "household-smoke-test", help="Verify enrollment, tracking, studies, and research hub together"
    )
    household_smoke.add_argument("--output", default="outputs/physioatlas/household-smoke")
    household_smoke.add_argument("--seed", type=int, default=73)
    household_smoke.add_argument("--members", type=int, default=4)
    household_smoke.add_argument("--sessions-per-member", type=int, default=3)

    household_fault = subparsers.add_parser(
        "household-fault-test", help="Prove household privacy and open-set gates fail closed"
    )
    household_fault.add_argument("--output", default="outputs/physioatlas/household-fault")
    household_fault.add_argument("--seed", type=int, default=97)

    research_hub = subparsers.add_parser(
        "serve-research-hub", help="Serve the household experiment and calibration hub"
    )
    research_hub.add_argument("--state", default="outputs/physioatlas/research-hub-state.json")
    research_hub.add_argument("--host", default="127.0.0.1")
    research_hub.add_argument("--port", type=int, default=8770)

    study_list = subparsers.add_parser("list-priority-studies", help="List implemented study targets")

    study_run = subparsers.add_parser("run-priority-study", help="Run one priority study protocol")
    study_run.add_argument("config")

    study_suite = subparsers.add_parser("run-priority-studies", help="Run all five priority studies")
    study_suite.add_argument("--data", required=True)
    study_suite.add_argument("--output", required=True)
    study_suite.add_argument("--sample-rate-hz", type=float, default=20.0)
    study_suite.add_argument("--minimum-sessions", type=int, default=2)

    household_init = subparsers.add_parser(
        "init-household-research", help="Create a consent-first household study workspace"
    )
    household_init.add_argument("--output", required=True)
    household_init.add_argument("--household-id", default="household-local")
    household_init.add_argument("--member", action="append", required=True, help="subject_id=Display Alias or Display Alias")

    household_jsonl = subparsers.add_parser(
        "household-replay-jsonl", help="Replay localized multi-person observations into the research hub"
    )
    household_jsonl.add_argument("--input", required=True)
    household_jsonl.add_argument("--registry", required=True)
    household_jsonl.add_argument("--state", default="outputs/physioatlas/research-hub-state.json")
    household_jsonl.add_argument("--session-id", default="household-jsonl-live")

    household_udp = subparsers.add_parser(
        "household-listen-udp", help="Receive localized multi-person observations over local UDP"
    )
    household_udp.add_argument("--registry", required=True)
    household_udp.add_argument("--state", default="outputs/physioatlas/research-hub-state.json")
    household_udp.add_argument("--host", default="127.0.0.1")
    household_udp.add_argument("--port", type=int, default=8790)
    household_udp.add_argument("--duration-s", type=float, default=30.0)
    household_udp.add_argument("--session-id", default="household-udp-live")

    csi_calibrate = subparsers.add_parser(
        "household-csi-calibrate",
        help="Build empty-room and single-person zone calibration for fixed CSI links",
    )
    csi_calibrate.add_argument("--empty", required=True)
    csi_calibrate.add_argument("--zones", required=True, help="JSON zone calibration manifest")
    csi_calibrate.add_argument("--output", required=True)
    csi_calibrate.add_argument("--environment-id", required=True)
    csi_calibrate.add_argument("--sample-rate-hz", type=float, default=20.0)
    csi_calibrate.add_argument("--minimum-links", type=int, default=3)
    csi_calibrate.add_argument(
        "--sync", help="Optional shared-packet JSONL capture with transmitter sequence IDs"
    )

    csi_process = subparsers.add_parser(
        "household-csi-process",
        help="Produce conservative zone-level people observations from synchronized mixed CSI",
    )
    csi_process.add_argument("--input", required=True)
    csi_process.add_argument("--calibration", required=True)
    csi_process.add_argument("--output", required=True)
    csi_process.add_argument("--config", help="Optional JSON CsiProcessingConfig")
    csi_process.add_argument(
        "--require-band",
        action="append",
        default=[],
        choices=["2.4ghz", "5ghz", "6ghz"],
    )

    hub_worker = subparsers.add_parser(
        "research-hub-worker", help="Process safe actions queued by the local research hub"
    )
    hub_worker.add_argument("--state", default="outputs/physioatlas/research-hub-state.json")
    hub_worker.add_argument("--data")
    hub_worker.add_argument("--registry")
    hub_worker.add_argument("--output", default="outputs/physioatlas/research-hub-actions")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init-household-research":
            _emit(initialize_household_workspace(
                args.output, household_id=args.household_id, members=args.member
            ))
            return 0
        if args.command == "household-replay-jsonl":
            _emit(run_household_jsonl(
                args.input, args.registry, args.state, session_id=args.session_id
            ))
            return 0
        if args.command == "household-listen-udp":
            _emit(run_household_udp(
                args.registry, args.state, host=args.host, port=args.port,
                duration_s=args.duration_s, session_id=args.session_id
            ))
            return 0
        if args.command == "household-csi-calibrate":
            zone_manifest_path = Path(args.zones)
            zone_manifest = json.loads(zone_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(zone_manifest, dict):
                raise ValueError("Zone manifest must be a JSON object keyed by zone ID")
            zone_captures = {}
            for zone_id, item in zone_manifest.items():
                if not isinstance(item, dict) or "center_m" not in item or "capture" not in item:
                    raise ValueError(f"Zone {zone_id!r} requires center_m and capture")
                capture = Path(str(item["capture"]))
                if not capture.is_absolute():
                    capture = zone_manifest_path.parent / capture
                zone_captures[str(zone_id)] = (list(item["center_m"]), capture)
            _emit(build_household_csi_calibration(
                args.empty,
                zone_captures,
                args.output,
                environment_id=args.environment_id,
                sample_rate_hz=args.sample_rate_hz,
                minimum_links=args.minimum_links,
                synchronization_capture=args.sync,
            ))
            return 0
        if args.command == "household-csi-process":
            config_payload = {}
            if args.config:
                config_payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
            if args.require_band:
                config_payload["require_frequency_bands"] = args.require_band
            _emit(process_mixed_household_csi(
                args.input,
                args.calibration,
                args.output,
                config=CsiProcessingConfig.model_validate(config_payload),
            ))
            return 0
        if args.command == "research-hub-worker":
            result = ResearchHubActionWorker(
                ResearchHubStore(args.state), dataset_root=args.data,
                registry_path=args.registry, output_root=args.output
            ).process_once()
            _emit(result)
            return 0 if all(item.get("status") != "error" for item in result["results"]) else 2
        if args.command == "household-enroll":
            aliases = {}
            if args.aliases:
                candidate = Path(args.aliases)
                aliases = json.loads(candidate.read_text(encoding="utf-8")) if candidate.exists() else json.loads(args.aliases)
                if not isinstance(aliases, dict):
                    raise ValueError("aliases must be a JSON object")
            result = enroll_household(
                args.data,
                args.registry,
                household_id=args.household_id,
                aliases={str(key): str(value) for key, value in aliases.items()},
                require_consent=not args.allow_without_consent,
                minimum_sessions=args.minimum_sessions,
            )
            _emit(result)
            return 0 if result["ready"] else 2
        if args.command == "household-evaluate":
            result = evaluate_household_registry(args.data, args.registry)
            _emit(result)
            return 0 if result["valid"] else 2
        if args.command == "household-simulate-live":
            result = simulate_household_live(
                args.registry,
                args.state,
                steps=args.steps,
                seed=args.seed,
                interval_s=args.interval_s,
                include_unknown=not args.no_unknown,
            )
            _emit(result)
            return 0 if result["valid"] else 2
        if args.command == "household-smoke-test":
            result = run_household_smoke_test(
                args.output,
                seed=args.seed,
                members=args.members,
                sessions_per_member=args.sessions_per_member,
            )
            _emit({
                "status": result["status"],
                "report_path": result.get("report_path"),
                "checks": result.get("checks", {}),
            })
            return 0 if result["status"] == "passed" else 2
        if args.command == "household-fault-test":
            result = run_household_fault_test(args.output, seed=args.seed)
            _emit(result)
            return 0 if result["status"] == "passed" else 2
        if args.command == "serve-research-hub":
            server = ResearchHubServer(ResearchHubStore(args.state), host=args.host, port=args.port)
            host, port = server.address
            print(f"PhysioAtlas research hub: http://{host}:{port}")
            server.serve_forever()
            return 0
        if args.command == "list-priority-studies":
            _emit(list_priority_studies())
            return 0
        if args.command == "run-priority-study":
            result = run_priority_study(load_study_protocol(args.config))
            _emit(result)
            return 0 if result["valid"] else 2
        if args.command == "run-priority-studies":
            result = run_all_priority_studies(
                args.data,
                args.output,
                sample_rate_hz=args.sample_rate_hz,
                minimum_sessions=args.minimum_sessions,
            )
            _emit(result)
            return 0 if result["valid"] else 2
        if args.command == "synthesize":
            paths = create_synthetic_cohort(
                args.output,
                subjects=args.subjects,
                sessions_per_subject=args.sessions_per_subject,
                duration_seconds=args.duration_seconds,
                sample_rate_hz=args.sample_rate_hz,
                seed=args.seed,
            )
            _emit({"created": len(paths), "manifests": [str(path) for path in paths]})
            return 0
        if args.command == "validate":
            result = validate_dataset(args.data, require_consent=args.require_consent)
            _emit(result)
            return 0 if result["valid"] else 2
        if args.command == "list-adapters":
            _emit(DEFAULT_REGISTRY.describe())
            return 0
        if args.command == "acquire":
            payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            _emit(run_acquisition(AcquisitionConfig.model_validate(payload)))
            return 0
        if args.command == "convert-ruview":
            _emit(ruview_jsonl_to_npz(args.input, args.output))
            return 0
        if args.command == "convert-csv":
            _emit(csv_to_npz(args.input, args.output, timestamp_column=args.timestamp_column, value_columns=args.value_columns, timestamp_scale=args.timestamp_scale))
            return 0
        if args.command == "extract-ultrasound":
            _emit(ultrasound_npz_to_displacement(args.input, args.output, pixel_spacing_m=(args.axial_spacing_m, args.lateral_spacing_m)))
            return 0
        if args.command == "calibrate-clock":
            calibration = estimate_clock_calibration(
                _load_vector(args.source_events),
                _load_vector(args.reference_events),
                source_clock=args.source_clock,
                reference_clock=args.reference_clock,
            )
            save_calibration(args.output, calibration)
            _emit(calibration.__dict__)
            return 0
        if args.command == "calibrate-geometry":
            source = np.load(args.source_points, allow_pickle=False) if Path(args.source_points).suffix == ".npy" else np.loadtxt(args.source_points, delimiter=",")
            target = np.load(args.target_points, allow_pickle=False) if Path(args.target_points).suffix == ".npy" else np.loadtxt(args.target_points, delimiter=",")
            transform = estimate_rigid_transform(source, target, source_frame=args.source_frame, target_frame=args.target_frame)
            save_calibration(args.output, transform)
            _emit(transform.__dict__)
            return 0
        if args.command == "analyze-propagation":
            result = analyze_dataset_propagation(args.data, sample_rate_hz=args.sample_rate_hz)
            _emit(result)
            return 0 if result["valid"] else 2
        if args.command == "privacy-audit":
            result = audit_dataset_consent(args.data, require_consent=args.require_consent)
            _emit(result)
            return 0 if result["valid"] else 2
        if args.command == "privacy-export":
            _emit(create_privacy_export(args.data, args.output, secret=args.secret, drop_modalities=[Modality(item) for item in args.drop_modality]))
            return 0
        if args.command == "serve-dashboard":
            server = DashboardServer(LiveStateStore(args.state), host=args.host, port=args.port)
            host, port = server.address
            print(f"PhysioAtlas dashboard: http://{host}:{port}")
            server.serve_forever()
            return 0
        if args.command == "publish-dashboard":
            candidate = Path(args.json)
            payload = json.loads(candidate.read_text(encoding="utf-8")) if candidate.exists() else json.loads(args.json)
            store = LiveStateStore(args.state)
            store.publish(payload)
            _emit(store.read())
            return 0
        if args.command == "pretrain-rf":
            examples = load_windows_from_root(
                args.data,
                sample_rate_hz=args.sample_rate_hz,
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                target_key=args.target_for_windowing,
                max_gap_s=args.max_gap_s,
            )
            _emit(train_rf_foundation(
                examples,
                args.output,
                RFPretrainConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    hidden_dim=args.hidden_dim,
                    mask_fraction=args.mask_fraction,
                    seed=args.seed,
                    device=args.device,
                ),
            ))
            return 0
        if args.command == "analyze-rf-field":
            with np.load(args.input, allow_pickle=False) as archive:
                if "values" not in archive:
                    raise ValueError("RF NPZ must contain a values array")
                values = np.asarray(archive["values"])
            complex_csi = canonical_to_complex(values, args.representation)
            cir = reconstruct_cir(complex_csi, n_delay_bins=complex_csi.shape[-1] * 2)
            features = delay_spread_features(cir, delay_resolution_s=1.0 / args.bandwidth_hz)
            time_subcarrier = complex_csi.reshape(-1, complex_csi.shape[-1])
            delay_doppler = delay_doppler_map(
                time_subcarrier, n_delay_bins=complex_csi.shape[-1] * 2
            )
            if args.output:
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination,
                    cir_real=cir.real.astype(np.float32),
                    cir_imag=cir.imag.astype(np.float32),
                    delay_doppler=delay_doppler,
                )
            _emit({
                "research_only": True,
                "physical_resolution_note": "IFFT features do not exceed the bandwidth-limited range resolution.",
                "features": features,
                "cir_shape": list(cir.shape),
                "delay_doppler_shape": list(delay_doppler.shape),
                "output": args.output,
            })
            return 0
        if args.command == "select-active-sensing":
            payload = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("Candidate file must contain a JSON list")
            decision = select_active_measurement(
                [ActiveSensingCandidate(**item) for item in payload]
            )
            _emit(decision.to_dict())
            return 0
        if args.command == "fit-conformal":
            truth = _load_vector(args.truth)
            prediction = _load_vector(args.prediction)
            calibration = fit_split_conformal(truth, prediction, alpha=args.alpha)
            lower, upper = conformal_interval(prediction, calibration)
            result = {
                "calibration": calibration.to_dict(),
                "empirical_coverage": interval_coverage(truth, lower, upper),
            }
            if args.output:
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(make_json_safe(result), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            _emit(result)
            return 0
        if args.command == "select-calibration-profile":
            profiles = [load_profile(path) for path in args.profile]
            geometry = _load_vector(args.geometry_fingerprint) if args.geometry_fingerprint else None
            selected, detail = select_calibration_profile(
                profiles,
                environment_id=args.environment_id,
                hardware_ids=args.hardware_id,
                geometry_fingerprint=geometry,
            )
            _emit({"selected": selected.model_dump(mode="json"), "selection": detail})
            return 0
        if args.command == "train":
            examples = load_windows_from_root(
                args.data,
                sample_rate_hz=args.sample_rate_hz,
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                target_key=args.target,
                max_gap_s=args.max_gap_s,
            )
            result = train_model(
                examples,
                args.output,
                TrainConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    hidden_dim=args.hidden_dim,
                    layers=args.layers,
                    seed=args.seed,
                    validation_fraction=args.validation_fraction,
                    split_strategy=args.split_strategy,
                    held_out_groups=tuple(args.held_out_group),
                    architecture=args.architecture,
                    causal=args.causal,
                    device=args.device,
                    use_auxiliary=not args.no_auxiliary,
                    use_geometry=not args.no_geometry,
                ),
            )
            _emit(result)
            return 0
        if args.command == "run-config":
            _emit(run_experiment(args.config, args.output))
            return 0
        if args.command == "inspect-run":
            _emit(json.loads(Path(args.run_json).read_text(encoding="utf-8")))
            return 0
        if args.command == "doctor":
            result = doctor(args.repo, data_root=args.data, config_path=args.config)
            _emit(result)
            return 0 if result["ready"] else 2
        if args.command == "smoke-test":
            result = run_smoke_test(args.output, seed=args.seed, subjects=args.subjects, duration_seconds=args.duration_seconds, sample_rate_hz=args.sample_rate_hz)
            experiment = result.get("experiment", {})
            neural = experiment.get("neural", {})
            _emit({
                "status": result["status"],
                "report_path": result.get("report_path"),
                "run_root": result.get("verification", {}).get("run_root"),
                "dataset_sha256": experiment.get("dataset_sha256"),
                "run_id": experiment.get("run_id"),
                "final_metrics": neural.get("final_metrics", {}),
                "checkpoint_sha256": neural.get("checkpoint_sha256"),
                "integrations_valid": result.get("integrations", {}).get("valid"),
                "doctor_ready": result.get("doctor", {}).get("ready"),
                "run_valid": result.get("verification", {}).get("valid"),
            })
            return 0 if result["status"] == "passed" else 2
        if args.command == "verify-run":
            result = verify_run(args.path)
            _emit(result)
            return 0 if result["valid"] else 2
        if args.command == "fault-test":
            result = run_fault_injection_test(args.output, seed=args.seed)
            _emit({"status": result["status"], "report_path": result.get("report_path"), "cases": [{"name": case["name"], "detected": case["detected"]} for case in result.get("cases", [])]})
            return 0 if result["status"] == "passed" else 2
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
