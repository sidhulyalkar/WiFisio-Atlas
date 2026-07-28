from __future__ import annotations

import hashlib
import importlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import yaml

from .acquisition import AcquisitionConfig, AdapterSpec, run_acquisition
from .calibration import estimate_clock_calibration, estimate_rigid_transform
from .calibration_memory import (
    CalibrationProfile,
    load_profile,
    save_profile,
    select_calibration_profile,
)
from .dashboard import LiveStateStore
from .distillation import privileged_distillation_loss
from .experiment import load_experiment_config, run_experiment
from .federated import aggregate_model_deltas
from .model import PhysioAtlasModel
from .dataset import load_windows_from_root
from .pretraining import RFPretrainConfig, train_rf_foundation
from .physiology import analyze_dataset_propagation
from .rf_field import (
    ActiveSensingCandidate,
    canonical_to_complex,
    delay_doppler_map,
    delay_spread_features,
    reconstruct_cir,
    select_active_measurement,
)
from .privacy import HashChainAuditLog, audit_dataset_consent, create_privacy_export
from .schema import Modality, RFRepresentation, SessionManifest
from .synthetic import create_synthetic_cohort
from .ultrasound import extract_displacement_trace
from .uncertainty import conformal_interval, fit_split_conformal, interval_coverage, should_abstain
from .io import load_stream
from .utils import make_json_safe
from .validation import validate_dataset

_REQUIRED_MODULES = ("numpy", "pydantic", "yaml", "sklearn", "scipy", "torch")
_REQUIRED_NULLS = ("null_ridge_rf_time_shift", "null_ridge_rf_label_permutation")


def _check(name: str, passed: bool, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else ("fail" if required else "warn"),
        "required": required,
        "detail": detail,
    }


def _version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def collect_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for module_name in _REQUIRED_MODULES:
        try:
            packages[module_name] = _version(module_name)
        except Exception as exc:
            packages[module_name] = f"unavailable: {exc}"
    torch_info: dict[str, Any] = {"available": False, "device_count": 0, "devices": []}
    try:
        torch_info["available"] = bool(torch.cuda.is_available())
        torch_info["device_count"] = int(torch.cuda.device_count())
        torch_info["devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except Exception:
        pass
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "working_directory": str(Path.cwd()),
        "packages": packages,
        "cuda": torch_info,
    }


def doctor(
    repo_root: Union[str, Path] = ".",
    *,
    data_root: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "python_version",
            sys.version_info >= (3, 9),
            f"Python {sys.version.split()[0]} (requires >=3.9)",
        )
    )
    for module_name in _REQUIRED_MODULES:
        try:
            checks.append(_check(f"import_{module_name}", True, f"version {_version(module_name)}"))
        except Exception as exc:
            checks.append(_check(f"import_{module_name}", False, str(exc)))
    expected_paths = (
        "physioatlas",
        "tests/physioatlas",
        "configs/physioatlas",
        "dashboard/physioatlas/index.html",
        "INNERLOOP.md",
        "PROJECT_GOALS_PHYSIOATLAS.md",
    )
    for relative in expected_paths:
        path = root / relative
        checks.append(_check(f"repo_path:{relative}", path.exists(), str(path)))
    for module_name in (
        "acquisition",
        "calibration",
        "physiology",
        "ultrasound",
        "privacy",
        "federated",
        "dashboard",
        "splits",
        "pretraining",
        "uncertainty",
        "rf_field",
        "distillation",
        "calibration_memory",
        "household",
        "tracking",
        "household_live",
        "household_csi",
        "household_csi_input",
        "household_csi_schema",
        "household_bfi",
        "household_bfi_schema",
        "rf_fusion",
        "rf_fusion_schema",
        "rf_fusion_evaluation",
        "calibration_reflector",
        "calibration_reflector_schema",
        "reflector_rf_atlas",
        "reflector_rf_atlas_schema",
        "rf_hub",
        "rf_cli",
        "household_verification",
        "studies",
        "research_hub",
        "household_protocol",
        "research_hub_worker",
    ):
        path = root / "physioatlas" / f"{module_name}.py"
        checks.append(_check(f"integration_module:{module_name}", path.exists(), str(path)))
    writable_root = root / "outputs" / "physioatlas"
    try:
        writable_root.mkdir(parents=True, exist_ok=True)
        probe = writable_root / ".write-probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        checks.append(_check("output_writable", True, str(writable_root)))
    except Exception as exc:
        checks.append(_check("output_writable", False, str(exc)))
    checks.append(
        _check(
            "git_available",
            bool(shutil.which("git")),
            shutil.which("git") or "git not found on PATH",
            required=False,
        )
    )
    if config_path is not None:
        try:
            config = load_experiment_config(config_path)
            checks.append(
                _check(
                    "experiment_config",
                    isinstance(config.get("data"), dict)
                    and isinstance(config.get("training"), dict)
                    and isinstance(config.get("split"), dict),
                    f"loaded {Path(config_path).resolve()}",
                )
            )
        except Exception as exc:
            checks.append(_check("experiment_config", False, str(exc)))
    dataset_report = None
    if data_root is not None:
        dataset_report = validate_dataset(data_root)
        checks.append(
            _check(
                "dataset_contract",
                bool(dataset_report["valid"]),
                f"{dataset_report['sessions']} sessions / {dataset_report['subjects']} subjects",
            )
        )
    required_failures = [item for item in checks if item["required"] and item["status"] == "fail"]
    warnings = [item for item in checks if item["status"] == "warn"]
    return make_json_safe(
        {
            "schema_version": "physioatlas.doctor.v2",
            "ready": not required_failures,
            "repo_root": str(root),
            "checks": checks,
            "required_failures": len(required_failures),
            "warnings": len(warnings),
            "environment": collect_environment(),
            "dataset": dataset_report,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _checkpoint_replay(checkpoint: Path) -> tuple[bool, str]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_config = dict(payload["model_config"])
        model = PhysioAtlasModel(**model_config)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        time_steps = 4
        rf_features = int(model_config["rf_features"])
        channels = int(model_config.get("linked_rf_channels") or 1)
        links = max(1, rf_features // channels)
        with torch.no_grad():
            output = model(
                torch.zeros(1, time_steps, rf_features),
                torch.zeros(1, time_steps, int(model_config["auxiliary_features"])),
                torch.zeros(1, int(model_config["geometry_features"])),
                torch.zeros(1, time_steps, links, channels),
                torch.zeros(1, links, 12),
                torch.ones(1, links),
                torch.ones(1, time_steps, links, 1),
            )
        finite = all(
            torch.isfinite(item).all().item()
            for item in (output.waveform, output.observability, output.log_variance)
        )
        return bool(finite), f"waveform_shape={tuple(output.waveform.shape)}"
    except Exception as exc:
        return False, str(exc)


def verify_run(path: Union[str, Path]) -> dict[str, Any]:
    supplied = Path(path).resolve()
    if supplied.is_dir():
        run_root = supplied
        experiment_path = run_root / "experiment.json"
    else:
        experiment_path = supplied
        run_root = supplied.parent
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    if not experiment_path.exists():
        return {
            "schema_version": "physioatlas.run-verification.v2",
            "valid": False,
            "run_root": str(run_root),
            "checks": [],
            "errors": [f"Missing {experiment_path}"],
            "warnings": [],
        }
    experiment = _load_json(experiment_path)
    neural_path = run_root / "neural" / "run.json"
    neural_file_present = neural_path.exists()
    neural = _load_json(neural_path) if neural_file_present else experiment.get("neural", {})

    def add(name: str, passed: bool, detail: str, *, required: bool = True) -> None:
        checks.append(_check(name, passed, detail, required=required))
        if not passed:
            (errors if required else warnings).append(f"{name}: {detail}")

    add(
        "experiment_schema",
        experiment.get("schema_version") in {"physioatlas.experiment.v1", "physioatlas.experiment.v2"},
        str(experiment.get("schema_version")),
    )
    add("dataset_valid", bool(experiment.get("dataset_validation", {}).get("valid")), "dataset validation gate")
    add("neural_record_file", neural_file_present, str(neural_path))
    add("neural_record_present", isinstance(neural, dict) and bool(neural), "parsed neural record")
    add("top_level_research_only", experiment.get("research_only") is True, str(experiment.get("research_only")))
    add("top_level_clinical_claim_disabled", experiment.get("clinical_claim_allowed") is False, str(experiment.get("clinical_claim_allowed")))
    add("observability_required", experiment.get("observability_required") is True or experiment.get("schema_version") == "physioatlas.experiment.v1", str(experiment.get("observability_required")))
    add("run_completed", neural.get("status") == "completed", str(neural.get("status")))
    add("research_only", neural.get("research_only") is True, str(neural.get("research_only")))
    add("clinical_claim_disabled", neural.get("clinical_claim_allowed") is False, str(neural.get("clinical_claim_allowed")))

    train_groups = set(neural.get("train_groups", neural.get("train_subjects", [])))
    validation_groups = set(neural.get("validation_groups", neural.get("validation_subjects", [])))
    add(
        "declared_group_disjoint",
        bool(train_groups) and bool(validation_groups) and train_groups.isdisjoint(validation_groups),
        f"train={sorted(train_groups)} validation={sorted(validation_groups)}",
    )
    if neural.get("split_strategy", "subject") == "subject":
        train_subjects = set(neural.get("train_subjects", []))
        validation_subjects = set(neural.get("validation_subjects", []))
        add(
            "subject_disjoint",
            bool(train_subjects) and bool(validation_subjects) and train_subjects.isdisjoint(validation_subjects),
            f"train={sorted(train_subjects)} validation={sorted(validation_subjects)}",
        )

    checkpoint_name = neural.get("checkpoint")
    checkpoint = run_root / "neural" / str(checkpoint_name) if checkpoint_name else None
    add("checkpoint_declared", bool(checkpoint_name), str(checkpoint_name))
    add("checkpoint_exists", bool(checkpoint and checkpoint.exists()), str(checkpoint))
    if checkpoint and checkpoint.exists():
        actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        expected_hash = neural.get("checkpoint_sha256")
        add("checkpoint_hash", actual_hash == expected_hash, f"expected={expected_hash} actual={actual_hash}")
        replay_ok, replay_detail = _checkpoint_replay(checkpoint)
        add("checkpoint_replay", replay_ok, replay_detail)

    final_metrics = neural.get("final_metrics", {})
    add("metrics_finite", _all_finite(final_metrics), "neural final metrics")
    add("experiment_json_finite", _all_finite(experiment), "all serialized numeric values")
    if "interval_90_coverage" in final_metrics:
        coverage = float(final_metrics["interval_90_coverage"])
        add("uncertainty_coverage_range", 0.0 <= coverage <= 1.0, str(coverage))
    if "mean_observability" in final_metrics:
        observability = float(final_metrics["mean_observability"])
        add("observability_range", 0.0 <= observability <= 1.0, str(observability))

    baselines = experiment.get("baselines", {})
    for null_name in _REQUIRED_NULLS:
        add(f"null_present:{null_name}", null_name in baselines, null_name)
    add("baseline_metrics_finite", _all_finite(baselines), f"{len(baselines) if isinstance(baselines, dict) else 0} baselines")
    add("privacy_audit_present", isinstance(experiment.get("privacy_audit"), dict), "privacy audit record", required=False)
    add("propagation_analysis_present", isinstance(experiment.get("propagation_analysis"), dict), "propagation analysis record", required=False)

    config_snapshot = run_root / "config.snapshot.yaml"
    add("config_snapshot", config_snapshot.exists(), str(config_snapshot))
    if config_snapshot.exists():
        try:
            config = yaml.safe_load(config_snapshot.read_text(encoding="utf-8"))
            add("hypothesis_present", bool(str(config.get("hypothesis", "")).strip()), "config hypothesis")
            add("claim_boundary_present", bool(str(config.get("claim_boundary", "")).strip()), "config claim boundary")
        except Exception as exc:
            add("config_snapshot_parse", False, str(exc))

    return make_json_safe(
        {
            "schema_version": "physioatlas.run-verification.v2",
            "valid": not errors,
            "run_root": str(run_root),
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "split_strategy": neural.get("split_strategy"),
                "train_groups": sorted(train_groups),
                "validation_groups": sorted(validation_groups),
                "final_metrics": final_metrics,
                "checkpoint_sha256": neural.get("checkpoint_sha256"),
            },
        }
    )


def _run_integration_checks(output: Path, data_root: Path, seed: int, sample_rate_hz: float) -> dict[str, Any]:
    # Clock calibration with a known 80 ppm drift and 30 ms offset.
    source = np.arange(0.0, 10.0, 1.0)
    reference = (1.0 + 80e-6) * source + 0.03
    clock = estimate_clock_calibration(source, reference)
    # Rigid calibration with a known translation.
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    rigid = estimate_rigid_transform(points, points + np.asarray([0.1, -0.2, 0.3]))

    first_session = next(data_root.rglob("manifest.json")).parent
    acquisition = run_acquisition(
        AcquisitionConfig(
            session_id="acquisition-smoke",
            subject_id="test-subject",
            protocol="adapter_contract_smoke",
            output_dir=str(output / "acquisition"),
            geometry_path=str(first_session / "geometry.json"),
            duration_seconds=4.0,
            environment_id="test-room",
            adapters=[
                AdapterSpec(
                    adapter_type="synthetic_wave",
                    sensor_id="synthetic-rf",
                    modality=Modality.WIFI_CSI,
                    output_name="synthetic_rf.npz",
                    sample_rate_hz=sample_rate_hz,
                    representation=RFRepresentation.FEATURES,
                    parameters={"seed": seed, "features": 4, "frequency_hz": 0.25},
                )
            ],
        )
    )

    image = np.zeros((32, 32), dtype=np.float32)
    image[10:16, 12:18] = 1.0
    frames = np.stack([np.roll(image, shift=index, axis=0) for index in range(4)])
    displacement = extract_displacement_trace(frames, pixel_spacing_m=(0.001, 0.001))

    privacy = audit_dataset_consent(data_root, require_consent=True)
    privacy_export = create_privacy_export(
        data_root,
        output / "privacy-export",
        secret="innerloop-test-secret",
        drop_modalities=(Modality.POSE,),
    )
    audit = HashChainAuditLog(output / "audit.jsonl")
    audit.append("integration_start", {"seed": seed})
    audit.append("integration_complete", {"valid": True})
    audit_verification = audit.verify()

    federated = aggregate_model_deltas(
        [
            {"weight": torch.tensor([1.0, 2.0])},
            {"weight": torch.tensor([3.0, 4.0])},
        ],
        weights=[1.0, 3.0],
        clip_norm=10.0,
    )
    store = LiveStateStore(output / "dashboard-state.json")
    store.publish(
        {
            "status": "integration-verified",
            "session_id": "synthetic",
            "observability": 0.75,
            "uncertainty": "synthetic-only",
        }
    )
    propagation = analyze_dataset_propagation(data_root, sample_rate_hz=sample_rate_hz)
    pretrain_examples = load_windows_from_root(
        data_root,
        sample_rate_hz=sample_rate_hz,
        window_seconds=4.0,
        stride_seconds=2.0,
        target_key="ecg",
    )
    pretraining = train_rf_foundation(
        pretrain_examples,
        output / "rf-pretraining",
        RFPretrainConfig(
            epochs=1,
            batch_size=4,
            learning_rate=0.001,
            hidden_dim=16,
            mask_fraction=0.25,
            seed=seed,
            device="cpu",
        ),
    )

    manifest = SessionManifest.model_validate(
        json.loads((first_session / "manifest.json").read_text(encoding="utf-8"))
    )
    wifi_reference = next(stream for stream in manifest.streams if stream.modality == Modality.WIFI_CSI)
    wifi_stream = load_stream(wifi_reference.model_copy(update={"path": str(first_session / wifi_reference.path)}))
    complex_csi = canonical_to_complex(wifi_stream.values, wifi_reference.representation)
    cir = reconstruct_cir(complex_csi, n_delay_bins=complex_csi.shape[-1] * 2)
    field_features = delay_spread_features(cir, delay_resolution_s=2.5e-8)
    delay_doppler = delay_doppler_map(
        complex_csi[: min(32, complex_csi.shape[0])],
        n_delay_bins=complex_csi.shape[-1] * 2,
    )
    active_decision = select_active_measurement(
        [
            ActiveSensingCandidate("link-0", 0.3, 0.2, novelty=0.1, energy_cost=0.1),
            ActiveSensingCandidate("link-1", 0.5, 0.4, novelty=0.2, energy_cost=0.2),
        ]
    )

    truth = np.linspace(-1.0, 1.0, 40)
    prediction = truth + 0.05 * np.sin(np.arange(40))
    conformal = fit_split_conformal(truth[:20], prediction[:20], alpha=0.1)
    lower, upper = conformal_interval(prediction[20:], conformal)
    conformal_coverage = interval_coverage(truth[20:], lower, upper)
    abstention = should_abstain(
        observability=0.9,
        interval_width=float(np.mean(upper - lower)),
        max_interval_width=1.0,
    )

    student = torch.nn.functional.normalize(torch.tensor([[1.0, 0.2], [0.4, 1.0]]), dim=-1)
    teacher = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    distillation_loss, distillation_report = privileged_distillation_loss(student, teacher)

    profile_path = output / "calibration-profile.json"
    profile = CalibrationProfile(
        profile_id="smoke-room-profile",
        environment_id="test-room",
        hardware_ids=["synthetic-rf"],
        geometry_fingerprint=[0.0, 1.0, 2.0],
        uncertainty_profile={"conformal_radius": conformal.residual_quantile},
    )
    save_profile(profile_path, profile)
    selected_profile, selection_detail = select_calibration_profile(
        [load_profile(profile_path)],
        environment_id="test-room",
        hardware_ids=["synthetic-rf"],
        geometry_fingerprint=np.asarray([0.0, 1.0, 2.0]),
    )
    return make_json_safe(
        {
            "clock": clock.__dict__,
            "rigid": rigid.__dict__,
            "acquisition": acquisition,
            "ultrasound": {
                "final_displacement_m": displacement[-1].tolist(),
                "finite": bool(np.isfinite(displacement).all()),
            },
            "privacy": privacy,
            "privacy_export": privacy_export,
            "audit_log": audit_verification,
            "federated": {"weight": federated["weight"].tolist()},
            "dashboard_state": store.read(),
            "propagation": propagation,
            "rf_pretraining": pretraining,
            "rf_field": {
                "features": field_features,
                "delay_doppler_shape": list(delay_doppler.shape),
                "active_decision": active_decision.to_dict(),
            },
            "uncertainty": {
                "calibration": conformal.to_dict(),
                "coverage": conformal_coverage,
                "abstain": abstention,
            },
            "distillation": distillation_report.to_dict(),
            "calibration_memory": {
                "selected_profile": selected_profile.profile_id,
                "selection": selection_detail,
            },
            "valid": (
                abs(clock.drift_ppm - 80.0) < 1e-3
                and clock.rmse_s < 1e-9
                and rigid.rmse_m < 1e-9
                and privacy["valid"]
                and audit_verification["valid"]
                and np.isfinite(displacement).all()
                and propagation["valid"]
                and pretraining["status"] == "completed"
                and len(pretraining["checkpoint_sha256"]) == 64
                and np.isfinite(delay_doppler).all()
                and field_features["dominant_tap_ratio"] > 0.0
                and active_decision.candidate_id == "link-1"
                and 0.0 <= conformal_coverage <= 1.0
                and not abstention
                and torch.isfinite(distillation_loss).item()
                and selected_profile.profile_id == "smoke-room-profile"
            ),
        }
    )


def run_smoke_test(
    output_dir: Union[str, Path],
    *,
    seed: int = 17,
    subjects: int = 3,
    duration_seconds: float = 8.0,
    sample_rate_hz: float = 10.0,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    data_root = output / "synthetic-data"
    run_root = output / "run"
    config_path = output / "smoke-config.yaml"
    create_synthetic_cohort(
        data_root,
        subjects=subjects,
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        seed=seed,
    )
    config = {
        "name": "physioatlas-innerloop-smoke",
        "hypothesis": "The complete research pipeline can ingest, validate, calibrate, split, train, estimate uncertainty, compare nulls, checkpoint, and verify a deterministic multimodal cohort.",
        "claim_boundary": "Infrastructure verification only; no biological, anatomical, diagnostic, or clinical claim.",
        "data": {
            "root": str(data_root),
            "target": "ecg",
            "sample_rate_hz": sample_rate_hz,
            "window_seconds": 4.0,
            "stride_seconds": 2.0,
            "max_gap_s": 0.25,
            "require_consent": True,
        },
        "split": {"strategy": "subject", "validation_fraction": 0.34},
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "hidden_dim": 16,
            "layers": 1,
            "seed": seed,
            "device": "cpu",
            "use_auxiliary": True,
            "use_geometry": True,
            "architecture": "complex_link",
            "causal": False,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    doctor_report = doctor(Path.cwd(), data_root=data_root, config_path=config_path)
    integrations = _run_integration_checks(output / "integrations", data_root, seed, sample_rate_hz)
    experiment = run_experiment(config_path, run_root)
    verification = verify_run(run_root)
    passed = doctor_report["ready"] and integrations["valid"] and verification["valid"]
    report = make_json_safe(
        {
            "schema_version": "physioatlas.smoke.v2",
            "status": "passed" if passed else "failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output),
            "doctor": doctor_report,
            "integrations": integrations,
            "experiment": experiment,
            "verification": verification,
        }
    )
    report_path = output / "smoke-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def run_fault_injection_test(
    output_dir: Union[str, Path],
    *,
    seed: int = 29,
) -> dict[str, Any]:
    """Prove that integrity, consent, clock, and representation gates fail closed."""
    output = Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = run_smoke_test(output / "baseline", seed=seed)
    cases: list[dict[str, Any]] = []

    corrupt_run = output / "faults" / "checkpoint-corruption"
    shutil.copytree(Path(baseline["verification"]["run_root"]), corrupt_run, dirs_exist_ok=True)
    checkpoint = corrupt_run / "neural" / "model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"physioatlas-fault-injection")
    corrupt_verification = verify_run(corrupt_run)
    cases.append(
        {
            "name": "checkpoint_corruption",
            "detected": not corrupt_verification["valid"] and any("checkpoint_hash" in error for error in corrupt_verification["errors"]),
            "verification": corrupt_verification,
        }
    )

    def cloned_data(name: str) -> Path:
        destination = output / "faults" / name
        shutil.copytree(output / "baseline" / "synthetic-data", destination, dirs_exist_ok=True)
        return destination

    corrupt_data = cloned_data("non_monotonic_timestamps")
    stream_path = next(corrupt_data.rglob("wifi_csi.npz"))
    with np.load(stream_path, allow_pickle=False) as archive:
        timestamps = np.asarray(archive["timestamps_s"]).copy()
        values = np.asarray(archive["values"]).copy()
        quality = np.asarray(archive["quality"]).copy()
    timestamps[1] = timestamps[0]
    np.savez_compressed(stream_path, timestamps_s=timestamps, values=values, quality=quality)
    report = validate_dataset(corrupt_data)
    cases.append(
        {
            "name": "non_monotonic_timestamps",
            "detected": not report["valid"] and any("strictly increasing" in error for error in report["errors"]),
            "validation": report,
        }
    )

    expired = cloned_data("expired_consent")
    consent_path = next(expired.rglob("consent.json"))
    consent = _load_json(consent_path)
    consent["expires_at_utc"] = "2020-01-01T00:00:00+00:00"
    consent_path.write_text(json.dumps(consent, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = validate_dataset(expired, require_consent=True)
    cases.append(
        {
            "name": "expired_consent",
            "detected": not report["valid"] and any("expired" in error.lower() for error in report["errors"]),
            "validation": report,
        }
    )

    drift = cloned_data("excessive_clock_drift")
    calibration_path = next(drift.rglob("calibration.json"))
    calibration = _load_json(calibration_path)
    calibration["clocks"]["reference"]["drift_ppm"] = 5000.0
    calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = validate_dataset(drift)
    cases.append(
        {
            "name": "excessive_clock_drift",
            "detected": not report["valid"] and any("clock drift" in error.lower() for error in report["errors"]),
            "validation": report,
        }
    )

    odd_rf = cloned_data("invalid_rf_representation")
    stream_path = next(odd_rf.rglob("wifi_csi.npz"))
    with np.load(stream_path, allow_pickle=False) as archive:
        timestamps = np.asarray(archive["timestamps_s"])
        values = np.asarray(archive["values"])[:, :-1]
        quality = np.asarray(archive["quality"])
    np.savez_compressed(stream_path, timestamps_s=timestamps, values=values, quality=quality)
    report = validate_dataset(odd_rf)
    cases.append(
        {
            "name": "invalid_rf_representation",
            "detected": not report["valid"] and any("odd feature count" in error for error in report["errors"]),
            "validation": report,
        }
    )

    audit = HashChainAuditLog(output / "faults" / "audit-tamper.jsonl")
    audit.append("first", {"value": 1})
    audit.append("second", {"value": 2})
    text = audit.path.read_text(encoding="utf-8").replace('"value": 1', '"value": 9', 1)
    audit.path.write_text(text, encoding="utf-8")
    audit_report = audit.verify()
    cases.append(
        {
            "name": "audit_log_tamper",
            "detected": not audit_report["valid"],
            "audit": audit_report,
        }
    )

    profile_path = output / "faults" / "calibration-profile.json"
    save_profile(
        profile_path,
        CalibrationProfile(profile_id="fault", environment_id="room-a", hardware_ids=["node-1"]),
    )
    profile_path.write_text(
        profile_path.read_text(encoding="utf-8").replace('"room-a"', '"room-b"'),
        encoding="utf-8",
    )
    profile_detected = False
    try:
        load_profile(profile_path)
    except ValueError as exc:
        profile_detected = "hash mismatch" in str(exc)
    cases.append(
        {
            "name": "calibration_profile_tamper",
            "detected": profile_detected,
        }
    )

    active_detected = False
    try:
        select_active_measurement(
            [ActiveSensingCandidate("bad", float("nan"), 0.1)]
        )
    except ValueError:
        active_detected = True
    cases.append(
        {
            "name": "invalid_active_sensing_candidates",
            "detected": active_detected,
        }
    )

    passed = all(case["detected"] for case in cases)
    report = make_json_safe(
        {
            "schema_version": "physioatlas.fault-test.v2",
            "status": "passed" if passed else "failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output),
            "cases": cases,
        }
    )
    report_path = output / "fault-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
