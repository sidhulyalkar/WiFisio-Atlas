from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Union

from .household import enroll_household, evaluate_household_registry
from .household_live import simulate_household_live
from .research_hub import ResearchHubStore
from .studies import run_all_priority_studies
from .synthetic import create_synthetic_cohort
from .utils import make_json_safe


def run_household_smoke_test(
    output_root: Union[str, Path],
    *,
    seed: int = 73,
    members: int = 4,
    sessions_per_member: int = 3,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    data = output / "data"
    total_sessions = max(sessions_per_member + 1, 3)
    create_synthetic_cohort(
        data,
        subjects=members,
        sessions_per_subject=total_sessions,
        duration_seconds=28.0,
        sample_rate_hz=20.0,
        seed=seed,
    )
    calibration_data = output / "calibration-data"
    validation_data = output / "validation-data"
    for subject_dir in sorted(data.glob("subject-*")):
        sessions = sorted(subject_dir.glob("*-session-*"))
        for session_dir in sessions[:-1]:
            destination = calibration_data / subject_dir.name / session_dir.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(session_dir, destination)
        held_out = sessions[-1]
        destination = validation_data / subject_dir.name / held_out.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(held_out, destination)
    registry_path = output / "household-registry.json"
    enrollment = enroll_household(
        calibration_data,
        registry_path,
        household_id="synthetic-household",
        aliases={f"subject-{index + 1:03d}": f"Member {index + 1}" for index in range(members)},
        minimum_sessions=2,
    )
    evaluation = evaluate_household_registry(validation_data, registry_path)
    studies = run_all_priority_studies(data, output / "studies", minimum_sessions=2)
    state_path = output / "research-hub-state.json"
    simulation = simulate_household_live(
        registry_path,
        state_path,
        steps=18,
        seed=seed + 1,
        include_unknown=True,
    )
    store = ResearchHubStore(state_path)
    state = store.read()
    state["studies"] = [
        {
            "study_id": report["study_id"],
            "target": report["target"],
            "sessions_analyzed": report["sessions_analyzed"],
            "valid": report["valid"],
            "catalog": report["catalog"],
        }
        for report in studies["studies"]
    ]
    store.publish(state)
    checks = {
        "all_members_enrolled": enrollment["members_enrolled"] == members,
        "registry_ready": enrollment["ready"],
        "identity_evaluation_valid": evaluation["valid"],
        "identity_split_disjoint": evaluation["split_is_disjoint"],
        "identity_accuracy_at_least_80pct": evaluation["accepted_accuracy"] >= 0.80,
        "live_tracks_identified": simulation["identified_tracks"] >= members,
        "unknown_remained_anonymous": simulation["unknown_remained_anonymous"],
        "all_priority_studies_executed": len(studies["studies"]) == 5,
        "all_priority_studies_valid": studies["valid"],
        "research_hub_state_exists": state_path.exists(),
    }
    report = make_json_safe(
        {
            "schema_version": "physioatlas.household-smoke.v1",
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "enrollment": enrollment,
            "evaluation": evaluation,
            "simulation": simulation,
            "study_suite_path": str((output / "studies" / "priority-study-suite.json").resolve()),
            "research_hub_state": str(state_path.resolve()),
            "research_only": True,
        }
    )
    report_path = output / "household-smoke-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def run_household_fault_test(
    output_root: Union[str, Path],
    *,
    seed: int = 97,
) -> dict[str, Any]:
    from .household import classify_household_identity, load_household_registry
    from .research_hub import ResearchHubStore
    from .schema import Modality
    import numpy as np

    output = Path(output_root).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    data = output / "data"
    create_synthetic_cohort(data, subjects=2, sessions_per_subject=2, duration_seconds=12, seed=seed)

    # Revoke one participant's identity permissions across every session.
    revoked_subject = "subject-002"
    for consent_path in (data / revoked_subject).rglob("consent.json"):
        payload = json.loads(consent_path.read_text(encoding="utf-8"))
        payload["allow_identity_enrollment"] = False
        payload["allow_live_tracking"] = False
        payload["allow_household_dashboard"] = False
        payload["participant_acknowledged"] = False
        consent_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry_path = output / "registry.json"
    enrollment = enroll_household(data, registry_path, minimum_sessions=2)
    consent_rejection_detected = revoked_subject not in enrollment["members"]

    registry = load_household_registry(registry_path)
    profile = registry.members[0].profiles[0]
    unknown_vector = np.roll(np.asarray(profile.centroid, dtype=float), max(1, len(profile.centroid) // 3))
    decision = classify_household_identity(registry, {Modality.WIFI_CSI: unknown_vector})
    open_set_rejection_detected = not decision.accepted

    store = ResearchHubStore(output / "state.json")
    try:
        store.enqueue_action("run_arbitrary_shell", {"command": "echo unsafe"})
        unsafe_action_rejected = False
    except ValueError:
        unsafe_action_rejected = True

    cases = [
        {"name": "revoked_identity_consent", "detected": consent_rejection_detected},
        {"name": "out_of_distribution_identity", "detected": open_set_rejection_detected},
        {"name": "unsafe_dashboard_action", "detected": unsafe_action_rejected},
    ]
    report = {
        "schema_version": "physioatlas.household-fault-report.v1",
        "status": "passed" if all(case["detected"] for case in cases) else "failed",
        "cases": cases,
        "research_only": True,
    }
    report_path = output / "household-fault-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
