from pathlib import Path
import shutil

import numpy as np

from physioatlas.household import (
    classify_household_identity,
    enroll_household,
    evaluate_household_registry,
    load_household_registry,
)
from physioatlas.schema import Modality
from physioatlas.synthetic import create_synthetic_cohort


def test_household_enrollment_and_open_set_unknown(tmp_path: Path):
    data = tmp_path / "data"
    create_synthetic_cohort(data, subjects=3, sessions_per_subject=3, duration_seconds=18, seed=91)
    calibration = tmp_path / "calibration"
    validation = tmp_path / "validation"
    for subject_dir in sorted(data.glob("subject-*")):
        sessions = sorted(subject_dir.glob("*-session-*"))
        for session_dir in sessions[:-1]:
            destination = calibration / subject_dir.name / session_dir.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(session_dir, destination)
        destination = validation / subject_dir.name / sessions[-1].name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sessions[-1], destination)
    registry_path = tmp_path / "registry.json"
    enrollment = enroll_household(calibration, registry_path, minimum_sessions=2)
    assert enrollment["ready"]
    assert enrollment["members_enrolled"] == 3

    evaluation = evaluate_household_registry(validation, registry_path)
    assert evaluation["split_is_disjoint"]
    assert evaluation["accepted_accuracy"] >= 0.80
    assert evaluation["coverage"] >= 0.50

    registry = load_household_registry(registry_path)
    dimension = len(registry.members[0].profiles[0].centroid)
    vector = np.zeros(dimension)
    vector[0] = 1.0
    decision = classify_household_identity(registry, {Modality.WIFI_CSI: vector})
    assert not decision.accepted
    assert decision.member_id is None


def test_household_requires_multiple_sessions(tmp_path: Path):
    data = tmp_path / "data"
    create_synthetic_cohort(data, subjects=2, sessions_per_subject=1, duration_seconds=10, seed=13)
    result = enroll_household(data, tmp_path / "registry.json", minimum_sessions=2)
    assert not result["ready"]
    assert result["members_enrolled"] == 0


def test_household_evaluation_rejects_enrollment_overlap(tmp_path: Path):
    data = tmp_path / "overlap-data"
    create_synthetic_cohort(data, subjects=2, sessions_per_subject=2, duration_seconds=12, seed=151)
    registry_path = tmp_path / "overlap-registry.json"
    assert enroll_household(data, registry_path, minimum_sessions=2)["ready"]
    report = evaluate_household_registry(data, registry_path)
    assert not report["split_is_disjoint"]
    assert report["overlapping_enrollment_sessions"]
    assert report["valid"] is False
