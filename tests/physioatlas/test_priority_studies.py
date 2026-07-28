from pathlib import Path

from physioatlas.studies import PriorityStudy, run_all_priority_studies
from physioatlas.synthetic import create_synthetic_cohort


def test_all_priority_studies_execute_with_references(tmp_path: Path):
    data = tmp_path / "data"
    create_synthetic_cohort(data, subjects=3, sessions_per_subject=2, duration_seconds=20, seed=31)
    report = run_all_priority_studies(data, tmp_path / "studies", minimum_sessions=2)
    assert report["valid"]
    assert {item["target"] for item in report["studies"]} == {item.value for item in PriorityStudy}
    for item in report["studies"]:
        assert item["sessions_analyzed"] == 6
        assert item["valid"]
        assert (tmp_path / "studies" / item["target"] / "study-report.json").exists()


def test_pulse_delay_recovers_positive_lag(tmp_path: Path):
    data = tmp_path / "data"
    create_synthetic_cohort(data, subjects=2, sessions_per_subject=2, duration_seconds=24, sample_rate_hz=20, seed=44)
    report = run_all_priority_studies(data, tmp_path / "studies", minimum_sessions=2)
    pulse = next(item for item in report["studies"] if item["target"] == "pulse_propagation")
    delays = [row["ecg_to_ppg_delay_s"] for row in pulse["session_results"]]
    assert all(0.02 <= value <= 0.6 for value in delays)
    assert all(row["matched_fraction"] >= 0.70 for row in pulse["session_results"])
    assert pulse["scientific_gate"]["passed"] is True
