import pytest

from physioatlas.sleep_benchmark import (
    collapse_aasm_stage,
    evaluate_sleep_predictions,
    leave_one_subject_out_folds,
)


def test_collapse_aasm_stage_builds_four_class_target():
    assert collapse_aasm_stage("W") == "wake"
    assert collapse_aasm_stage("N1") == "light"
    assert collapse_aasm_stage("N2") == "light"
    assert collapse_aasm_stage("N3") == "deep"
    assert collapse_aasm_stage("R") == "rem"


def test_perfect_sleep_predictions_report_perfect_selective_metrics():
    stages = ["W", "N1", "N2", "N3", "R"] * 2
    probabilities = [
        [0.97, 0.01, 0.01, 0.01],
        [0.01, 0.97, 0.01, 0.01],
        [0.01, 0.97, 0.01, 0.01],
        [0.01, 0.01, 0.97, 0.01],
        [0.01, 0.01, 0.01, 0.97],
    ] * 2
    predictions = ["wake", "light", "light", "deep", "rem"] * 2

    report = evaluate_sleep_predictions(
        stages,
        predictions,
        subject_ids=["a"] * 5 + ["b"] * 5,
        probabilities=probabilities,
    )

    assert report["accuracy"] == pytest.approx(1.0)
    assert report["balanced_accuracy"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx(1.0)
    assert report["cohen_kappa"] == pytest.approx(1.0)
    assert report["coverage"] == pytest.approx(1.0)
    assert report["subject_macro_f1_mean"] == pytest.approx(1.0)
    assert report["probability_metrics"]["multiclass_brier"] < 0.01


def test_sleep_benchmark_reports_abstention_coverage():
    report = evaluate_sleep_predictions(
        ["wake", "light", "deep", "rem"],
        ["wake", "light", "wake", "rem"],
        confidences=[0.9, 0.8, 0.2, 0.7],
        minimum_confidence=0.5,
    )

    assert report["evaluated_epochs"] == 3
    assert report["coverage"] == pytest.approx(0.75)
    assert report["accuracy"] == pytest.approx(1.0)


def test_leave_one_subject_out_folds_are_disjoint():
    subject_ids = ["a", "a", "b", "b", "c"]
    folds = leave_one_subject_out_folds(subject_ids)

    assert len(folds) == 3
    for fold in folds:
        train_subjects = {subject_ids[index] for index in fold["train_indices"]}
        validation_subjects = {
            subject_ids[index] for index in fold["validation_indices"]
        }
        assert train_subjects.isdisjoint(validation_subjects)
        assert validation_subjects == {fold["held_out_subject"]}
