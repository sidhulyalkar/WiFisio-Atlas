from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
)


SLEEP_STAGES = ("wake", "light", "deep", "rem")
_STAGE_TO_INDEX = {stage: index for index, stage in enumerate(SLEEP_STAGES)}


def collapse_aasm_stage(stage: str) -> str:
    """Collapse common AASM/R&K labels into a four-stage research target."""

    normalized = str(stage).strip().upper().replace(" ", "")
    mapping = {
        "W": "wake",
        "WAKE": "wake",
        "0": "wake",
        "N1": "light",
        "S1": "light",
        "1": "light",
        "N2": "light",
        "S2": "light",
        "2": "light",
        "N3": "deep",
        "S3": "deep",
        "S4": "deep",
        "3": "deep",
        "4": "deep",
        "R": "rem",
        "REM": "rem",
        "5": "rem",
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported sleep-stage label: {stage!r}")
    return mapping[normalized]


def _normalize_stage_sequence(stages: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for stage in stages:
        value = str(stage).strip().lower()
        if value in _STAGE_TO_INDEX:
            normalized.append(value)
        else:
            normalized.append(collapse_aasm_stage(stage))
    return normalized


def _validate_probabilities(
    probabilities: Sequence[Sequence[float]], total_epochs: int
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (total_epochs, len(SLEEP_STAGES)):
        raise ValueError(
            "probabilities must have shape "
            f"({total_epochs}, {len(SLEEP_STAGES)}) in {SLEEP_STAGES} order"
        )
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must be in [0, 1]")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("each probability row must sum to 1")
    return values


def expected_calibration_error(
    correct: Sequence[bool], confidence: Sequence[float], bins: int = 10
) -> float:
    """Return top-label expected calibration error over equal-width bins."""

    if bins < 2:
        raise ValueError("bins must be >= 2")
    correct_array = np.asarray(correct, dtype=np.float64)
    confidence_array = np.asarray(confidence, dtype=np.float64)
    if correct_array.shape != confidence_array.shape:
        raise ValueError("correct and confidence must have matching shapes")
    if correct_array.size == 0:
        raise ValueError("at least one observation is required")
    if np.any(confidence_array < 0.0) or np.any(confidence_array > 1.0):
        raise ValueError("confidence must be in [0, 1]")

    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (confidence_array >= lower) & (confidence_array <= upper)
        else:
            mask = (confidence_array >= lower) & (confidence_array < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_accuracy = float(correct_array[mask].mean())
        bin_confidence = float(confidence_array[mask].mean())
        error += (count / correct_array.size) * abs(bin_accuracy - bin_confidence)
    return float(error)


def leave_one_subject_out_folds(subject_ids: Sequence[str]) -> list[dict]:
    """Create deterministic patient-disjoint folds for sleep model evaluation."""

    ids = [str(subject_id) for subject_id in subject_ids]
    unique_subjects = sorted(set(ids))
    if len(unique_subjects) < 2:
        raise ValueError("leave-one-subject-out evaluation requires >= 2 subjects")

    folds: list[dict] = []
    for held_out in unique_subjects:
        validation = [index for index, value in enumerate(ids) if value == held_out]
        train = [index for index, value in enumerate(ids) if value != held_out]
        folds.append(
            {
                "held_out_subject": held_out,
                "train_indices": train,
                "validation_indices": validation,
            }
        )
    return folds


def _transition_rate(stages: Sequence[str]) -> float:
    if len(stages) < 2:
        return 0.0
    transitions = sum(a != b for a, b in zip(stages[:-1], stages[1:]))
    return float(transitions / (len(stages) - 1))


def evaluate_sleep_predictions(
    references: Sequence[str],
    predictions: Sequence[str],
    *,
    subject_ids: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float]] = None,
    probabilities: Optional[Sequence[Sequence[float]]] = None,
    minimum_confidence: float = 0.0,
    calibration_bins: int = 10,
) -> dict:
    """Evaluate four-stage sleep predictions with abstention and calibration.

    The function is deliberately model-agnostic. It accepts epoch-level predictions
    produced by any WiFi, radar, wearable, or multimodal model and reports metrics
    against PSG-derived labels. Accuracy is included for comparability with prior
    literature, but macro F1, balanced accuracy, kappa, calibration, coverage, and
    patient-level summaries are treated as first-class outputs.
    """

    reference = _normalize_stage_sequence(references)
    predicted = _normalize_stage_sequence(predictions)
    if len(reference) != len(predicted):
        raise ValueError("references and predictions must have matching lengths")
    if not reference:
        raise ValueError("at least one sleep epoch is required")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")

    total = len(reference)
    probability_array = (
        _validate_probabilities(probabilities, total) if probabilities is not None else None
    )

    if confidences is None:
        if probability_array is not None:
            confidence_array = probability_array.max(axis=1)
        else:
            confidence_array = np.ones(total, dtype=np.float64)
    else:
        confidence_array = np.asarray(confidences, dtype=np.float64)
        if confidence_array.shape != (total,):
            raise ValueError("confidences must have one value per epoch")
        if not np.isfinite(confidence_array).all():
            raise ValueError("confidences must be finite")
        if np.any(confidence_array < 0.0) or np.any(confidence_array > 1.0):
            raise ValueError("confidences must be in [0, 1]")

    selected = confidence_array >= minimum_confidence
    selected_count = int(selected.sum())
    if selected_count == 0:
        raise ValueError("confidence threshold abstained on every epoch")

    selected_reference = [value for value, keep in zip(reference, selected) if keep]
    selected_predicted = [value for value, keep in zip(predicted, selected) if keep]
    selected_confidence = confidence_array[selected]

    report = classification_report(
        selected_reference,
        selected_predicted,
        labels=list(SLEEP_STAGES),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        selected_reference,
        selected_predicted,
        labels=list(SLEEP_STAGES),
    )
    correct = np.asarray(
        [a == b for a, b in zip(selected_reference, selected_predicted)],
        dtype=bool,
    )

    result: dict = {
        "schema_version": "physioatlas.sleep-benchmark.v1",
        "research_only": True,
        "clinical_claim_allowed": False,
        "claim_boundary": (
            "Epoch-level sleep research benchmark against PSG-derived labels; "
            "not a clinical sleep assessment or diagnostic device evaluation."
        ),
        "stage_order": list(SLEEP_STAGES),
        "total_epochs": total,
        "evaluated_epochs": selected_count,
        "coverage": float(selected_count / total),
        "minimum_confidence": float(minimum_confidence),
        "accuracy": float(accuracy_score(selected_reference, selected_predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(selected_reference, selected_predicted)
        ),
        "macro_f1": float(
            f1_score(
                selected_reference,
                selected_predicted,
                labels=list(SLEEP_STAGES),
                average="macro",
                zero_division=0,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(
                selected_reference,
                selected_predicted,
                labels=list(SLEEP_STAGES),
            )
        ),
        "expected_calibration_error": expected_calibration_error(
            correct,
            selected_confidence,
            bins=calibration_bins,
        ),
        "reference_transition_rate": _transition_rate(selected_reference),
        "predicted_transition_rate": _transition_rate(selected_predicted),
        "per_stage": {
            stage: {
                "precision": float(report[stage]["precision"]),
                "recall": float(report[stage]["recall"]),
                "f1": float(report[stage]["f1-score"]),
                "support": int(report[stage]["support"]),
            }
            for stage in SLEEP_STAGES
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }

    if probability_array is not None:
        selected_probabilities = probability_array[selected]
        integer_reference = np.asarray(
            [_STAGE_TO_INDEX[value] for value in selected_reference], dtype=np.int64
        )
        one_hot = np.eye(len(SLEEP_STAGES), dtype=np.float64)[integer_reference]
        result["probability_metrics"] = {
            "log_loss": float(
                log_loss(
                    integer_reference,
                    selected_probabilities,
                    labels=list(range(len(SLEEP_STAGES))),
                )
            ),
            "multiclass_brier": float(
                np.mean(np.sum((selected_probabilities - one_hot) ** 2, axis=1))
            ),
        }

    if subject_ids is not None:
        ids = [str(subject_id) for subject_id in subject_ids]
        if len(ids) != total:
            raise ValueError("subject_ids must have one value per epoch")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, (subject_id, keep) in enumerate(zip(ids, selected)):
            if keep:
                grouped[subject_id].append(index)

        subject_metrics: dict[str, dict] = {}
        for subject_id, indices in sorted(grouped.items()):
            subject_reference = [reference[index] for index in indices]
            subject_predicted = [predicted[index] for index in indices]
            subject_metrics[subject_id] = {
                "epochs": len(indices),
                "accuracy": float(
                    accuracy_score(subject_reference, subject_predicted)
                ),
                "macro_f1": float(
                    f1_score(
                        subject_reference,
                        subject_predicted,
                        labels=list(SLEEP_STAGES),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "cohen_kappa": float(
                    cohen_kappa_score(
                        subject_reference,
                        subject_predicted,
                        labels=list(SLEEP_STAGES),
                    )
                ),
            }
        result["by_subject"] = subject_metrics
        result["subject_macro_f1_mean"] = float(
            np.mean([metrics["macro_f1"] for metrics in subject_metrics.values()])
        )

    return result
