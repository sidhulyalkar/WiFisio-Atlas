from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .rf_fusion import fuse_zone_evidence
from .rf_fusion_schema import (
    CsiBfiFusionConfig,
    FusionAblationReport,
    FusionDecision,
    FusionEvaluationMetrics,
    FusionTruthWindow,
    ZoneEvidence,
)


def evaluate_fusion_decisions(
    decisions: Iterable[FusionDecision],
    truth: Iterable[FusionTruthWindow],
) -> FusionEvaluationMetrics:
    predicted = list(decisions)
    references = list(truth)
    if len(predicted) != len(references):
        raise ValueError("one fusion decision is required per reference window")
    if not references:
        return FusionEvaluationMetrics(
            windows=0,
            coverage=0.0,
            abstention_rate=0.0,
            zone_precision=0.0,
            zone_recall=0.0,
            zone_f1=0.0,
            count_mae=0.0,
            exact_set_accuracy=0.0,
            selective_exact_accuracy=0.0,
        )
    true_positive = false_positive = false_negative = 0
    exact = selective_exact = emitted = 0
    count_errors = []
    for decision, reference in zip(predicted, references):
        if abs(decision.timestamp_s - reference.timestamp_s) > 0.5:
            raise ValueError("fusion and reference windows are not time aligned")
        expected = set(reference.active_zones)
        actual = set(decision.active_zones)
        true_positive += len(expected & actual)
        false_positive += len(actual - expected)
        false_negative += len(expected - actual)
        count_errors.append(abs(len(actual) - len(expected)))
        exact += int(actual == expected)
        if decision.status == "fused":
            emitted += 1
            selective_exact += int(actual == expected)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    windows = len(references)
    return FusionEvaluationMetrics(
        windows=windows,
        coverage=emitted / windows,
        abstention_rate=1.0 - emitted / windows,
        zone_precision=precision,
        zone_recall=recall,
        zone_f1=f1,
        count_mae=float(np.mean(count_errors)),
        exact_set_accuracy=exact / windows,
        selective_exact_accuracy=selective_exact / max(emitted, 1),
    )


def run_fusion_ablation(
    windows: Iterable[tuple[list[ZoneEvidence], FusionTruthWindow]],
    config: CsiBfiFusionConfig,
) -> FusionAblationReport:
    records = list(windows)
    if not records:
        raise ValueError("ablation requires at least one evidence/reference window")
    modes = {
        "csi_only": "wifi_csi",
        "bfi_only": "wifi_bfi",
        "csi_bfi_fusion": None,
    }
    metrics = {}
    truth = [item[1] for item in records]
    for mode, source in modes.items():
        decisions = []
        mode_config = config.model_copy(
            update={"require_both_sources": source is None}
        )
        for evidence, reference in records:
            selected = evidence if source is None else [
                item for item in evidence if item.source == source
            ]
            if selected:
                decisions.append(fuse_zone_evidence(selected, mode_config))
            else:
                decisions.append(
                    FusionDecision(
                        timestamp_s=reference.timestamp_s,
                        status="abstained",
                        reason="source_absent",
                        active_zones=[],
                        fused_probabilities={},
                        observability=0.0,
                        sources_present=[],
                        evidence_ids=[],
                    )
                )
        metrics[mode] = evaluate_fusion_decisions(decisions, truth)
    return FusionAblationReport(
        modes=metrics,
        reference_ids=sorted({item.reference_id for item in truth}),
    )
