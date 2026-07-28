from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from .dataset import WindowExample


_VALID_STRATEGIES = {"subject", "session", "environment", "posture", "condition", "hardware"}


def example_group(example: WindowExample, strategy: str) -> str:
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unknown split strategy {strategy!r}")
    mapping = {
        "subject": example.subject_id,
        "session": example.session_id,
        "environment": example.environment_id,
        "posture": example.posture or "unknown",
        "condition": example.condition or "unknown",
        "hardware": example.hardware_domain or "unknown",
    }
    return mapping[strategy]


def group_split(
    examples: Sequence[WindowExample],
    *,
    strategy: str,
    validation_fraction: float,
    seed: int,
    held_out_groups: Iterable[str] = (),
) -> tuple[list[WindowExample], list[WindowExample], dict[str, list[str]]]:
    if not examples:
        raise ValueError("Cannot split an empty example collection")
    groups = sorted({example_group(example, strategy) for example in examples})
    if len(groups) < 2:
        raise ValueError(f"{strategy}-held-out evaluation needs at least two groups")
    explicit = sorted(set(held_out_groups))
    unknown = sorted(set(explicit) - set(groups))
    if unknown:
        raise ValueError(f"Held-out groups do not exist: {unknown}")
    if explicit:
        validation_groups = set(explicit)
    else:
        rng = np.random.default_rng(seed)
        shuffled = list(rng.permutation(groups))
        n_validation = max(1, int(round(len(groups) * validation_fraction)))
        n_validation = min(n_validation, len(groups) - 1)
        validation_groups = set(shuffled[:n_validation])
    train_groups = set(groups) - validation_groups
    train = [e for e in examples if example_group(e, strategy) in train_groups]
    validation = [e for e in examples if example_group(e, strategy) in validation_groups]
    if not train or not validation:
        raise ValueError("Group split produced an empty train or validation partition")
    # Leakage means the same unit of the declared split strategy appears on both sides.
    train_seen = {example_group(e, strategy) for e in train}
    validation_seen = {example_group(e, strategy) for e in validation}
    if not train_seen.isdisjoint(validation_seen):
        raise AssertionError("Group split leaked groups across partitions")
    return train, validation, {
        "train_groups": sorted(train_seen),
        "validation_groups": sorted(validation_seen),
    }


def count_examples_by_domain(examples: Sequence[WindowExample]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for strategy in sorted(_VALID_STRATEGIES):
        values: defaultdict[str, int] = defaultdict(int)
        for example in examples:
            values[example_group(example, strategy)] += 1
        counts[strategy] = dict(sorted(values.items()))
    return counts
