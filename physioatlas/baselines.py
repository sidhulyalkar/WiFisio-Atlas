from __future__ import annotations

from dataclasses import replace

import numpy as np

from .dataset import WindowExample
from .metrics import masked_correlation, masked_mae, masked_rmse


def _stack_samples(examples: list[WindowExample], include_auxiliary: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for example in examples:
        parts = [example.rf]
        if include_auxiliary:
            parts.append(example.auxiliary)
        features.append(np.concatenate(parts, axis=1))
        targets.append(example.target)
        masks.append(example.target_mask)
    return np.concatenate(features), np.concatenate(targets), np.concatenate(masks)


def evaluate_mean_baseline(
    train_examples: list[WindowExample], validation_examples: list[WindowExample]
) -> dict[str, float]:
    _, train_target, train_mask = _stack_samples(train_examples, include_auxiliary=False)
    valid_train = np.broadcast_to(train_mask.astype(bool), train_target.shape)
    mean = np.zeros(train_target.shape[1], dtype=np.float64)
    for column in range(train_target.shape[1]):
        mask = valid_train[:, column]
        mean[column] = float(train_target[mask, column].mean()) if np.any(mask) else 0.0
    _, target, mask = _stack_samples(validation_examples, include_auxiliary=False)
    prediction = np.broadcast_to(mean, target.shape)
    return {
        "rmse": masked_rmse(prediction, target, mask),
        "mae": masked_mae(prediction, target, mask),
        "correlation": masked_correlation(prediction, target, mask),
    }


def evaluate_ridge_baseline(
    train_examples: list[WindowExample],
    validation_examples: list[WindowExample],
    *,
    alpha: float = 1.0,
    include_auxiliary: bool = False,
) -> dict[str, float]:
    """Closed-form ridge baseline over synchronized per-timestamp features."""
    train_x, train_y, train_mask = _stack_samples(train_examples, include_auxiliary)
    validation_x, validation_y, validation_mask = _stack_samples(
        validation_examples, include_auxiliary
    )
    observed = train_mask[:, 0].astype(bool)
    train_x = train_x[observed]
    train_y = train_y[observed]
    if train_x.shape[0] < 2:
        raise ValueError("Not enough observed samples for ridge baseline")
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    x = (train_x - mean) / scale
    vx = (validation_x - mean) / scale
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    vx = np.concatenate([vx, np.ones((vx.shape[0], 1))], axis=1)
    gram = x.T @ x
    regularizer = alpha * np.eye(gram.shape[0])
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(gram + regularizer, x.T @ train_y)
    prediction = vx @ weights
    return {
        "rmse": masked_rmse(prediction, validation_y, validation_mask),
        "mae": masked_mae(prediction, validation_y, validation_mask),
        "correlation": masked_correlation(prediction, validation_y, validation_mask),
    }


def time_shift_targets(
    examples: list[WindowExample], *, fraction: float = 0.5
) -> list[WindowExample]:
    """Circularly shift each target and mask to create a temporal null."""
    shifted: list[WindowExample] = []
    for example in examples:
        amount = max(1, int(round(example.target.shape[0] * fraction)))
        shifted.append(
            replace(
                example,
                target=np.roll(example.target, amount, axis=0).copy(),
                target_mask=np.roll(example.target_mask, amount, axis=0).copy(),
                observability_target=np.roll(example.observability_target, amount, axis=0).copy(),
            )
        )
    return shifted


def permute_target_windows(
    examples: list[WindowExample], *, seed: int
) -> list[WindowExample]:
    """Permute target windows across observations while preserving shapes."""
    if not examples:
        return []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(examples))
    permuted: list[WindowExample] = []
    for index, source_index in enumerate(order):
        source = examples[int(source_index)]
        destination = examples[index]
        if source.target.shape != destination.target.shape:
            raise ValueError("Target permutation requires equal window shapes")
        permuted.append(
            replace(
                destination,
                target=source.target.copy(),
                target_mask=source.target_mask.copy(),
                observability_target=source.observability_target.copy(),
            )
        )
    return permuted
