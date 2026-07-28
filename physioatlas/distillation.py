from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class DistillationReport:
    available_fraction: float
    cosine_loss: float
    scale_loss: float
    total_loss: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class PrivilegedTeacherProjector(nn.Module):
    """Project a privileged teacher modality into the RF embedding space."""

    def __init__(self, teacher_features: int, embedding_dim: int = 128) -> None:
        super().__init__()
        if teacher_features < 1 or embedding_dim < 1:
            raise ValueError("teacher_features and embedding_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(teacher_features, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, teacher_features: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.network(teacher_features), dim=-1)


def privileged_distillation_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
    *,
    teacher_available: Optional[torch.Tensor] = None,
    scale_weight: float = 0.1,
) -> tuple[torch.Tensor, DistillationReport]:
    """Masked RF-to-teacher alignment that remains valid when teachers disappear."""
    if student_embedding.shape != teacher_embedding.shape or student_embedding.ndim != 2:
        raise ValueError("student and teacher embeddings must share shape [batch, features]")
    if teacher_available is None:
        available = torch.ones(student_embedding.shape[0], device=student_embedding.device)
    else:
        available = teacher_available.to(student_embedding.device).float().reshape(-1)
        if available.numel() != student_embedding.shape[0]:
            raise ValueError("teacher_available must have one value per batch item")
    valid = available > 0.5
    if not bool(valid.any()):
        zero = student_embedding.sum() * 0.0
        return zero, DistillationReport(0.0, 0.0, 0.0, 0.0)
    student = student_embedding[valid]
    teacher = teacher_embedding[valid].detach()
    cosine = (1.0 - nn.functional.cosine_similarity(student, teacher, dim=-1)).mean()
    scale = (student.norm(dim=-1) - teacher.norm(dim=-1)).square().mean()
    total = cosine + scale_weight * scale
    report = DistillationReport(
        available_fraction=float(valid.float().mean().detach().cpu()),
        cosine_loss=float(cosine.detach().cpu()),
        scale_loss=float(scale.detach().cpu()),
        total_loss=float(total.detach().cpu()),
    )
    return total, report


def teacher_removal_gap(
    with_teacher_metric: float,
    without_teacher_metric: float,
    *,
    lower_is_better: bool = True,
) -> float:
    """Quantify deployment degradation after removing privileged supervision."""
    if lower_is_better:
        return float(without_teacher_metric - with_teacher_metric)
    return float(with_teacher_metric - without_teacher_metric)
