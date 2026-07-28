from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .io import LoadedSession, load_session
from .schema import AnatomyRegion, Modality
from .synchronization import align_streams


class PhysiologyNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    modality: Modality
    sensor_id: str
    anatomy_region: AnatomyRegion
    units: str = "arbitrary"
    description: str = ""


class PhysiologyEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    min_delay_s: float = Field(default=-0.5)
    max_delay_s: float = Field(default=1.0)
    expected_direction: str = Field(default="source_precedes_target")
    mechanism: str = "association"

    @model_validator(mode="after")
    def validate_delay(self) -> "PhysiologyEdge":
        if self.max_delay_s <= self.min_delay_s:
            raise ValueError("max_delay_s must exceed min_delay_s")
        return self


class PhysiologyGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.physiology-graph.v1"
    graph_id: str
    nodes: list[PhysiologyNode]
    edges: list[PhysiologyEdge]
    research_only: bool = True

    @model_validator(mode="after")
    def validate_graph(self) -> "PhysiologyGraph":
        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Physiology node IDs must be unique")
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(f"Edge {edge.source}->{edge.target} references an unknown node")
        return self


@dataclass(frozen=True)
class DelayEstimate:
    delay_s: float
    correlation: float
    samples: int
    within_declared_bounds: bool


def load_physiology_graph(path: Union[str, Path]) -> PhysiologyGraph:
    return PhysiologyGraph.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def estimate_delay(
    source: np.ndarray,
    target: np.ndarray,
    *,
    sample_rate_hz: float,
    min_delay_s: float,
    max_delay_s: float,
) -> DelayEstimate:
    """Estimate target lag relative to source using normalized correlation."""
    x = np.asarray(source, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n < 8:
        raise ValueError("Delay estimation needs at least eight samples")
    x = np.nan_to_num(x[:n] - np.nanmean(x[:n]))
    y = np.nan_to_num(y[:n] - np.nanmean(y[:n]))
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        raise ValueError("Delay estimation signals need non-zero variance")
    min_lag = int(np.floor(min_delay_s * sample_rate_hz))
    max_lag = int(np.ceil(max_delay_s * sample_rate_hz))
    best_lag = 0
    best_corr = -np.inf
    best_samples = 0
    for lag in range(min_lag, max_lag + 1):
        if lag >= 0:
            a, b = x[: n - lag], y[lag:]
        else:
            a, b = x[-lag:], y[: n + lag]
        if a.size < 4:
            continue
        correlation = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if correlation > best_corr:
            best_corr = correlation
            best_lag = lag
            best_samples = int(a.size)
    delay_s = best_lag / sample_rate_hz
    return DelayEstimate(
        delay_s=delay_s,
        correlation=best_corr,
        samples=best_samples,
        within_declared_bounds=min_delay_s <= delay_s <= max_delay_s,
    )


def analyze_session_propagation(
    session: LoadedSession,
    graph: PhysiologyGraph,
    *,
    sample_rate_hz: float = 50.0,
    max_gap_s: float = 0.25,
) -> dict:
    aligned = align_streams(
        session.streams,
        sample_rate_hz=sample_rate_hz,
        max_gap_s=max_gap_s,
    )
    node_values: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for node in graph.nodes:
        key = f"{node.modality.value}:{node.sensor_id}"
        if key not in aligned.values:
            missing.append(node.node_id)
            continue
        node_values[node.node_id] = aligned.values[key][:, 0]
    edges = []
    for edge in graph.edges:
        if edge.source not in node_values or edge.target not in node_values:
            edges.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "status": "missing_stream",
                }
            )
            continue
        estimate = estimate_delay(
            node_values[edge.source],
            node_values[edge.target],
            sample_rate_hz=sample_rate_hz,
            min_delay_s=edge.min_delay_s,
            max_delay_s=edge.max_delay_s,
        )
        direction_ok = (
            estimate.delay_s >= 0
            if edge.expected_direction == "source_precedes_target"
            else True
        )
        edges.append(
            {
                "source": edge.source,
                "target": edge.target,
                "mechanism": edge.mechanism,
                "status": "estimated",
                "delay_s": estimate.delay_s,
                "correlation": estimate.correlation,
                "samples": estimate.samples,
                "within_declared_bounds": estimate.within_declared_bounds,
                "direction_ok": direction_ok,
            }
        )
    return {
        "schema_version": "physioatlas.propagation-analysis.v1",
        "session_id": session.manifest.session_id,
        "graph_id": graph.graph_id,
        "sample_rate_hz": sample_rate_hz,
        "missing_nodes": missing,
        "edges": edges,
        "valid": not missing and all(
            edge.get("within_declared_bounds", False) and edge.get("direction_ok", False)
            for edge in edges
        ),
    }


def analyze_dataset_propagation(
    root: Union[str, Path],
    *,
    sample_rate_hz: float = 50.0,
) -> dict:
    reports = []
    for manifest_path in sorted(Path(root).rglob("manifest.json")):
        session = load_session(manifest_path)
        graph_path = session.manifest.physiology_graph_path
        if not graph_path:
            continue
        reports.append(
            analyze_session_propagation(
                session,
                load_physiology_graph(graph_path),
                sample_rate_hz=sample_rate_hz,
            )
        )
    return {
        "schema_version": "physioatlas.propagation-dataset.v1",
        "sessions": reports,
        "valid": bool(reports) and all(report["valid"] for report in reports),
    }


def propagation_consistency_loss(
    node_waveforms: dict[str, torch.Tensor],
    graph: PhysiologyGraph,
    *,
    sample_rate_hz: float,
) -> torch.Tensor:
    """Differentiable lag-order regularizer for multi-node waveform models.

    It penalizes a target whose center of temporal energy occurs before a source
    on edges declared ``source_precedes_target``. This is intentionally a weak
    prior and never substitutes for reference measurements.
    """
    losses: list[torch.Tensor] = []
    for edge in graph.edges:
        if edge.source not in node_waveforms or edge.target not in node_waveforms:
            continue
        source = node_waveforms[edge.source]
        target = node_waveforms[edge.target]
        if source.shape != target.shape:
            raise ValueError("Propagation graph waveforms must have matching shapes")
        time = torch.arange(source.shape[-2], device=source.device, dtype=source.dtype) / sample_rate_hz
        source_energy = source.square().mean(dim=-1)
        target_energy = target.square().mean(dim=-1)
        source_center = (source_energy * time).sum(dim=-1) / source_energy.sum(dim=-1).clamp_min(1e-6)
        target_center = (target_energy * time).sum(dim=-1) / target_energy.sum(dim=-1).clamp_min(1e-6)
        if edge.expected_direction == "source_precedes_target":
            losses.append(torch.relu(source_center - target_center).mean())
    if not losses:
        first = next(iter(node_waveforms.values()), None)
        return torch.tensor(0.0, device=first.device if first is not None else None)
    return torch.stack(losses).mean()
