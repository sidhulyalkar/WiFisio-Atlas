from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from .household import evaluate_household_registry, load_household_registry
from .household_live import calibration_matrix
from .research_hub import ResearchHubStore
from .studies import run_all_priority_studies
from .utils import make_json_safe


class ResearchHubActionWorker:
    def __init__(
        self,
        store: ResearchHubStore,
        *,
        dataset_root: Optional[Union[str, Path]] = None,
        registry_path: Optional[Union[str, Path]] = None,
        output_root: Union[str, Path] = "outputs/physioatlas/research-hub-actions",
    ) -> None:
        self.store = store
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else None
        self.registry_path = Path(registry_path).resolve() if registry_path else None
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cursor_path = self.output_root / "action-cursor.txt"

    def _cursor(self) -> int:
        return int(self.cursor_path.read_text().strip()) if self.cursor_path.exists() else 0

    def _save_cursor(self, value: int) -> None:
        self.cursor_path.write_text(str(value) + "\n", encoding="utf-8")

    def _handle(self, entry: dict[str, Any]) -> dict[str, Any]:
        action = entry["action"]
        state = self.store.read()
        if action == "start_recording":
            state["status"] = "recording_requested"
            self.store.publish(state)
            return {"action": action, "status": "requested", "note": "Acquisition adapter must be configured by InnerLoop or the operator."}
        if action == "stop_recording":
            state["status"] = "stopped"
            self.store.publish(state)
            return {"action": action, "status": "completed"}
        if action == "clear_events":
            state["events"] = []
            self.store.publish(state)
            return {"action": action, "status": "completed"}
        if action == "run_calibration":
            if self.dataset_root is None or self.registry_path is None:
                raise ValueError("run_calibration requires dataset_root and registry_path")
            evaluation = evaluate_household_registry(self.dataset_root, self.registry_path)
            registry = load_household_registry(self.registry_path)
            state["calibration"] = calibration_matrix(registry)
            state.setdefault("experiments", []).append(
                {
                    "experiment_id": "household-calibration",
                    "status": "passed" if evaluation["valid"] else "needs_more_data",
                    "accepted_accuracy": evaluation["accepted_accuracy"],
                    "coverage": evaluation["coverage"],
                }
            )
            self.store.publish(state)
            return {"action": action, "status": "completed", "evaluation": evaluation}
        if action == "run_priority_studies":
            if self.dataset_root is None:
                raise ValueError("run_priority_studies requires dataset_root")
            reports = run_all_priority_studies(self.dataset_root, self.output_root / "studies")
            state["studies"] = [
                {
                    "study_id": report["study_id"],
                    "target": report["target"],
                    "sessions_analyzed": report["sessions_analyzed"],
                    "valid": report["valid"],
                    "catalog": report["catalog"],
                    "aggregate": report["aggregate"],
                }
                for report in reports["studies"]
            ]
            self.store.publish(state)
            return {"action": action, "status": "completed", "valid": reports["valid"]}
        raise ValueError(f"Unsupported action: {action}")

    def process_once(self) -> dict[str, Any]:
        if not self.store.action_path.exists():
            return {"processed": 0, "results": []}
        lines = [line for line in self.store.action_path.read_text(encoding="utf-8").splitlines() if line]
        cursor = self._cursor()
        results = []
        for index, line in enumerate(lines[cursor:], start=cursor):
            entry = json.loads(line)
            try:
                result = self._handle(entry)
                result["queue_index"] = index
                results.append(result)
                self.store.append_event("action_completed", result)
            except Exception as exc:
                result = {"queue_index": index, "action": entry.get("action"), "status": "error", "error": str(exc)}
                results.append(result)
                self.store.append_event("action_failed", result)
            self._save_cursor(index + 1)
        return make_json_safe({"processed": len(results), "results": results})
