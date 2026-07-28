from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Union

from .utils import make_json_safe


class ResearchHubStore:
    """Atomic local state for the household research dashboard."""

    def __init__(self, state_path: Union[str, Path]):
        self.path = Path(state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.action_path = self.path.with_name("research-hub-actions.jsonl")
        self._lock = threading.RLock()
        if not self.path.exists():
            self.publish(
                {
                    "schema_version": "physioatlas.research-hub-state.v1",
                    "status": "waiting",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "household": {"members": [], "tracks": []},
                    "modalities": {},
                    "calibration": {},
                    "studies": [],
                    "experiments": [],
                    "events": [],
                    "privacy": {
                        "local_only": True,
                        "consent_required": True,
                        "unknown_people_anonymous": True,
                    },
                    "research_only": True,
                }
            )

    def read(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(self.path.read_text(encoding="utf-8"))

    def publish(self, state: dict[str, Any]) -> None:
        payload = make_json_safe(dict(state))
        payload.setdefault("schema_version", "physioatlas.research-hub-state.v1")
        payload.setdefault("updated_at_utc", datetime.now(timezone.utc).isoformat())
        payload.setdefault("research_only", True)
        payload.setdefault(
            "privacy",
            {"local_only": True, "consent_required": True, "unknown_people_anonymous": True},
        )
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(self.path)

    def update(self, **sections: Any) -> dict[str, Any]:
        state = self.read()
        state.update(sections)
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.publish(state)
        return state

    def append_event(self, event_type: str, payload: dict[str, Any], *, maximum: int = 200) -> dict[str, Any]:
        state = self.read()
        events = list(state.get("events", []))
        events.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                "payload": make_json_safe(payload),
            }
        )
        state["events"] = events[-maximum:]
        self.publish(state)
        return state

    def enqueue_action(self, action: str, parameters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        allowed = {
            "start_recording",
            "stop_recording",
            "run_calibration",
            "run_priority_studies",
            "clear_events",
        }
        if action not in allowed:
            raise ValueError(f"Unsupported research-hub action: {action}")
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "parameters": make_json_safe(parameters or {}),
            "status": "queued",
        }
        with self._lock, self.action_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        self.append_event("action_queued", entry)
        return entry


class ResearchHubServer:
    def __init__(
        self,
        store: ResearchHubStore,
        *,
        host: str = "127.0.0.1",
        port: int = 8770,
        static_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.store = store
        self.host = host
        self.port = port
        self.static_dir = Path(static_dir) if static_dir else Path(__file__).resolve().parent / "static" / "research_hub"
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/api/health":
                    self._json({"status": "ok", "research_only": True, "local_only": True})
                    return
                state = outer.store.read()
                endpoint_map = {
                    "/api/hub": state,
                    "/api/tracks": state.get("household", {}).get("tracks", []),
                    "/api/members": state.get("household", {}).get("members", []),
                    "/api/modalities": state.get("modalities", {}),
                    "/api/calibration": state.get("calibration", {}),
                    "/api/studies": state.get("studies", []),
                    "/api/experiments": state.get("experiments", []),
                    "/api/events": state.get("events", []),
                }
                if path in endpoint_map:
                    self._json(endpoint_map[path])
                    return
                relative = "index.html" if path in {"/", ""} else path.lstrip("/")
                base = outer.static_dir.resolve()
                target = (base / relative).resolve()
                if target != base and base not in target.parents:
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                if not target.exists() or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content_types = {
                    ".html": "text/html; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".json": "application/json",
                    ".svg": "image/svg+xml",
                }
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_types.get(target.suffix, "application/octet-stream"))
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path != "/api/actions":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 64_000:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                try:
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict) or not isinstance(payload.get("action"), str):
                        raise ValueError("Action body must be an object with an action string")
                    result = outer.store.enqueue_action(payload["action"], payload.get("parameters"))
                except Exception as exc:
                    self._json({"status": "error", "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json({"status": "queued", "entry": result}, status=HTTPStatus.ACCEPTED)

            def _json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
                data = json.dumps(make_json_safe(payload), sort_keys=True).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: object) -> None:
                return

        self.httpd = ThreadingHTTPServer((host, port), Handler)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
