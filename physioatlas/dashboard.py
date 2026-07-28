from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Union


class LiveStateStore:
    def __init__(self, state_path: Union[str, Path]):
        self.path = Path(state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.publish({"status": "waiting", "research_only": True})

    def publish(self, state: dict[str, Any]) -> None:
        payload = dict(state)
        payload.setdefault("research_only", True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(self.path)

    def read(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(self.path.read_text(encoding="utf-8"))


class DashboardServer:
    def __init__(
        self,
        store: LiveStateStore,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        static_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.store = store
        self.host = host
        self.port = port
        self.static_dir = Path(static_dir) if static_dir else Path(__file__).resolve().parent / "static"
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/health":
                    self._json({"status": "ok", "research_only": True})
                    return
                if self.path == "/api/state":
                    self._json(outer.store.read())
                    return
                relative = "index.html" if self.path in {"/", ""} else self.path.lstrip("/")
                target = (outer.static_dir / relative).resolve()
                if outer.static_dir.resolve() not in target.parents and target != outer.static_dir.resolve():
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                if not target.exists() or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content_type = "text/html" if target.suffix == ".html" else "text/plain"
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _json(self, payload: dict[str, Any]) -> None:
                data = json.dumps(payload, sort_keys=True).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
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
