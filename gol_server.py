#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local web server for GraphOfLife.

Runs entirely on your machine: it serves the viewer from web/ and exposes a
small JSON API for creating, running, resuming and inspecting simulations.
Nothing is published anywhere — generate data on whatever machine you are
sitting at, and inspect it there.

    python3 gol_server.py --port 8000

Standard library only, so cloning the repo onto a new machine needs no install
step beyond numpy and networkx.

API
---
    GET    /api/defaults              default config plus form metadata
    GET    /api/runs                  every run on this machine
    POST   /api/runs                  create a run  {name, config}
    GET    /api/runs/<id>             one run's metadata
    DELETE /api/runs/<id>             delete a run and all its data
    POST   /api/runs/<id>/start       start or resume  {steps?}
    POST   /api/runs/<id>/stop        ask a running worker to stop
    GET    /api/runs/<id>/frames/<n>  one recorded frame
    GET    /api/runs/<id>/series      per-frame statistics for the whole run
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

import gol_store as store
from gol_config import SimConfig

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# Requests are capped so a malformed or hostile body cannot exhaust memory.
MAX_BODY_BYTES = 1 << 20


# ----------------------------------------------------------------------------
# Workers — one background thread per running simulation
# ----------------------------------------------------------------------------

class Worker:
    """
    Advances a single run in the background.

    A worker either starts a brand new world or resumes from the run's
    checkpoint. On resume it first truncates any frames recorded past the
    checkpoint, because those describe a timeline the resumed world will not
    reproduce.
    """

    def __init__(self, run_id: str, steps: int | None) -> None:
        self.run_id = run_id
        self.steps = steps
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"gol-{run_id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()

    def _run(self) -> None:
        run_id = self.run_id
        try:
            from GraphOfLifeSimple import new_world

            cfg = store.load_config(run_id)

            world = store.load_checkpoint(run_id, cfg)
            if world is None:
                # Fresh start: nothing from a previous attempt survives.
                store.truncate_frames_from(run_id, 0)
                world = new_world(cfg)
                frame_cursor = 0
            else:
                # Resume: drop the future the checkpoint never lived through.
                frame_cursor = store.frames_recorded_before(world.iteration, cfg.export_every)
                store.truncate_frames_from(run_id, frame_cursor)

            store.update_meta(run_id, status="running", error=None,
                              iteration=world.iteration, frame_count=frame_cursor)

            budget = self.steps if self.steps and self.steps > 0 else cfg.max_steps
            completed = 0
            final_status = "idle"

            while completed < budget and world.iteration < cfg.max_steps:
                if self.stop_event.is_set():
                    final_status = "stopped"
                    break

                record = (world.iteration % cfg.export_every == 0)
                frames = world.step(record_decisions=cfg.export_decisions and record)

                if record:
                    for frame in frames:
                        store.write_frame(run_id, frame_cursor, frame)
                        frame_cursor += 1

                completed += 1

                if cfg.checkpoint_every and world.iteration % cfg.checkpoint_every == 0:
                    store.save_checkpoint(run_id, world)

                store.update_meta(run_id, iteration=world.iteration, frame_count=frame_cursor)

                if world.is_extinct():
                    final_status = "extinct"
                    break

            # However the loop ended, leave behind a checkpoint so the run can
            # always be picked up again from exactly where it stopped.
            if cfg.checkpoint_every:
                store.save_checkpoint(run_id, world)
            store.update_meta(run_id, status=final_status,
                              iteration=world.iteration, frame_count=frame_cursor)

        except Exception:
            traceback.print_exc()
            try:
                store.update_meta(run_id, status="error", error=traceback.format_exc(limit=4))
            except OSError:
                pass


class WorkerPool:
    """Tracks the live workers, one per run at most."""

    def __init__(self) -> None:
        self._workers: Dict[str, Worker] = {}
        self._lock = threading.Lock()

    def start(self, run_id: str, steps: int | None) -> bool:
        with self._lock:
            existing = self._workers.get(run_id)
            if existing and existing.alive:
                return False
            worker = Worker(run_id, steps)
            self._workers[run_id] = worker
        worker.start()
        return True

    def stop(self, run_id: str) -> bool:
        with self._lock:
            worker = self._workers.get(run_id)
        if not worker or not worker.alive:
            return False
        worker.stop()
        return True

    def is_running(self, run_id: str) -> bool:
        with self._lock:
            worker = self._workers.get(run_id)
        return bool(worker and worker.alive)

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.stop()


POOL = WorkerPool()


# ----------------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "GraphOfLife/1.0"

    # ---- plumbing --------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        # One tidy line per request instead of the default noise.
        print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")) or {}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"malformed JSON body: {exc}") from exc

    # ---- routing ---------------------------------------------------------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self._route_get(path)
        else:
            self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            self._route_post(path)
        except ValueError as exc:
            self._error(str(exc))
        except FileNotFoundError:
            self._error("run not found", 404)
        except Exception as exc:  # noqa: BLE001 - surface failures to the UI
            traceback.print_exc()
            self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        parts = [p for p in path.split("/") if p]
        if len(parts) == 3 and parts[:2] == ["api", "runs"]:
            try:
                POOL.stop(parts[2])
                store.delete_run(parts[2])
                self._send_json({"ok": True})
            except ValueError as exc:
                self._error(str(exc))
            return
        self._error("not found", 404)

    def _route_get(self, path: str) -> None:
        parts = [p for p in path.split("/") if p]
        try:
            if parts == ["api", "defaults"]:
                self._send_json(self._defaults())
                return

            if parts == ["api", "runs"]:
                self._send_json({"runs": [self._decorate(m) for m in store.list_runs()]})
                return

            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                self._send_json(self._decorate(store.load_meta(parts[2])))
                return

            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "series":
                import gol_series
                self._send_json(gol_series.build_series(parts[2]))
                return

            if len(parts) == 5 and parts[:2] == ["api", "runs"] and parts[3] == "frames":
                run_id, index = parts[2], int(parts[4])
                if not store.has_frame(run_id, index):
                    self._error("frame not found", 404)
                    return
                self._send_json(store.read_frame(run_id, index))
                return

        except FileNotFoundError:
            self._error("run not found", 404)
            return
        except ValueError as exc:
            self._error(str(exc))
            return

        self._error("not found", 404)

    def _route_post(self, path: str) -> None:
        parts = [p for p in path.split("/") if p]

        if parts == ["api", "runs"]:
            body = self._read_json()
            cfg = SimConfig.from_dict(body.get("config", {}))
            meta = store.create_run(body.get("name", ""), cfg)
            self._send_json(self._decorate(meta), 201)
            return

        if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "start":
            run_id = parts[2]
            store.load_meta(run_id)  # 404s if the run is unknown
            body = self._read_json()
            steps = body.get("steps")
            steps = int(steps) if steps else None
            if not POOL.start(run_id, steps):
                self._error("run is already in progress", 409)
                return
            self._send_json({"ok": True})
            return

        if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "stop":
            run_id = parts[2]
            store.load_meta(run_id)
            POOL.stop(run_id)
            self._send_json({"ok": True})
            return

        self._error("not found", 404)

    # ---- payload helpers -------------------------------------------------

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        cfg = SimConfig()
        return {
            "config": cfg.to_dict(),
            "derived": {
                "n_inputs": cfg.n_inputs(),
                "n_outputs": cfg.n_outputs(),
                "resolved_n": cfg.resolved_n(),
                "resolved_k": cfg.resolved_k(),
                "heads": cfg.head_layout(),
            },
        }

    @staticmethod
    def _decorate(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Attach live facts the metadata file cannot know on its own."""
        run_id = meta["id"]
        meta = dict(meta)
        meta["running"] = POOL.is_running(run_id)
        meta["has_checkpoint"] = store.has_checkpoint(run_id)
        try:
            meta["size_bytes"] = store.run_size_bytes(run_id)
        except (OSError, ValueError):
            meta["size_bytes"] = 0
        # A stale "running" status survives a server restart; correct it here so
        # the UI never shows a run as live when no worker exists.
        if meta.get("status") == "running" and not meta["running"]:
            meta["status"] = "interrupted"
        return meta

    # ---- static files ----------------------------------------------------

    def _serve_static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = os.path.abspath(os.path.join(WEB_DIR, rel))

        # Never serve anything outside web/.
        if not target.startswith(os.path.abspath(WEB_DIR) + os.sep) and target != os.path.abspath(WEB_DIR):
            self._error("forbidden", 403)
            return
        if not os.path.isfile(target):
            self._error("not found", 404)
            return

        ctype, _ = mimetypes.guess_type(target)
        try:
            with open(target, "rb") as f:
                body = f.read()
        except OSError:
            self._error("could not read file", 500)
            return

        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local GraphOfLife web server")
    parser.add_argument("--port", type=int, default=8000)
    # Binds to localhost by default: these runs are yours and stay on this box.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"🌍 GraphOfLife running at {url}")
    print(f"   runs stored in {store.BASE_DIR}")
    print("   Ctrl-C to stop")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping workers…")
        POOL.stop_all()
        httpd.shutdown()


if __name__ == "__main__":
    main()
