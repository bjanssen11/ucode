#!/usr/bin/env python3
"""Arch B: a WebSocket MITM that lets you keep the REAL Codex TUI while the model
is switched under program control.

Codex's remote transport (`codex --remote ws://…`) is WebSocket (a plain-JSONL
client is rejected with HTTP 400 "Connection header did not include 'upgrade'";
a proper upgrade returns 101). Each JSON-RPC message is one WebSocket text frame.
This proxy sits between the TUI and a real `codex app-server`, forwarding every
frame untouched except:

  - `turn/start` (TUI->engine): after an initial hold of `--after` turns, its
    `model` is rewritten to `--model`. `turn/start.model` is documented as
    "override the model for this turn and subsequent turns", so the live session
    retargets with history preserved.
  - When the hold expires (right after your Nth prompt completes) it INJECTS a
    `thread/settings/updated` notification (engine->TUI) carrying the new model,
    so the TUI's on-screen model indicator follows the switch.

So the demo is: start the TUI on model X, submit your first prompt (answered by
X), and from then on the session runs on `--model` (and the chip flips to it).

Topology:
    codex app-server --listen ws://127.0.0.1:8801                (real engine)
    this interposer  ws://127.0.0.1:8802 -> ws://127.0.0.1:8801   (switches model)
    codex --remote   ws://127.0.0.1:8802 --model system.ai.gpt-5-6-luna   (real TUI)

Run via uv so nothing is installed globally:

    uv run --with websockets python scripts/codex_tui_interposer.py \
        --listen 127.0.0.1:8802 --upstream ws://127.0.0.1:8801 \
        --model gpt-5.5 --after 1

Self-test (spawns its own app-server + a simulated TUI; proves hold + switch end
to end against the gateway):

    uv run --with websockets --with tomlkit python \
        scripts/codex_tui_interposer.py --selftest
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

SETTINGS_UPDATED = "thread/settings/updated"


class Session:
    """Per-TUI-connection state: hold the first `after` turns, then switch model."""

    def __init__(self, target_model: str, after: int, log) -> None:
        self.target = target_model
        self.after = after
        self.log = log
        self.turns = 0
        self.thread_id: str | None = None
        self.settings: dict | None = None
        self.injected = False

    def on_tui_frame(self, raw: str) -> str:
        """TUI->engine: rewrite turn/start.model once past the hold."""
        try:
            msg = json.loads(raw)
        except ValueError:
            return raw
        if not isinstance(msg, dict):
            return raw
        params = msg.get("params")
        if msg.get("method") == "turn/start" and isinstance(params, dict):
            self.turns += 1
            if isinstance(params.get("threadId"), str):
                self.thread_id = params["threadId"]
            if self.turns > self.after:
                old = params.get("model")
                if old != self.target:
                    params["model"] = self.target
                    self.log(f"[REWRITE] turn #{self.turns}: model {old!r} -> {self.target!r}")
                    return json.dumps(msg)
        return raw

    def on_engine_frame(self, raw: str):
        """engine->TUI: capture thread id/settings; after the hold's last turn
        completes, return an injected settings-updated notification (or None)."""
        try:
            msg = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(msg, dict):
            return None
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        result = msg.get("result") if isinstance(msg.get("result"), dict) else {}
        # Capture threadId + a real threadSettings object wherever it appears.
        for src in (params, result):
            tid = src.get("threadId") or (src.get("thread") or {}).get("id")
            if isinstance(tid, str):
                self.thread_id = tid
            ts = src.get("threadSettings")
            if isinstance(ts, dict):
                self.settings = ts
        # When the hold's final turn completes, flip the on-screen model.
        if (
            msg.get("method") == "turn/completed"
            and not self.injected
            and self.turns >= self.after
            and self.thread_id
        ):
            self.injected = True
            settings = dict(self.settings) if isinstance(self.settings, dict) else {}
            settings["model"] = self.target
            self.log(f"[INJECT] {SETTINGS_UPDATED}: model -> {self.target!r} (flip TUI chip)")
            return {
                "method": SETTINGS_UPDATED,
                "params": {"threadId": self.thread_id, "threadSettings": settings},
            }
        return None


async def _handle_tui(tui, upstream_uri: str, target_model: str, after: int, log) -> None:
    path = getattr(getattr(tui, "request", None), "path", "/") or "/"
    uri = upstream_uri.rstrip("/") + path
    log(f"[CONN] TUI connected (path={path}); dialing app-server {uri}")
    sess = Session(target_model, after, log)
    async with connect(uri, max_size=None) as upstream:

        async def tui_to_app():
            async for frame in tui:
                if isinstance(frame, str):
                    frame = sess.on_tui_frame(frame)
                await upstream.send(frame)

        async def app_to_tui():
            async for frame in upstream:
                await tui.send(frame)
                if isinstance(frame, str):
                    inj = sess.on_engine_frame(frame)
                    if inj is not None:
                        await tui.send(json.dumps(inj))

        a = asyncio.create_task(tui_to_app())
        b = asyncio.create_task(app_to_tui())
        _done, pending = await asyncio.wait({a, b}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
    log("[CONN] TUI session closed")


async def serve_interposer(host: str, port: int, upstream_uri: str, model: str, after: int, *, quiet=False):
    def log(m: str) -> None:
        if not quiet:
            print(m, file=sys.stderr, flush=True)

    async def handler(tui):
        try:
            await _handle_tui(tui, upstream_uri, model, after, log)
        except Exception as exc:  # noqa: BLE001 - one session must not kill the server
            log(f"[ERR] session: {exc!r}")

    server = await serve(handler, host, port, max_size=None)
    log(f"[READY] ws://{host}:{port} -> {upstream_uri}  (hold {after} turn(s), then switch to {model!r})")
    return server


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

UCODE_CODEX_CONFIG = Path.home() / ".codex" / "ucode.config.toml"
SELFTEST_HOME = Path.home() / ".cache" / "ucode-codex-interposer"
START_MODEL = "system.ai.gpt-5-6-luna"
TARGET_MODEL = "gpt-5.5"
BOGUS_MODEL = "totally-bogus-model-zzz"


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _build_codex_home() -> Path:
    import tomllib
    import tomlkit

    if not UCODE_CODEX_CONFIG.exists():
        sys.exit(f"Missing {UCODE_CODEX_CONFIG}; run `ucode codex` once so the provider block exists.")
    src = tomllib.loads(UCODE_CODEX_CONFIG.read_text())
    doc = tomlkit.document()
    for k in ("model_provider", "model", "model_reasoning_effort", "model_providers"):
        if k in src:
            doc[k] = src[k]
    SELFTEST_HOME.mkdir(parents=True, exist_ok=True)
    (SELFTEST_HOME / "config.toml").write_text(tomlkit.dumps(doc))
    return SELFTEST_HOME


async def _wait_healthz(port: int, timeout: float = 30.0) -> bool:
    import urllib.request

    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            await asyncio.sleep(0.25)
    return False


async def _simulated_tui(port: int) -> dict:
    """Turn 1 uses START_MODEL (should pass through). Turn 2 sends a BOGUS model
    (should be rewritten to TARGET_MODEL and therefore succeed). Also watches for
    the injected settings-updated frame after turn 1."""
    out = {"t1": None, "t2": None, "injected_model": None, "error": None}
    nid = 0
    async with connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
        async def send(method, params=None, notify=False):
            nonlocal nid
            m = {"method": method}
            if not notify:
                nid += 1
                m["id"] = nid
            if params is not None:
                m["params"] = params
            await ws.send(json.dumps(m))
            return m.get("id")

        async def until(pred, timeout=180):
            end = time.time() + timeout
            while time.time() < end:
                try:
                    frame = await asyncio.wait_for(ws.recv(), timeout=min(5, end - time.time()))
                except asyncio.TimeoutError:
                    continue
                if not isinstance(frame, str):
                    continue
                try:
                    msg = json.loads(frame)
                except ValueError:
                    continue
                if msg.get("method") == SETTINGS_UPDATED:
                    out["injected_model"] = (msg.get("params", {}).get("threadSettings", {}) or {}).get("model")
                if pred(msg):
                    return msg
            return None

        await send("initialize", {"clientInfo": {"name": "sim", "version": "0"}, "capabilities": {}})
        await until(lambda m: m.get("id") == 1 and ("result" in m or "error" in m), 30)
        await send("initialized", {}, notify=True)
        rid = await send("thread/start", {"model": START_MODEL, "cwd": os.getcwd(), "approvalPolicy": "never"})
        ts = await until(lambda m: m.get("id") == rid and "result" in m, 60)
        tid = ((ts or {}).get("result", {}).get("thread", {}) or {}).get("id")
        if not tid:
            out["error"] = f"thread/start failed: {json.dumps(ts)[:200]}"
            return out
        await send("turn/start", {"threadId": tid, "input": [{"type": "text", "text": "Say A"}], "model": START_MODEL})
        tc1 = await until(lambda m: m.get("method") == "turn/completed", 180)
        out["t1"] = (tc1 or {}).get("params", {}).get("turn", {}).get("status")
        await send("turn/start", {"threadId": tid, "input": [{"type": "text", "text": "Say B"}], "model": BOGUS_MODEL})
        tc2 = await until(lambda m: m.get("method") == "turn/completed", 180)
        out["t2"] = (tc2 or {}).get("params", {}).get("turn", {}).get("status")
    return out


async def _selftest() -> int:
    home = _build_codex_home()
    port_a, port_b = _free_port(), _free_port()
    env = dict(os.environ); env["CODEX_HOME"] = str(home)
    print(f"Starting codex app-server on ws://127.0.0.1:{port_a} …")
    proc = subprocess.Popen(
        ["codex", "app-server", "--listen", f"ws://127.0.0.1:{port_a}"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    server = None
    try:
        if not await _wait_healthz(port_a):
            print("app-server did not become healthy", file=sys.stderr)
            return 1
        server = await serve_interposer("127.0.0.1", port_b, f"ws://127.0.0.1:{port_a}", TARGET_MODEL, after=1)
        print(f"Interposer up: hold 1 turn on the TUI's model, then switch -> {TARGET_MODEL!r}\n")
        r = await _simulated_tui(port_b)
        print()
        ok = (
            r["t1"] == "completed"                       # turn 1 ran on the pass-through START_MODEL
            and r["t2"] == "completed"                   # turn 2 sent BOGUS but was rewritten -> succeeded
            and r["injected_model"] == TARGET_MODEL      # settings-updated injected to flip the chip
        )
        print("=== RESULT ===")
        print(f"  turn1 (start model, passthrough): {r['t1']}")
        print(f"  turn2 (client sent BOGUS -> rewritten): {r['t2']}")
        print(f"  injected settings-updated model: {r['injected_model']!r}")
        print(f"  error: {r['error']!r}")
        print("=== SUCCESS ===" if ok else "=== FAILED ===")
        return 0 if ok else 1
    finally:
        if server is not None:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)


def main() -> int:
    ap = argparse.ArgumentParser(description="WebSocket MITM interposer for the Codex TUI (arch B).")
    ap.add_argument("--listen", default="127.0.0.1:8802", help="host:port for the TUI to connect to")
    ap.add_argument("--upstream", default="ws://127.0.0.1:8801", help="real app-server ws:// URI")
    ap.add_argument("--model", default=TARGET_MODEL, help="model to switch to after the hold")
    ap.add_argument("--after", type=int, default=1, help="pass through this many turns before switching (default 1)")
    ap.add_argument("--selftest", action="store_true", help="spawn app-server + simulated TUI and prove hold+switch")
    args = ap.parse_args()

    if args.selftest:
        return asyncio.run(_selftest())

    host, _, port = args.listen.partition(":")

    async def _run():
        await serve_interposer(host, int(port), args.upstream, args.model, args.after)
        await asyncio.Future()

    try:
        return asyncio.run(_run()) or 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
