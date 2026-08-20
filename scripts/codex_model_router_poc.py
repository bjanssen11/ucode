#!/usr/bin/env python3
"""POC: a minimal interactive Codex client that can switch models mid-session.

Launches `codex app-server` under the hood, gives you a prompt, and lets you
change the model live with `/model <name>` — the switch happens by setting the
per-turn `model` field on `turn/start`, so history is preserved across it.

This is arch A from the plan (a thin app-server client). It is NOT Codex's
polished TUI; it's the smallest thing that proves "launch, type, switch".

Run it with the repo's Python 3.12 venv (system python3 here is 3.6):

    /home/lilly.luo/ucode/.venv/bin/python scripts/codex_model_router_poc.py

For interactive mode, omit all flags. For a self-test that proves mid-session
model switching with context preservation:

    /home/lilly.luo/ucode/.venv/bin/python scripts/codex_model_router_poc.py --selftest

Auth/gateway config is generated from your existing ~/.codex/ucode.config.toml
provider block into an isolated CODEX_HOME, so it uses the same Databricks
gateway + `ucode auth-token` refresh that `ucode codex` uses.

In-session commands:
    /model <name>   switch the model for subsequent turns (e.g. /model gpt-5.5)
    /model          show the current model
    /quit           exit
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import tomlkit

UCODE_CODEX_CONFIG = Path.home() / ".codex" / "ucode.config.toml"
POC_HOME = Path.home() / ".cache" / "ucode-codex-router-poc"
DEFAULT_MODEL = "system.ai.gpt-5-6-luna"
EXAMPLE_MODELS = ["system.ai.gpt-5-6-luna", "gpt-5.5"]


def build_codex_home() -> Path:
    """Generate an isolated CODEX_HOME whose config.toml carries ONLY the ucode
    gateway provider block (model_provider + model + [model_providers.*]), copied
    from ~/.codex/ucode.config.toml. Keeps the app-server pointed at the same
    Databricks gateway + auth-token refresh, without the hooks/tui cruft."""
    if not UCODE_CODEX_CONFIG.exists():
        sys.exit(
            f"Missing {UCODE_CODEX_CONFIG}. Run `ucode configure codex` (or `ucode codex`) first "
            "so the Databricks provider block exists."
        )
    src = tomllib.loads(UCODE_CODEX_CONFIG.read_text())
    minimal = tomlkit.document()
    if "model_provider" in src:
        minimal["model_provider"] = src["model_provider"]
    minimal["model"] = src.get("model", DEFAULT_MODEL)
    if "model_reasoning_effort" in src:
        minimal["model_reasoning_effort"] = src["model_reasoning_effort"]
    if "model_providers" in src:
        minimal["model_providers"] = src["model_providers"]
    POC_HOME.mkdir(parents=True, exist_ok=True)
    (POC_HOME / "config.toml").write_text(tomlkit.dumps(minimal))
    return POC_HOME


class AppServer:
    """Thin newline-delimited-JSON stdio client for `codex app-server`."""

    def __init__(self, codex_home: Path) -> None:
        env = dict(os.environ)
        env["CODEX_HOME"] = str(codex_home)
        self.proc = subprocess.Popen(
            ["codex", "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._q: queue.Queue = queue.Queue()
        self._id = 0
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if line:
                try:
                    self._q.put(json.loads(line))
                except ValueError:
                    pass

    def _drain_stderr(self) -> None:
        # app-server logs benign catalog-refresh 404s here; keep them out of the UI
        # but available if the user wants them (uncomment to debug).
        for _line in self.proc.stderr:  # type: ignore[union-attr]
            pass

    def _send(self, method: str, params: dict | None = None, *, notify: bool = False):
        msg: dict = {"method": method}
        if not notify:
            self._id += 1
            msg["id"] = self._id
        if params is not None:
            msg["params"] = params
        self.proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
        self.proc.stdin.flush()  # type: ignore[union-attr]
        return msg.get("id")

    def _wait(self, pred, timeout: float):
        end = time.time() + timeout
        while time.time() < end:
            try:
                msg = self._q.get(timeout=min(1.0, max(0.05, end - time.time())))
            except queue.Empty:
                continue
            if pred(msg):
                return msg
        return None

    def request(self, method: str, params: dict | None = None, *, timeout: float = 60.0):
        rid = self._send(method, params)
        return self._wait(lambda m: m.get("id") == rid and ("result" in m or "error" in m), timeout)

    def initialize(self) -> None:
        self.request(
            "initialize",
            {"clientInfo": {"name": "ucode-codex-router-poc", "version": "0.1"}, "capabilities": {}},
            timeout=30,
        )
        self._send("initialized", {}, notify=True)

    def start_thread(self, model: str) -> str:
        resp = self.request(
            "thread/start", {"model": model, "cwd": os.getcwd(), "approvalPolicy": "never"}, timeout=60
        )
        result = (resp or {}).get("result", {})
        tid = result.get("thread", {}).get("id") or result.get("threadId")
        if not tid:
            sys.exit(f"thread/start failed: {json.dumps(resp)[:400]}")
        return tid

    def run_turn(self, thread_id: str, text: str, model: str, *, timeout: float = 300.0) -> None:
        """Send one user turn on `model`, streaming assistant text to stdout live."""
        rid = self._send(
            "turn/start",
            {"threadId": thread_id, "input": [{"type": "text", "text": text}], "model": model},
        )
        # Ack (status inProgress) — then stream until turn/completed.
        self._wait(lambda m: m.get("id") == rid and ("result" in m or "error" in m), 30)
        end = time.time() + timeout
        printed_any = False
        while time.time() < end:
            try:
                msg = self._q.get(timeout=min(1.0, max(0.05, end - time.time())))
            except queue.Empty:
                continue
            method = msg.get("method")
            params = msg.get("params") or {}
            if method == "item/agentMessage/delta":
                delta = _find_str(params, ("delta", "text"))
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    printed_any = True
            elif method == "turn/completed":
                turn = params.get("turn", {})
                if turn.get("status") == "failed":
                    err = turn.get("error", {})
                    print(f"\n  [turn failed: {err.get('message', err)}]")
                elif not printed_any:
                    # No deltas seen (some models don't stream) — print final items.
                    print(_final_text(turn) or "  [no text returned]")
                print()
                return
        print("\n  [timed out waiting for the turn to complete]")

    def close(self) -> None:
        try:
            self.proc.stdin.close()  # type: ignore[union-attr]
        except Exception:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def _find_str(obj, keys) -> str | None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, str):
                return v
            r = _find_str(v, keys)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_str(v, keys)
            if r:
                return r
    return None


def _final_text(turn: dict) -> str:
    out = []
    for item in turn.get("items", []) or []:
        if isinstance(item, dict) and item.get("type") == "agentMessage":
            t = item.get("text")
            if t:
                out.append(t)
    return "\n".join(out)


def _capture_turn_text(server: AppServer, thread_id: str, text: str, model: str) -> str:
    """Run a turn and capture the full assistant response text."""
    rid = server._send(
        "turn/start",
        {"threadId": thread_id, "input": [{"type": "text", "text": text}], "model": model},
    )
    # Ack (status inProgress) — then stream until turn/completed.
    server._wait(lambda m: m.get("id") == rid and ("result" in m or "error" in m), 30)
    captured_text = []
    end = time.time() + 300.0
    while time.time() < end:
        try:
            msg = server._q.get(timeout=min(1.0, max(0.05, end - time.time())))
        except queue.Empty:
            continue
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "item/agentMessage/delta":
            delta = _find_str(params, ("delta", "text"))
            if delta:
                captured_text.append(delta)
        elif method == "turn/completed":
            turn = params.get("turn", {})
            if turn.get("status") == "failed":
                err = turn.get("error", {})
                return f"[FAILED: {err.get('message', err)}]"
            # Collect any remaining text from final items
            final = _final_text(turn)
            if final and not captured_text:
                captured_text.append(final)
            return "".join(captured_text)
    return "[TIMEOUT]"


def selftest() -> int:
    """Non-interactive self-test: prove mid-session model switch with context."""
    home = build_codex_home()
    server = AppServer(home)
    try:
        print("Starting codex app-server for self-test…")
        server.initialize()
        thread_id = server.start_thread(DEFAULT_MODEL)
        print(f"Thread created with model {DEFAULT_MODEL}")

        # Turn 1: simple model A request
        print("\n=== Turn 1 (model A) ===")
        t1_prompt = "Reply with exactly: TURN1_OK"
        print(f"Prompt: {t1_prompt}")
        t1_response = _capture_turn_text(server, thread_id, t1_prompt, "system.ai.gpt-5-6-luna")
        print(f"Response: {t1_response!r}")
        if "TURN1_OK" not in t1_response:
            print(f"ERROR: Turn 1 did not contain TURN1_OK")
            return 1

        # Turn 2: switch model and test context preservation
        print("\n=== Turn 2 (model B, testing context) ===")
        t2_prompt = "What token did you reply on the previous turn? Then say TURN2_OK."
        print(f"Switching to gpt-5.5…")
        print(f"Prompt: {t2_prompt}")
        t2_response = _capture_turn_text(server, thread_id, t2_prompt, "gpt-5.5")
        print(f"Response: {t2_response!r}")

        # Verify context was preserved: t2 should mention TURN1_OK
        if "TURN1_OK" not in t2_response:
            print(f"ERROR: Turn 2 did not contain TURN1_OK (context not preserved)")
            return 1

        if "TURN2_OK" not in t2_response:
            print(f"WARNING: Turn 2 did not contain TURN2_OK (but context was preserved)")

        print("\n=== SUCCESS ===")
        print("Mid-session model switch with context preservation verified!")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        server.close()


def main() -> int:
    # Parse command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--selftest":
            return selftest()
        elif sys.argv[1] in ("--help", "-h"):
            print(__doc__)
            return 0
        else:
            print(f"Unknown argument: {sys.argv[1]}", file=sys.stderr)
            print(f"Use: {sys.argv[0]} [--selftest] [--help]", file=sys.stderr)
            return 1

    # Interactive mode
    home = build_codex_home()
    server = AppServer(home)
    current_model = DEFAULT_MODEL
    try:
        print("Starting codex app-server…")
        server.initialize()
        thread_id = server.start_thread(current_model)
        print(f"\nCodex ready. model = {current_model}")
        print(f"Commands: /model <name>   /quit    (try: {', '.join(EXAMPLE_MODELS)})\n")
        while True:
            try:
                line = input(f"[{current_model}] › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line == "/quit":
                break
            if line.startswith("/model"):
                arg = line[len("/model"):].strip()
                if not arg:
                    print(f"  current model: {current_model}")
                else:
                    current_model = arg
                    print(f"  → switched to {current_model} (applies to the next turn; history kept)")
                continue
            server.run_turn(thread_id, line, current_model)
        return 0
    finally:
        server.close()


if __name__ == "__main__":
    sys.exit(main())
