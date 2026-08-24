"""PTY wrapper for Claude Code's TUI (smart routing v2).

Claude Code has no ``app-server``/JSON-RPC seam like Codex, so there is nothing to
interpose on the wire. Instead this module runs the real ``claude`` TUI inside a PTY:
it forwards stdin<->master and master<->stdout untouched, and drives a *model switch*
by typing ``/model <name>`` into the TUI and auto-confirming the "Switch model?" dialog
by watching the PTY output. A Unix-domain control socket accepts line-delimited JSON-RPC
``model.set`` requests (parity with the reference POC and a clean test surface); the
automatic first-prompt CUJ drives the same ``on_model_set`` path internally.

``ucode.agents.claude`` owns the lifecycle: it enters this from the single ``ucode claude``
command when ``ENABLE_SMART_ROUTING_V2=1``. Logs go to ``log_path`` (appended) only — never
stdout/stderr, which the foreground TUI owns (same discipline as ``codex_interposer``).

Detecting *when the user submits their first prompt* is done from **stdin** (the Enter
keystroke after typed content), not by scraping output — Claude Code's TUI renders spaces
as cursor moves and uses randomized spinner text, so output markers are unreliable. Marker
matching that remains (readiness, the confirm dialog) is whitespace-insensitive (see
:func:`_squash`). The only runtime-spike output constants left are :data:`READY_MARKERS`
(pre-emptive mode only) and :data:`ConfirmationState.PROMPT_MARKERS`.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pty
import re
import select
import signal
import socket
import struct
import termios
import threading
import time
import tty
from collections.abc import Callable
from pathlib import Path

# --- runtime-spike markers (whitespace-squashed substrings; tune from a real claude launch) ---
# TUI is ready for input. A *bonus* signal for the pre-emptive switch mode; the primary
# readiness signal is version-independent (idle after the initial paint — see READY_QUIET_S).
# Matching is whitespace-insensitive (see _squash), because Claude Code renders spaces as
# cursor moves that vanish under strip_ansi.
READY_MARKERS = ("? for shortcuts", "Welcome to Claude Code")

MAX_MODEL_NAME_LEN = 200
CONFIRM_TIMEOUT_S = 5.0
# Pre-emptive readiness heuristic: once the TUI has produced output and then stays quiet
# this long, its initial paint is done and it is idle waiting for input — safe to type
# `/model`. Marker-independent, so it survives Claude Code TUI changes.
READY_QUIET_S = 0.75
# select() wake interval while waiting to switch, so the quiet period is observable.
SELECT_TIMEOUT_S = 0.2
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:/\-\[\]]+$")
# CSI/OSC/simple escape sequences — enough to make substring matching robust across
# the styled bytes Claude Code's Ink renderer emits.
_ANSI_RE = re.compile(rb"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


def strip_ansi(data: bytes) -> str:
    """Drop ANSI escape sequences and decode to text for substring matching."""
    return _ANSI_RE.sub(b"", data).decode("utf-8", "replace")


def _squash(text: str) -> str:
    """Remove ALL whitespace. Claude Code's TUI positions words with cursor-move
    escapes rather than literal spaces, so after stripping ANSI the words run
    together (e.g. 'esc to interrupt' -> 'esctointerrupt'). Squashing both the
    observed text and the markers makes multi-word markers match regardless."""
    return "".join(text.split())


def _match_text(data: bytes) -> str:
    """ANSI-stripped, whitespace-squashed text for robust marker matching."""
    return _squash(strip_ansi(data))


def valid_model_name(name: object) -> bool:
    """Guard the value before it is typed into a ``/model`` command.

    Rejects anything that could break out of the slash command (spaces, ``;``,
    carriage returns/newlines, control chars) or is absurdly long. Accepts the
    gateway/UC ids Claude Code's picker shows, e.g. ``system.ai.claude-opus-4-8[1m]``.
    """
    return (
        isinstance(name, str)
        and 0 < len(name) <= MAX_MODEL_NAME_LEN
        and bool(_MODEL_NAME_RE.match(name))
    )


class ConfirmationState:
    """Watch PTY output for Claude's "Switch model?" dialog and auto-confirm it.

    ``arm`` after typing ``/model`` (with a deadline), feed each output chunk to
    ``observe``; when the confirmation prompt is seen it returns the keystrokes to
    send (``b"\\r"``) and disarms. A rolling buffer keeps the last ``window`` chars so
    a marker split across two PTY reads (or interrupted by an ANSI escape) still matches.
    """

    PROMPT_MARKERS = ("Switch model?", "Yes, switch to")

    def __init__(self, window: int = 4096) -> None:
        self._buf = ""
        self._armed_until = 0.0
        self._window = window

    def arm(self, deadline: float) -> None:
        self._armed_until = deadline
        self._buf = ""

    def clear(self) -> None:
        self._armed_until = 0.0
        self._buf = ""

    def observe(self, chunk: bytes, now: float) -> bytes | None:
        if self._armed_until == 0.0:
            return None
        if now > self._armed_until:
            self.clear()
            return None
        self._buf = (self._buf + _match_text(chunk))[-self._window :]
        if any(_squash(marker) in self._buf for marker in self.PROMPT_MARKERS):
            self.clear()
            return b"\r"
        return None


class OutputMarkerDetector:
    """Latching substring detector over an ANSI-stripped rolling buffer.

    Returns ``True`` from ``observe`` once any configured marker has been seen, and
    stays ``True`` thereafter (``triggered``). Used for TUI-readiness and turn-start
    detection where there is no JSON-RPC frame to key off.
    """

    def __init__(self, markers: tuple[str, ...], window: int = 4096) -> None:
        self._markers = markers
        self._buf = ""
        self._window = window
        self.triggered = False

    def observe(self, chunk: bytes) -> bool:
        if self.triggered:
            return True
        self._buf = (self._buf + _match_text(chunk))[-self._window :]
        if any(_squash(marker) in self._buf for marker in self._markers):
            self.triggered = True
        return self.triggered


def inject_model_switch(master_fd: int, model: str, confirm: ConfirmationState, now: float) -> None:
    """Type ``/model <model>\\r`` into the TUI and arm the confirm watcher."""
    os.write(master_fd, f"/model {model}\r".encode())
    confirm.arm(now + CONFIRM_TIMEOUT_S)


def inject_note(out_fd: int, message: str) -> None:
    """Splice a wrapper-authored, cyan note into the output stream (see module docstring §5).

    The wrapper owns stdout, so it can write its own bytes; timing this at the readiness
    boundary (before the first prompt) lands it in static scroll-back above the input box.
    """
    os.write(out_fd, ("\r\n\x1b[36m" + message + "\x1b[0m\r\n").encode())


# --- JSON-RPC control channel ---------------------------------------------------------

_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602


def _rpc_error(rid: object, code: int, message: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})


def handle_jsonrpc_line(line: str, on_model_set: Callable[[str], None]) -> str | None:
    """Handle one JSON-RPC request line.

    Returns the response JSON string, or ``None`` for a notification (no ``id``).
    Dispatches ``model.set`` -> ``on_model_set(name)`` after validating the name.
    """
    if not line.strip():
        return None
    try:
        request = json.loads(line)
    except ValueError:
        return _rpc_error(None, _PARSE_ERROR, "Parse error")
    if not isinstance(request, dict):
        return _rpc_error(None, _INVALID_REQUEST, "Invalid Request")

    rid = request.get("id")
    is_notification = "id" not in request

    def reply(response: str) -> str | None:
        return None if is_notification else response

    if request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
        return reply(_rpc_error(rid, _INVALID_REQUEST, "Invalid Request"))
    if request.get("method") != "model.set":
        return reply(_rpc_error(rid, _METHOD_NOT_FOUND, "Method not found"))

    params = request.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if not isinstance(name, str) or not valid_model_name(name):
        return reply(_rpc_error(rid, _INVALID_PARAMS, "Invalid params: model name"))

    on_model_set(name)  # name narrowed to str by the isinstance guard above
    if is_notification:
        return None
    return json.dumps({"jsonrpc": "2.0", "id": rid, "result": {"model": name, "injected": True}})


def serve_control_socket(
    path: Path,
    on_model_set: Callable[[str], None],
    stop: threading.Event,
    *,
    log: Callable[[str], None] = lambda _m: None,
) -> threading.Thread:
    """Start a daemon AF_UNIX server thread that dispatches line-delimited JSON-RPC.

    Unlinks a stale socket file, binds with owner-only perms, and accepts connections
    until ``stop`` is set. Returns the started thread.
    """

    def serve() -> None:
        try:
            if path.exists():
                path.unlink()
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(str(path))
            os.chmod(path, 0o600)
            srv.listen(4)
            srv.settimeout(0.5)
        except OSError as exc:
            log(f"[ERR] control socket bind failed: {exc!r}")
            return
        log(f"[READY] control socket {path}")
        try:
            while not stop.is_set():
                try:
                    conn, _ = srv.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                with conn, conn.makefile("rwb") as stream:
                    for raw in stream:
                        try:
                            response = handle_jsonrpc_line(
                                raw.decode("utf-8", "replace"), on_model_set
                            )
                        except Exception as exc:  # noqa: BLE001 - one bad line must not kill the server
                            log(f"[ERR] control line: {exc!r}")
                            continue
                        if response is not None:
                            stream.write((response + "\n").encode())
                            stream.flush()
        finally:
            srv.close()

    thread = threading.Thread(target=serve, name="claude-pty-control", daemon=True)
    thread.start()
    return thread


# --- terminal / window-size plumbing --------------------------------------------------


class TerminalModeGuard:
    """Put stdin in raw mode for the PTY session; always restore on exit.

    A no-op when stdin is not a TTY (piped input under tests/CI). Ports the Rust POC's
    ``TerminalModeGuard``.
    """

    def __init__(self, fd: int = 0) -> None:
        self.fd = fd
        self._saved: list | None = None

    def __enter__(self) -> TerminalModeGuard:
        if os.isatty(self.fd):
            self._saved = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None


def sync_winsize(master_fd: int, stdin_fd: int = 0) -> None:
    """Propagate the controlling terminal's window size onto the PTY master."""
    if not os.isatty(stdin_fd):
        return
    try:
        packed = fcntl.ioctl(stdin_fd, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, packed)
    except OSError:
        pass


# --- orchestrator ---------------------------------------------------------------------


def run_claude_pty(
    argv: list[str],
    *,
    target_model: str,
    switch_message: str,
    socket_path: Path,
    log_path: Path | None = None,
    switch_mode: str = "preemptive",
) -> int:
    """Spawn *argv* in a PTY and run the smart-router CUJ, returning the child exit code.

    Pumps stdin<->master and master<->stdout; runs the JSON-RPC control socket; and
    auto-switches to ``target_model`` — in ``"reactive"`` mode on the first prompt the
    user submits (detected from the Enter keystroke on stdin), or in ``"preemptive"``
    mode once the TUI paints and goes idle — surfacing ``switch_message`` in the transcript.
    """

    def log(message: str) -> None:
        if log_path is None:
            return
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")
        except OSError:
            pass

    # UCODE_CLAUDE_PTY_DEBUG=1 dumps each ANSI-stripped output chunk to the log so the
    # readiness / confirm-dialog / turn markers can be grepped from a real launch.
    debug = os.environ.get("UCODE_CLAUDE_PTY_DEBUG") == "1"

    pid, master_fd = pty.fork()
    if pid == 0:  # child: become claude
        os.execvp(argv[0], argv)
        os._exit(127)  # unreachable if execvp succeeds

    confirm = ConfirmationState()
    ready = OutputMarkerDetector(READY_MARKERS)
    lock = threading.Lock()
    switched = {"done": False}
    stop = threading.Event()

    def switch_to(model: str) -> None:
        with lock:
            if switched["done"]:
                return
            switched["done"] = True
            inject_note(1, switch_message)
            inject_model_switch(master_fd, model, confirm, time.monotonic())
            log(f"[SWITCH] -> {model!r}")

    serve_control_socket(socket_path, switch_to, stop, log=log)

    def on_winch(_signum: int, _frame: object) -> None:
        sync_winsize(master_fd)

    try:
        with TerminalModeGuard(0):
            signal.signal(signal.SIGWINCH, on_winch)
            sync_winsize(master_fd)
            stdin_open = True
            last_output = 0.0  # monotonic of the most recent TUI output (0 = none yet)
            typed_content = False  # printable input seen since the last Enter (reactive mode)
            pending_switch = False  # first prompt submitted; switch when Claude next goes idle
            while True:
                readable = [master_fd, 0] if stdin_open else [master_fd]
                # Both modes fire on an idle gap, so they need periodic wakes while waiting:
                # preemptive from the start, reactive only after the first prompt is submitted.
                need_idle_wake = not switched["done"] and (
                    switch_mode == "preemptive"
                    or (switch_mode == "reactive" and pending_switch)
                )
                timeout = SELECT_TIMEOUT_S if need_idle_wake else None
                try:
                    ready_fds, _, _ = select.select(readable, [], [], timeout)
                except InterruptedError:  # SIGWINCH etc.
                    continue

                if 0 in ready_fds:
                    try:
                        data = os.read(0, 4096)
                    except OSError:
                        data = b""
                    if not data:
                        stdin_open = False  # EOF on stdin: stop selecting it, keep pumping
                    else:
                        os.write(master_fd, data)
                        # Reactive: the user submits their first prompt when they press Enter
                        # after typing. This is a stdin keystroke (fully under our control) —
                        # far more reliable than scraping Claude's rendered output. Arm here;
                        # the actual /model keystroke fires once Claude goes idle (below), so it
                        # lands in an idle input box rather than queued mid-response.
                        if switch_mode == "reactive" and not switched["done"] and not pending_switch:
                            if any(byte >= 0x20 and byte != 0x7F for byte in data):
                                typed_content = True
                            if typed_content and (b"\r" in data or b"\n" in data):
                                pending_switch = True

                if master_fd in ready_fds:
                    try:
                        chunk = os.read(master_fd, 8192)
                    except OSError:  # child exited -> EIO on Linux
                        chunk = b""
                    if not chunk:
                        break
                    os.write(1, chunk)
                    last_output = time.monotonic()
                    if debug:
                        log(f"[OUT] {strip_ansi(chunk)[:400]!r}")
                    with lock:
                        keystroke = confirm.observe(chunk, last_output)
                    if keystroke is not None:
                        os.write(master_fd, keystroke)
                    ready.observe(chunk)

                # Fire on an idle gap: preemptive before any prompt (TUI painted + idle);
                # reactive only after the first prompt was submitted (turn done + idle), so the
                # /model command types into an idle input box.
                if not switched["done"]:
                    now = time.monotonic()
                    idle = last_output > 0.0 and (now - last_output) >= READY_QUIET_S
                    if switch_mode == "preemptive" and (ready.triggered or idle):
                        switch_to(target_model)
                    elif switch_mode == "reactive" and pending_switch and idle:
                        switch_to(target_model)
    finally:
        stop.set()
        with contextlib.suppress(OSError):
            os.close(master_fd)
        if socket_path.exists():
            with contextlib.suppress(OSError):
                socket_path.unlink()

    _pid, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1
