"""PTY wrapper for Claude Code's TUI (smart routing v2).

Claude Code has no ``app-server``/JSON-RPC seam like Codex, so there is nothing to
interpose on the wire. Instead this module runs the real ``claude`` TUI inside a PTY:
it forwards stdin<->master and master<->stdout untouched, and drives a *model switch*
through Claude Code's ``/model`` picker, using its ``s`` (session-only) action, and
auto-confirming the optional "Switch model?" cache dialog by watching the PTY output.

``ucode.agents.claude`` owns the lifecycle: it enters this from the single ``ucode claude``
command when ``ENABLE_SMART_ROUTING_V2=1``. Logs go to ``log_path`` (appended) only — never
stdout/stderr, which the foreground TUI owns (same discipline as ``codex_interposer``).

The first prompt comes from a ``UserPromptSubmit`` hook over an owner-only Unix socket.
The hook blocks that one submission, allowing the wrapper to type ``/model`` while the
TUI is idle and then replay the exact prompt.  A second hook invocation (the replay) is
allowed through.  This makes the selected model the real Claude Code session model before
the first inference request, rather than merely rewriting the request below the client.
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

# Output markers are whitespace-squashed because Claude Code renders spaces as cursor
# moves that vanish under strip_ansi. READY_MARKERS remains useful to runtime probes/tests;
# the first-prompt path itself waits for the hook-blocked TUI to go idle.
READY_MARKERS = ("? for shortcuts", "Welcome to Claude Code")

MAX_MODEL_NAME_LEN = 200
CONFIRM_TIMEOUT_S = 3.0
SWITCH_TIMEOUT_S = 6.0
# Once the TUI stays quiet this long after rendering the hook block/model result, it is
# idle and safe for the wrapper to type the next command.
READY_QUIET_S = 0.75
# select() wake interval while waiting to switch, so the quiet period is observable.
SELECT_TIMEOUT_S = 0.2
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:/\-\[\]]+$")
_CLAUDE_MODEL_RE = re.compile(
    r"^(?:system\.ai\.)?claude-(opus|sonnet|haiku)-(\d+)(?:-(\d+))?(\[1m\])?$",
    re.IGNORECASE,
)
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


def model_picker_labels(model: str) -> tuple[str, ...]:
    """Return raw and friendly labels Claude Code may render for a model.

    Claude Code 2.1.243 changed gateway-backed picker rows from raw endpoint IDs such
    as ``system.ai.claude-sonnet-5`` to friendly labels such as ``Sonnet 5``. Keep the
    raw ID first for older versions and add the derived label for newer versions.
    """
    labels = [model]
    match = _CLAUDE_MODEL_RE.fullmatch(model)
    if match is not None:
        family, major, minor, long_context = match.groups()
        version = major if minor is None else f"{major}.{minor}"
        friendly = f"{family.title()} {version}"
        if long_context:
            friendly += " (1M)"
        labels.append(friendly)
    return tuple(labels)


class ConfirmationState:
    """Watch PTY output for Claude's "Switch model?" dialog and auto-confirm it.

    ``arm`` after choosing the model with ``s`` (with a deadline), feed each output chunk to
    ``observe``; when the complete confirmation prompt is seen it returns Enter and
    disarms. The session-only choice has already been made with ``s`` in the model
    picker; this dialog only acknowledges the cache cost of changing models. A rolling
    buffer keeps the last ``window`` chars so
    a marker split across two PTY reads (or interrupted by an ANSI escape) still matches.
    """

    # Waiting for the final option is important: reacting to the title alone can send
    # Enter while Ink is still mounting the dialog, which leaks into the preceding UI.
    PROMPT_MARKERS = ("Switch model?", "Yes, switch to", "No, go back")

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
        if all(_squash(marker) in self._buf for marker in self.PROMPT_MARKERS):
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


class ModelPickerRows:
    """Discover the routed and currently focused row numbers from picker output."""

    def __init__(self, model: str, window: int = 16384) -> None:
        self._targets = tuple(_squash(label) for label in model_picker_labels(model))
        self._buf = ""
        self._window = window
        self.target_row: int | None = None
        self.focused_row: int | None = None

    def observe(self, chunk: bytes) -> None:
        self._buf = (self._buf + _match_text(chunk))[-self._window :]
        targets = [
            match
            for target in self._targets
            for match in re.finditer(rf"(\d+)\.{re.escape(target)}", self._buf)
        ]
        focused = list(re.finditer(r"❯(\d+)\.", self._buf))
        if targets:
            self.target_row = int(targets[-1].group(1))
        if focused:
            self.focused_row = int(focused[-1].group(1))

    @property
    def navigation(self) -> bytes | None:
        """Arrow keys required to move from the focused row to the routed row."""
        if self.target_row is None or self.focused_row is None:
            return None
        delta = self.target_row - self.focused_row
        key = b"\x1b[B" if delta > 0 else b"\x1b[A"
        return key * abs(delta)


def inject_model_switch(master_fd: int) -> None:
    """Open Claude Code's model picker.

    Passing the model directly (``/model <name>``) always persists it in interactive
    Claude Code. Only the full picker exposes the ``s`` session-only action.
    """
    os.write(master_fd, b"/model\r")


def inject_prompt(master_fd: int, prompt: str, *, submit: bool = True) -> None:
    """Replay a hook-captured prompt using terminal bracketed-paste mode.

    Bracketed paste preserves multiline input and prevents embedded newlines from
    submitting partial prompts. Escape/NUL bytes cannot be meaningful prompt text here
    and are removed so captured content cannot terminate the paste envelope.
    """
    clean = prompt.replace("\r\n", "\n").replace("\r", "\n")
    clean = clean.replace("\x00", "").replace("\x1b", "")
    suffix = b"\r" if submit else b""
    os.write(master_fd, b"\x1b[200~" + clean.encode() + b"\x1b[201~" + suffix)


def inject_note(out_fd: int, message: str) -> None:
    """Splice a wrapper-authored, cyan note into the output stream (see module docstring §5).

    The wrapper owns stdout, so it can write its own bytes; timing this at the readiness
    boundary (before the first prompt) lands it in static scroll-back above the input box.
    """
    os.write(out_fd, ("\r\n\x1b[36m" + message + "\x1b[0m\r\n").encode())


# --- first-prompt hook channel --------------------------------------------------------


def request_first_prompt_route(path: Path, payload: dict, *, timeout: float = 5.0) -> dict | None:
    """Send a ``UserPromptSubmit`` payload to the PTY wrapper.

    Hook failures deliberately fail open: returning ``None`` makes the hook emit no
    blocking decision, so Claude processes the user's prompt on its existing model.
    """
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    request = {
        "method": "route_first_prompt",
        "prompt": prompt,
        "session_id": payload.get("session_id"),
    }
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        with client:
            client.connect(str(path))
            client.sendall((json.dumps(request) + "\n").encode())
            with client.makefile("rb") as stream:
                raw = stream.readline()
        response = json.loads(raw) if raw else None
    except (OSError, ValueError):
        return None
    return response if isinstance(response, dict) else None


def first_prompt_hook_output(response: dict | None) -> dict | None:
    """Translate the wrapper response into Claude's UserPromptSubmit hook output."""
    if not isinstance(response, dict) or response.get("action") != "block":
        return None
    model = response.get("model")
    if not valid_model_name(model):
        return None
    return {
        "decision": "block",
        "reason": (
            f"✨ Smart Router selected {model} due to low complexity, unclear intent, "
            "and no code reference."
        ),
    }


def serve_first_prompt_socket(
    path: Path,
    route_prompt: Callable[[str], str],
    on_blocked_prompt: Callable[[str, str], None],
    stop: threading.Event,
    *,
    log: Callable[[str], None] = lambda _m: None,
) -> threading.Thread:
    """Serve the hook protocol, blocking exactly one non-command prompt.

    Slash commands are allowed without claiming the first-prompt slot. After the first
    prompt is blocked, every later request—including the wrapper's replay—is allowed.
    """

    def serve() -> None:
        claimed = False
        try:
            if path.exists():
                path.unlink()
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(str(path))
            os.chmod(path, 0o600)
            srv.listen(4)
            srv.settimeout(0.5)
        except OSError as exc:
            log(f"[ERR] first-prompt socket bind failed: {exc!r}")
            return
        log(f"[READY] first-prompt socket {path}")
        try:
            while not stop.is_set():
                try:
                    conn, _ = srv.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                with conn, conn.makefile("rwb") as stream:
                    raw = stream.readline()
                    response: dict = {"action": "allow"}
                    blocked: tuple[str, str] | None = None
                    try:
                        request = json.loads(raw)
                        prompt = request.get("prompt") if isinstance(request, dict) else None
                        is_route = (
                            isinstance(request, dict)
                            and request.get("method") == "route_first_prompt"
                        )
                        is_command = isinstance(prompt, str) and prompt.lstrip().startswith("/")
                        if (
                            is_route
                            and isinstance(prompt, str)
                            and prompt.strip()
                            and not is_command
                        ):
                            if not claimed:
                                model = route_prompt(prompt)
                                if valid_model_name(model):
                                    claimed = True
                                    response = {"action": "block", "model": model}
                                    blocked = (prompt, model)
                    except Exception as exc:  # noqa: BLE001 - hooks must fail open
                        log(f"[ERR] first-prompt request: {exc!r}")
                    stream.write((json.dumps(response) + "\n").encode())
                    stream.flush()
                    if blocked is not None:
                        on_blocked_prompt(*blocked)
        finally:
            srv.close()

    thread = threading.Thread(target=serve, name="claude-first-prompt", daemon=True)
    thread.start()
    return thread


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
    route_prompt: Callable[[str], str],
    switch_message: str,
    socket_path: Path,
    log_path: Path | None = None,
) -> int:
    """Spawn *argv* in a PTY and run the smart-router CUJ, returning the child exit code.

    A UserPromptSubmit hook sends the first prompt to ``socket_path`` and blocks it.
    Once Claude returns to an idle prompt box, this wrapper types ``/model``, confirms
    the switch, and replays the captured prompt. The replay is allowed by the socket's
    one-shot gate, so the first inference runs on Claude Code's newly selected model.
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

    # Hold the child immediately before exec until the hook socket is listening. Without
    # this small gate, a positional prompt on a very fast launch could invoke the hook
    # before the server thread has bound and fail open on the starting model.
    gate_read, gate_write = os.pipe()
    pid, master_fd = pty.fork()
    if pid == 0:  # child: become claude
        os.close(gate_write)
        try:
            os.read(gate_read, 1)
        finally:
            os.close(gate_read)
        os.execvp(argv[0], argv)
        os._exit(127)  # unreachable if execvp succeeds
    os.close(gate_read)

    confirm = ConfirmationState()
    stop = threading.Event()
    pending_lock = threading.Lock()
    pending: dict[str, tuple[str, str] | None] = {"value": None}

    def on_blocked_prompt(prompt: str, model: str) -> None:
        with pending_lock:
            pending["value"] = (prompt, model)
        log(f"[ROUTE] first prompt -> {model!r}")

    server_thread = serve_first_prompt_socket(
        socket_path, route_prompt, on_blocked_prompt, stop, log=log
    )
    socket_deadline = time.monotonic() + 2.0
    while (
        not socket_path.exists() and server_thread.is_alive() and time.monotonic() < socket_deadline
    ):
        time.sleep(0.01)
    if not socket_path.exists():
        log("[ERR] first-prompt socket was not ready before Claude launch")
    os.write(gate_write, b"1")
    os.close(gate_write)

    def on_winch(_signum: int, _frame: object) -> None:
        sync_winsize(master_fd)

    try:
        with TerminalModeGuard(0):
            signal.signal(signal.SIGWINCH, on_winch)
            sync_winsize(master_fd)
            stdin_open = True
            last_output = 0.0  # monotonic of the most recent TUI output (0 = none yet)
            phase = "waiting_prompt"
            routed_prompt = ""
            routed_model = ""
            switch_started = 0.0
            picker_ready: OutputMarkerDetector | None = None
            picker_rows: ModelPickerRows | None = None
            switch_complete: OutputMarkerDetector | None = None
            switch_step = ""
            navigation_output_seen = False
            while True:
                readable = [master_fd, 0] if stdin_open else [master_fd]
                try:
                    ready_fds, _, _ = select.select(readable, [], [], SELECT_TIMEOUT_S)
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
                    keystroke = confirm.observe(chunk, last_output)
                    if keystroke is not None:
                        os.write(master_fd, keystroke)
                    if phase == "switching":
                        if picker_ready is not None:
                            picker_ready.observe(chunk)
                        if picker_rows is not None:
                            picker_rows.observe(chunk)
                        if switch_step == "navigating":
                            navigation_output_seen = True
                        if switch_complete is not None:
                            switch_complete.observe(chunk)

                if phase == "waiting_prompt":
                    with pending_lock:
                        captured = pending["value"]
                    if captured is not None:
                        routed_prompt, routed_model = captured
                        phase = "waiting_to_switch"

                now = time.monotonic()
                idle = last_output > 0.0 and (now - last_output) >= READY_QUIET_S
                if phase == "waiting_to_switch" and idle:
                    inject_note(1, switch_message)
                    inject_model_switch(master_fd)
                    picker_ready = OutputMarkerDetector(("use this session only",))
                    picker_rows = ModelPickerRows(routed_model)
                    switch_started = now
                    switch_step = "opening_picker"
                    phase = "switching"
                    log(f"[SWITCH] -> {routed_model!r}")
                elif (
                    phase == "switching"
                    and switch_complete is not None
                    and switch_complete.triggered
                ):
                    inject_prompt(master_fd, routed_prompt)
                    phase = "done"
                    log("[REPLAY] first prompt submitted")
                elif (
                    phase == "switching"
                    and switch_step == "opening_picker"
                    and picker_ready is not None
                    and picker_ready.triggered
                    and picker_rows is not None
                    and picker_rows.navigation is not None
                ):
                    navigation = picker_rows.navigation
                    log(
                        f"[PICKER] row {picker_rows.focused_row} -> "
                        f"{picker_rows.target_row}"
                    )
                    if navigation:
                        os.write(master_fd, navigation)
                        navigation_output_seen = False
                        switch_step = "navigating"
                    else:
                        os.write(master_fd, b"s")
                        confirm.arm(now + CONFIRM_TIMEOUT_S)
                        switch_complete = OutputMarkerDetector(("for this session only",))
                        switch_step = "selected"
                elif (
                    phase == "switching"
                    and switch_step == "navigating"
                    and navigation_output_seen
                ):
                    # `s` is Claude Code's model-picker action for this session only.
                    os.write(master_fd, b"s")
                    confirm.arm(now + CONFIRM_TIMEOUT_S)
                    switch_complete = OutputMarkerDetector(("for this session only",))
                    switch_step = "selected"
                elif phase == "switching" and now - switch_started >= SWITCH_TIMEOUT_S:
                    # Do not silently run on the wrong model. Dismiss any open picker and put
                    # the original text back in the editor without submitting it, so the user
                    # can recover manually without losing their prompt.
                    os.write(master_fd, b"\x1b")
                    inject_note(
                        1,
                        "Smart Routing could not confirm the model switch. "
                        "Your prompt was restored but not submitted.",
                    )
                    inject_prompt(master_fd, routed_prompt, submit=False)
                    phase = "failed"
                    reason = (
                        "model picker did not render the selected model"
                        if picker_rows is not None and picker_rows.target_row is None
                        else "model switch confirmation timed out"
                    )
                    log(f"[ERR] {reason} for {routed_model!r}")
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
