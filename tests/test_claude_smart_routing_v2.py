"""Tests for the experimental ENABLE_SMART_ROUTING_V2 Claude Code (PTY wrapper) launch path."""

from __future__ import annotations

import fcntl
import json
import os
import pty
import socket
import struct
import termios
import threading
import time

import pytest

from ucode.agents import claude
from ucode.smart_routing import claude_pty


class TestStripAnsi:
    def test_strips_csi_and_leaves_text(self):
        assert claude_pty.strip_ansi(b"\x1b[1mSwitch\x1b[0m model?") == "Switch model?"

    def test_partial_sequence_is_lenient(self):
        # A dangling ESC without a full sequence should not raise or eat later text.
        assert "hello" in claude_pty.strip_ansi(b"hello\x1b")


class TestValidModelName:
    @pytest.mark.parametrize(
        "name",
        [
            "system.ai.claude-opus-4-8[1m]",
            "databricks-claude-sonnet-4-6",
            "opus",
            "claude-3-5-haiku",
        ],
    )
    def test_accepts_gateway_ids(self, name):
        assert claude_pty.valid_model_name(name)

    @pytest.mark.parametrize(
        "name",
        ["", "a b", "a;b", "a\nb", "a\rb", "x" * 201, 123, None, "cmd`whoami`"],
    )
    def test_rejects_injection_and_junk(self, name):
        assert not claude_pty.valid_model_name(name)


class TestConfirmationState:
    def test_full_prompt_confirms_and_disarms(self):
        state = claude_pty.ConfirmationState()
        now = time.monotonic()
        state.arm(now + 5)
        assert state.observe(b"Switch model? Yes, switch to Opus / No, go back", now) == b"\r"
        # Disarmed after firing: a second identical chunk is ignored.
        assert state.observe(b"Switch model?", now) is None

    def test_marker_split_across_chunks(self):
        state = claude_pty.ConfirmationState()
        now = time.monotonic()
        state.arm(now + 5)
        assert state.observe(b"\x1b[1mSwitch mo", now) is None
        assert state.observe(b"del?\x1b[0m Yes, switch to Opus", now) == b"\r"

    def test_past_deadline_clears(self):
        state = claude_pty.ConfirmationState()
        state.arm(time.monotonic() - 1)  # already expired
        assert state.observe(b"Switch model? Yes, switch to X", time.monotonic()) is None

    def test_not_armed_returns_none(self):
        state = claude_pty.ConfirmationState()
        assert state.observe(b"Switch model? Yes, switch to X", time.monotonic()) is None


class TestOutputMarkerDetector:
    def test_latches_after_marker(self):
        det = claude_pty.OutputMarkerDetector(("? for shortcuts",))
        assert det.observe(b"booting up") is False
        assert det.observe(b"ready: ? for shortcuts") is True
        # Latched: stays True even on unrelated later output.
        assert det.observe(b"nothing here") is True

    def test_marker_split_across_chunks(self):
        det = claude_pty.OutputMarkerDetector(("esc to interrupt",))
        assert det.observe(b"\x1b[2mesc to ") is False
        assert det.observe(b"interr") is False
        assert det.observe(b"upt\x1b[0m") is True


class TestHandleJsonRpc:
    def test_valid_model_set(self):
        seen: list[str] = []
        resp = claude_pty.handle_jsonrpc_line(
            json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "model.set", "params": {"name": "opus"}}
            ),
            seen.append,
        )
        assert seen == ["opus"]
        assert json.loads(resp)["result"]["model"] == "opus"

    def test_notification_has_no_response_but_dispatches(self):
        seen: list[str] = []
        resp = claude_pty.handle_jsonrpc_line(
            json.dumps({"jsonrpc": "2.0", "method": "model.set", "params": {"name": "opus"}}),
            seen.append,
        )
        assert resp is None
        assert seen == ["opus"]

    def test_unknown_method(self):
        resp = claude_pty.handle_jsonrpc_line(
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "nope"}), lambda _m: None
        )
        assert json.loads(resp)["error"]["code"] == -32601

    def test_missing_model_is_invalid_params(self):
        resp = claude_pty.handle_jsonrpc_line(
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "model.set", "params": {}}),
            lambda _m: pytest.fail("must not dispatch"),
        )
        assert json.loads(resp)["error"]["code"] == -32602

    def test_injection_model_name_rejected(self):
        resp = claude_pty.handle_jsonrpc_line(
            json.dumps(
                {"jsonrpc": "2.0", "id": 4, "method": "model.set", "params": {"name": "x\r/help"}}
            ),
            lambda _m: pytest.fail("must not dispatch"),
        )
        assert json.loads(resp)["error"]["code"] == -32602

    def test_bad_json_is_parse_error(self):
        resp = claude_pty.handle_jsonrpc_line("{not json", lambda _m: None)
        assert json.loads(resp)["error"]["code"] == -32700

    def test_non_object_is_invalid_request(self):
        resp = claude_pty.handle_jsonrpc_line("123", lambda _m: None)
        assert json.loads(resp)["error"]["code"] == -32600


class TestInjectors:
    def test_inject_model_switch_types_command_and_arms(self):
        read_fd, write_fd = os.pipe()
        try:
            confirm = claude_pty.ConfirmationState()
            claude_pty.inject_model_switch(write_fd, "opus", confirm, time.monotonic())
            assert os.read(read_fd, 100) == b"/model opus\r"
            # Armed: it now auto-confirms the dialog.
            assert confirm.observe(b"Switch model? Yes, switch to X", time.monotonic()) == b"\r"
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_inject_note_writes_message(self):
        read_fd, write_fd = os.pipe()
        try:
            claude_pty.inject_note(write_fd, "router picked opus")
            out = os.read(read_fd, 200).decode()
            assert "router picked opus" in out
        finally:
            os.close(read_fd)
            os.close(write_fd)


class TestTerminalModeGuard:
    def test_noop_when_not_a_tty(self):
        read_fd, write_fd = os.pipe()
        try:
            # A pipe fd is not a TTY: entering/exiting must not touch termios or raise.
            with claude_pty.TerminalModeGuard(read_fd) as guard:
                assert guard._saved is None
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_restores_tty_attrs(self):
        master_fd, slave_fd = pty.openpty()
        try:
            before = termios.tcgetattr(slave_fd)
            with claude_pty.TerminalModeGuard(slave_fd):
                pass
            assert termios.tcgetattr(slave_fd) == before
        finally:
            os.close(master_fd)
            os.close(slave_fd)


class TestSyncWinsize:
    def test_propagates_window_size(self):
        stdin_master, stdin_slave = pty.openpty()
        out_master, out_slave = pty.openpty()
        try:
            want = struct.pack("HHHH", 40, 120, 0, 0)  # rows, cols
            fcntl.ioctl(stdin_slave, termios.TIOCSWINSZ, want)
            claude_pty.sync_winsize(out_master, stdin_slave)
            got = fcntl.ioctl(out_slave, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
            assert struct.unpack("HHHH", got)[:2] == (40, 120)
        finally:
            for fd in (stdin_master, stdin_slave, out_master, out_slave):
                os.close(fd)


class TestServeControlSocket:
    def test_dispatches_model_set_over_socket(self, tmp_path):
        sock_path = tmp_path / "ctl.sock"
        seen: list[str] = []
        stop = threading.Event()
        claude_pty.serve_control_socket(sock_path, seen.append, stop)
        try:
            deadline = time.monotonic() + 5
            while not sock_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(str(sock_path))
            request = (
                json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "method": "model.set", "params": {"name": "opus"}}
                )
                + "\n"
            )
            client.sendall(request.encode())
            response = client.makefile("rb").readline()
            client.close()
            assert seen == ["opus"]
            assert json.loads(response)["result"]["model"] == "opus"
        finally:
            stop.set()


class TestV2Router:
    def test_stub_routes_everything_to_fixed_model(self):
        route = claude._v2_router({})
        assert route({"model": "opus", "messages": []}) == claude.SMART_ROUTING_V2_MODEL
        assert claude.SMART_ROUTING_V2_MODEL == "system.ai.claude-sonnet-5"


class TestLaunchGate:
    def test_v2_gate_routes_to_pty_when_enabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SMART_ROUTING_V2", "1")
        calls: dict[str, bool] = {}
        monkeypatch.setattr(
            claude, "_launch_smart_routing_v2", lambda *_a: calls.__setitem__("v2", True)
        )
        monkeypatch.setattr(claude, "exec_or_spawn", lambda *_a: calls.__setitem__("exec", True))
        claude.launch({"workspace": "https://example.databricks.com"}, [])
        assert calls == {"v2": True}

    def test_normal_launch_when_disabled(self, monkeypatch):
        monkeypatch.delenv("ENABLE_SMART_ROUTING_V2", raising=False)
        calls: dict[str, bool] = {}
        monkeypatch.setattr(
            claude, "_launch_smart_routing_v2", lambda *_a: calls.__setitem__("v2", True)
        )
        monkeypatch.setattr(claude, "get_databricks_token", lambda *_a, **_k: "tok")
        monkeypatch.setattr(claude, "exec_or_spawn", lambda *_a: calls.__setitem__("exec", True))
        claude.launch({"workspace": "https://example.databricks.com"}, [])
        assert calls == {"exec": True}
