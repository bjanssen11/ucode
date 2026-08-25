"""Tests for the experimental ENABLE_SMART_ROUTING_V2 Claude Code (PTY wrapper) launch path."""

from __future__ import annotations

import fcntl
import json
import os
import pty
import socket
import struct
import sys
import termios
import threading
import time
from pathlib import Path

import pytest

from ucode.agents import claude
from ucode.smart_routing import claude_hooks, claude_pty


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


class TestModelPickerLabels:
    @pytest.mark.parametrize(
        ("model", "friendly"),
        [
            ("system.ai.claude-sonnet-5", "Sonnet 5"),
            ("system.ai.claude-haiku-4-5", "Haiku 4.5"),
            ("system.ai.claude-opus-4-8[1m]", "Opus 4.8 (1M)"),
        ],
    )
    def test_derives_current_claude_code_picker_label(self, model, friendly):
        assert claude_pty.model_picker_labels(model) == (model, friendly)

    def test_keeps_unknown_model_as_raw_label(self):
        assert claude_pty.model_picker_labels("custom-model") == ("custom-model",)


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
        assert state.observe(b"del?\x1b[0m Yes, switch to Opus", now) is None
        assert state.observe(b" / No, go back", now) == b"\r"

    def test_waits_for_complete_dialog(self):
        state = claude_pty.ConfirmationState()
        now = time.monotonic()
        state.arm(now + 5)
        assert state.observe(b"Switch model? Yes, switch to Opus", now) is None
        assert state.observe(b"No, go back", now) == b"\r"

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
    def test_model_picker_rows_navigates_to_target(self):
        rows = claude_pty.ModelPickerRows("system.ai.claude-sonnet-5")
        rows.observe(b"  3. system.ai.claude-sonnet-5  Custom Sonnet model\r")
        rows.observe("\x1b[1m  ❯ 4. system.ai.claude-haiku-4-5\x1b[0m".encode())
        assert rows.target_row == 3
        assert rows.focused_row == 4
        assert rows.navigation == b"\x1b[A"

    def test_model_picker_rows_can_move_down_or_stay(self):
        rows = claude_pty.ModelPickerRows("target")
        rows.observe("❯ 2. current\r  4. target".encode())
        assert rows.navigation == b"\x1b[B\x1b[B"
        rows.observe("❯ 4. target".encode())
        assert rows.navigation == b""

    def test_model_picker_rows_accepts_friendly_label(self):
        rows = claude_pty.ModelPickerRows("system.ai.claude-sonnet-5")
        rows.observe("  2. Haiku 4.5\r❯ 3. Sonnet 5".encode())
        assert rows.target_row == 3
        assert rows.focused_row == 3
        assert rows.navigation == b""

    def test_inject_model_switch_opens_picker(self):
        read_fd, write_fd = os.pipe()
        try:
            claude_pty.inject_model_switch(write_fd)
            assert os.read(read_fd, 100) == b"/model\r"
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

    def test_replays_multiline_prompt_as_one_bracketed_paste(self):
        read_fd, write_fd = os.pipe()
        try:
            claude_pty.inject_prompt(write_fd, "first\nsecond")
            assert os.read(read_fd, 200) == b"\x1b[200~first\nsecond\x1b[201~\r"
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_can_restore_prompt_without_submitting(self):
        read_fd, write_fd = os.pipe()
        try:
            claude_pty.inject_prompt(write_fd, "try again", submit=False)
            assert os.read(read_fd, 200) == b"\x1b[200~try again\x1b[201~"
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


class TestFirstPromptHook:
    def test_blocks_once_then_allows_replay(self, tmp_path):
        sock_path = tmp_path / "first.sock"
        blocked: list[tuple[str, str]] = []
        stop = threading.Event()
        claude_pty.serve_first_prompt_socket(
            sock_path,
            lambda prompt: "sonnet" if prompt else "opus",
            lambda prompt, model: blocked.append((prompt, model)),
            stop,
        )
        try:
            deadline = time.monotonic() + 5
            while not sock_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            first = claude_pty.request_first_prompt_route(
                sock_path, {"session_id": "s1", "prompt": "fix the parser"}
            )
            replay = claude_pty.request_first_prompt_route(
                sock_path, {"session_id": "s1", "prompt": "fix the parser"}
            )
            assert first == {"action": "block", "model": "sonnet"}
            assert replay == {"action": "allow"}
            assert blocked == [("fix the parser", "sonnet")]
        finally:
            stop.set()

    def test_slash_command_does_not_claim_first_prompt(self, tmp_path):
        sock_path = tmp_path / "first.sock"
        stop = threading.Event()
        claude_pty.serve_first_prompt_socket(
            sock_path, lambda _prompt: "opus", lambda *_args: None, stop
        )
        try:
            deadline = time.monotonic() + 5
            while not sock_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            command = claude_pty.request_first_prompt_route(sock_path, {"prompt": "/hooks"})
            prompt = claude_pty.request_first_prompt_route(sock_path, {"prompt": "do work"})
            assert command == {"action": "allow"}
            assert prompt == {"action": "block", "model": "opus"}
        finally:
            stop.set()

    def test_hook_output_blocks_with_actionable_message(self):
        output = claude_pty.first_prompt_hook_output(
            {"action": "block", "model": "system.ai.claude-sonnet-5"}
        )
        assert output["decision"] == "block"
        assert output["reason"] == (
            "✨ Smart Router selected system.ai.claude-sonnet-5 due to low complexity, "
            "unclear intent, and no code reference."
        )

    def test_hook_failure_allows_prompt(self, tmp_path):
        assert (
            claude_pty.request_first_prompt_route(
                tmp_path / "missing.sock", {"prompt": "do work"}, timeout=0.01
            )
            is None
        )
        assert claude_pty.first_prompt_hook_output(None) is None


class TestFirstPromptHookSettings:
    def test_adds_stable_hook_without_removing_other_routing_hooks(self):
        doc = {
            "hooks": {
                "PreToolUse": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "ucode claude-router-hook route-subagent",
                            }
                        ]
                    }
                ]
            }
        }
        claude_hooks.sync_first_prompt_hook(doc, "/bin/ucode")
        claude_hooks.sync_first_prompt_hook(doc, "/bin/ucode")
        command = doc["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
        assert command == "/bin/ucode claude-router-hook route-first-prompt"
        assert len(doc["hooks"]["UserPromptSubmit"]) == 1
        assert "route-subagent" in str(doc["hooks"]["PreToolUse"])


class TestV2Launch:
    def test_uses_unique_hook_settings_and_cleans_them_up(self, tmp_path, monkeypatch):
        settings_path = tmp_path / "ucode-settings.json"
        settings_path.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://gateway"}}))
        monkeypatch.setattr(claude, "APP_DIR", tmp_path)
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", settings_path)
        monkeypatch.setattr(claude, "SMART_ROUTING_V2_CLAUDE_LOG", tmp_path / "v2.log")
        monkeypatch.setattr(claude, "get_databricks_token", lambda *_a, **_k: "token")
        monkeypatch.setattr(claude, "build_auth_token_argv", lambda *_a, **_k: ["/bin/ucode"])
        captured: dict = {}

        def fake_run(argv, **kwargs):
            generated = argv[argv.index("--settings") + 1]
            captured["path"] = generated
            captured["settings"] = json.loads(Path(generated).read_text())
            captured["kwargs"] = kwargs
            return 0

        monkeypatch.setattr(claude_pty, "run_claude_pty", fake_run)
        with pytest.raises(SystemExit) as exc:
            claude._launch_smart_routing_v2({"workspace": "https://example.com"}, ["--debug"])

        assert exc.value.code == 0
        assert not Path(captured["path"]).exists()
        env = captured["settings"]["env"]
        assert env["ANTHROPIC_BASE_URL"] == "https://gateway"
        assert env[claude_hooks.FIRST_PROMPT_SOCKET_ENV].endswith(".sock")
        hook = captured["settings"]["hooks"]["UserPromptSubmit"][0]["hooks"][0]
        assert hook["command"] == "/bin/ucode claude-router-hook route-first-prompt"
        assert captured["kwargs"]["route_prompt"]("anything") == claude.SMART_ROUTING_V2_MODEL


class TestPtyFlow:
    def test_hook_block_switch_confirm_and_replay(self, tmp_path):
        fake_claude = tmp_path / "fake_claude.py"
        capture = tmp_path / "capture.bin"
        socket_path = tmp_path / "first.sock"
        fake_claude.write_text(
            """
import json
import os
import socket
import sys
import tty
from pathlib import Path

socket_path = sys.argv[1]
capture_path = Path(sys.argv[2])

client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(socket_path)
client.sendall((json.dumps({
    "method": "route_first_prompt",
    "prompt": "fix\\nthe parser",
    "session_id": "s1",
}) + "\\n").encode())
response = client.makefile("rb").readline()
client.close()
assert json.loads(response) == {
    "action": "block",
    "model": "system.ai.claude-sonnet-5",
}
print("Smart Routing blocked the prompt", flush=True)

tty.setraw(0)

def read_until(suffix):
    data = b""
    while not data.endswith(suffix):
        data += os.read(0, 1)
    return data

def read_exact(size):
    data = b""
    while len(data) < size:
        data += os.read(0, size - len(data))
    return data

model_command = read_until(b"\\r")
print(
    "Select model\\n"
    "  3. system.ai.claude-sonnet-5  Custom Sonnet model\\n"
    "❯ 4. system.ai.claude-haiku-4-5  Custom Haiku model\\n"
    "Enter to set as default  s use this session only",
    flush=True,
)
navigation = read_exact(3)
print("model picker focus moved", flush=True)
choice = read_exact(1)
print("Switch model? Yes, switch to Sonnet / No, go back", flush=True)
confirmation = read_exact(1)
print("Set model to Sonnet for this session only", flush=True)
replayed = read_until(b"\\x1b[201~\\r")
capture_path.write_bytes(
    model_command + b"|" + navigation + b"|" + choice + b"|" + confirmation
    + b"|" + replayed
)
""".lstrip()
        )

        result = claude_pty.run_claude_pty(
            [sys.executable, str(fake_claude), str(socket_path), str(capture)],
            route_prompt=lambda _prompt: "system.ai.claude-sonnet-5",
            switch_message="router selected sonnet",
            socket_path=socket_path,
        )

        assert result == 0
        assert capture.read_bytes() == (
            b"/model\r|\x1b[A|s|\r|\x1b[200~fix\nthe parser\x1b[201~\r"
        )


class TestV2Router:
    def test_stub_routes_everything_to_fixed_model(self):
        route = claude._v2_router({})
        assert route("fix the parser") == claude.SMART_ROUTING_V2_MODEL
        assert claude.SMART_ROUTING_V2_MODEL == "system.ai.claude-sonnet-4-6[1m]"


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
