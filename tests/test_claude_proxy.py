"""Tests for the smart-routing-v2 model-routing proxy's request rewriting."""

from __future__ import annotations

import json

from ucode.smart_routing import claude_proxy


def _stub(model: str):
    return lambda _body: model


class TestRewriteModel:
    def test_rewrites_model_in_json_body(self):
        body = json.dumps({"model": "opus", "messages": [{"role": "user", "content": "hi"}]}).encode()
        new_body, original, chosen = claude_proxy.rewrite_model(
            body, is_json=True, route=_stub("system.ai.claude-sonnet-5")
        )
        assert original == "opus"
        assert chosen == "system.ai.claude-sonnet-5"
        assert json.loads(new_body)["model"] == "system.ai.claude-sonnet-5"
        # Everything else in the body is preserved.
        assert json.loads(new_body)["messages"] == [{"role": "user", "content": "hi"}]

    def test_unchanged_when_router_picks_same_model(self):
        body = json.dumps({"model": "system.ai.claude-sonnet-5"}).encode()
        new_body, original, chosen = claude_proxy.rewrite_model(
            body, is_json=True, route=_stub("system.ai.claude-sonnet-5")
        )
        assert new_body == body
        assert original == "system.ai.claude-sonnet-5"
        assert chosen == "system.ai.claude-sonnet-5"

    def test_unchanged_when_not_json(self):
        body = b"\x00\x01 not json"
        new_body, original, chosen = claude_proxy.rewrite_model(
            body, is_json=False, route=_stub("x")
        )
        assert new_body == body
        assert original is None and chosen is None

    def test_unchanged_when_body_has_no_model(self):
        body = json.dumps({"messages": []}).encode()
        new_body, original, chosen = claude_proxy.rewrite_model(body, is_json=True, route=_stub("x"))
        assert new_body == body
        assert original is None and chosen is None

    def test_unchanged_on_empty_body(self):
        new_body, original, chosen = claude_proxy.rewrite_model(b"", is_json=True, route=_stub("x"))
        assert new_body == b""
        assert original is None and chosen is None

    def test_malformed_json_passes_through(self):
        body = b'{"model": "opus"'  # truncated
        new_body, original, chosen = claude_proxy.rewrite_model(body, is_json=True, route=_stub("y"))
        assert new_body == body
        assert original is None and chosen is None
