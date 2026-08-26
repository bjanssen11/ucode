"""Anthropic model discovery transformations for the gateway proxy."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from ucode import gateway_proxy

_MODEL_ALIAS_PREFIX = "anthropic-aigw-"
_ANTHROPIC_MODELS_PATH = "/v1/models"
_ANTHROPIC_MESSAGES_PATH = "/v1/messages"


class _AnthropicModelAliases:
    """Maps Claude-compatible discovery IDs back to their gateway model IDs."""

    def __init__(self) -> None:
        self._original_by_alias: dict[str, str] = {}
        self._lock = threading.Lock()

    def prefix_model_ids(self, body: bytes) -> bytes:
        try:
            payload = json.loads(body)
            models = payload["data"]
            if not isinstance(models, list):
                return body
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            return body

        aliases: dict[str, str] = {}
        for model in models:
            if not isinstance(model, dict) or not isinstance(model.get("id"), str):
                continue
            model_id = model["id"]
            lowered = model_id.lower()
            if "claude" in lowered or "anthropic" in lowered:
                continue
            alias = f"{_MODEL_ALIAS_PREFIX}{model_id}"
            model["id"] = alias
            aliases[alias] = model_id

        with self._lock:
            self._original_by_alias.update(aliases)

        for cursor in ("first_id", "last_id"):
            model_id = payload.get(cursor)
            alias = f"{_MODEL_ALIAS_PREFIX}{model_id}"
            if alias in aliases:
                payload[cursor] = alias
        return json.dumps(payload, separators=(",", ":")).encode()

    def original_id(self, model_id: str) -> str:
        with self._lock:
            return self._original_by_alias.get(model_id, model_id)

    def rewrite_path(self, path: str) -> str:
        parsed = urlsplit(path)
        if parsed.path != _ANTHROPIC_MODELS_PATH:
            return path
        query = [
            (key, self.original_id(value) if key == "after_id" else value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
        )

    def rewrite_body(self, path: str, body: bytes | None) -> bytes | None:
        if urlsplit(path).path != _ANTHROPIC_MESSAGES_PATH or body is None:
            return body
        try:
            payload = json.loads(body)
            model_id = payload.get("model")
            if not isinstance(model_id, str):
                return body
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            return body
        original_id = self.original_id(model_id)
        if original_id == model_id:
            return body
        payload["model"] = original_id
        return json.dumps(payload, separators=(",", ":")).encode()


class _AnthropicModelDiscoveryHandler(gateway_proxy._ProxyHandler):
    anthropic_model_aliases: _AnthropicModelAliases

    def _transform_request(self, body: bytes | None) -> tuple[str, bytes | None]:
        body = self.anthropic_model_aliases.rewrite_body(self.path, body)
        url = self.anthropic_model_aliases.rewrite_path(self.path).lstrip("/")
        return url, body

    def _transform_response(self, resp: httpx.Response) -> bytes | None:
        should_prefix_model_ids = (
            self.command == "GET"
            and urlsplit(self.path).path == _ANTHROPIC_MODELS_PATH
            and HTTPStatus.OK <= resp.status_code < HTTPStatus.MULTIPLE_CHOICES
        )
        if not should_prefix_model_ids:
            return None
        return self.anthropic_model_aliases.prefix_model_ids(resp.read())


def start_proxy(
    workspace: str,
    profile: str | None,
    port: int,
    token_header: str,
    force_refresh_near_expiry: bool,
):
    return gateway_proxy._start_proxy(
        workspace,
        profile,
        port,
        token_header,
        force_refresh_near_expiry,
        handler_class=_AnthropicModelDiscoveryHandler,
        handler_attributes={"anthropic_model_aliases": _AnthropicModelAliases()},
    )
