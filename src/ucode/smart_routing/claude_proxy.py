"""Model-routing proxy for Claude Code (smart routing v2).

Claude Code has no ``app-server``/JSON-RPC seam like Codex, and — as
``claude_routing.route_launch_model`` notes — "no hook/MCP can retarget the root
model once the session is running." But every turn Claude Code makes an Anthropic
Messages API request to ``ANTHROPIC_BASE_URL``, and that request body carries BOTH
the prompt and the ``model``. So the routing seam is the HTTP request: point
``ANTHROPIC_BASE_URL`` at this loopback proxy, and it reads the prompt out of each
request, picks a model, rewrites the body's ``model`` field, and forwards to the
workspace gateway. This is the direct analog of the Codex interposer (which rewrites
``turn/start.model`` on the WebSocket) — just one layer down, at the HTTP request.

Because it routes below Claude Code:
  - the FIRST prompt is routed correctly (the request carries it),
  - every turn can route independently,
  - Claude Code never has to know the target model exists (no ``/model``, no gateway
    model discovery, no mutated default), and
  - nothing scrapes or drives the TUI.

Auth is passthrough: in the normal (non-relayed) launch Claude Code's ``apiKeyHelper``
already mints the gateway credential and sends it, so — unlike ``gateway_proxy`` — this
proxy needs no token management. It forwards headers verbatim (minus hop-by-hop) and
streams the response back byte-for-byte (SSE token streaming is not buffered).

Security: binds 127.0.0.1 only; never logs header values or bodies (the routing log
records only the model ids swapped, never prompt text).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

# Hop-by-hop headers must not be forwarded across the proxy; content-length is dropped
# too because we may rewrite the body (httpx recomputes it from the content we pass).
_HOP_BY_HOP = frozenset(
    h.lower()
    for h in (
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    )
)
# Generous read timeout: a turn streams over one response with SSE pings between chunks.
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=10.0)

# A router maps a parsed request body to the model id to use. The stub ignores the body
# and always returns the fixed pick; a real router inspects body["messages"].
RouteFn = Callable[[dict], str]


def rewrite_model(
    body: bytes, *, is_json: bool, route: RouteFn
) -> tuple[bytes, str | None, str | None]:
    """Rewrite the ``model`` field of a Messages API request body.

    Returns ``(new_body, original_model, chosen_model)``. When the body is not a JSON
    object with a ``model`` key (or the router picks the same model), ``new_body`` is the
    input unchanged. Pure and side-effect-free, so it is unit-testable without a server.
    """
    if not body or not is_json:
        return body, None, None
    try:
        parsed = json.loads(body)
    except ValueError:
        return body, None, None
    if not isinstance(parsed, dict) or "model" not in parsed:
        return body, None, None
    original = parsed.get("model")
    chosen = route(parsed)
    if not chosen or chosen == original:
        return body, original if isinstance(original, str) else None, chosen
    parsed["model"] = chosen
    return json.dumps(parsed).encode(), original if isinstance(original, str) else None, chosen


def _is_json_request(handler: BaseHTTPRequestHandler) -> bool:
    """True when the request body is plain (uncompressed) JSON we can safely rewrite."""
    if "content-encoding" in {k.lower() for k in handler.headers}:
        return False  # compressed body: don't touch it
    return "json" in handler.headers.get("Content-Type", "").lower()


def _forwarded_headers(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    return {k: v for k, v in handler.headers.items() if k.lower() not in _HOP_BY_HOP}


class _RouterProxyHandler(BaseHTTPRequestHandler):
    # Set by the server factory.
    client: httpx.Client
    route: RouteFn
    log_fn: Callable[[str], None]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _safe_send_error(self, code: int, message: str) -> None:
        try:
            self.send_error(code, message)
        except OSError:
            pass

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        url = self.path.lstrip("/")

        new_body = body
        original = chosen = None
        if body and self.command == "POST":
            new_body, original, chosen = rewrite_model(
                body, is_json=_is_json_request(self), route=self.route
            )
        # One line per request (model ids + path only, never prompt text) so the log shows
        # ALL traffic — confirming requests reach the proxy, not just rewrites.
        action = "ROUTE" if (chosen and chosen != original) else "PASS"
        self.log_fn(f"[{action}] {self.command} /{url} model={original!r}->{chosen!r}")

        try:
            with self.client.stream(
                self.command, url, headers=_forwarded_headers(self), content=new_body or None
            ) as resp:
                self.log_fn(f"[UPSTREAM] /{url} {resp.status_code}")
                self._relay_response(resp)
        except (BrokenPipeError, ConnectionResetError):
            return  # client (Claude Code) closed before/while relaying — routine on cancel
        except httpx.HTTPError as exc:
            self.log_fn(f"[ERR] /{url} {type(exc).__name__}: {exc}")
            self._safe_send_error(502, "model router proxy upstream error")

    def _relay_response(self, resp: httpx.Response) -> None:
        # Stream chunks as they arrive so SSE token streaming is not buffered; iter_raw
        # preserves any Content-Encoding verbatim (we relay that header).
        try:
            self.send_response(resp.status_code)
            for key, value in resp.headers.items():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, value)
            self.end_headers()
            for chunk in resp.iter_raw():
                if chunk:
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # client closed mid-response — routine on cancelled turns / SSE teardown
        except httpx.HTTPError:
            return  # upstream dropped mid-stream; headers already sent, stop cleanly

    # Transparent pass-through for every method (GET model discovery, POST messages, …).
    def __getattr__(self, name: str):
        if name.startswith("do_"):
            return self._handle
        raise AttributeError(name)


def start_proxy(
    workspace: str,
    route: RouteFn,
    *,
    log_path: Path | None = None,
) -> tuple[ThreadingHTTPServer, httpx.Client, int]:
    """Start the loopback model-routing proxy on an OS-assigned port.

    Forwards to ``{workspace}/ai-gateway/anthropic/`` with headers passed through
    verbatim (Claude Code's apiKeyHelper credential included). Returns
    ``(server, client, port)``; the caller runs ``server`` (e.g. in a thread) and calls
    ``server.shutdown()`` / ``client.close()`` on exit. Point ``ANTHROPIC_BASE_URL`` at
    ``http://127.0.0.1:{port}``.
    """

    def log(message: str) -> None:
        if log_path is None:
            return
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")
        except OSError:
            pass

    upstream_base = f"{workspace.rstrip('/')}/ai-gateway/anthropic/"
    client = httpx.Client(base_url=upstream_base, timeout=_UPSTREAM_TIMEOUT, follow_redirects=False)
    handler = type(
        "BoundRouterProxyHandler",
        (_RouterProxyHandler,),
        {"client": client, "route": staticmethod(route), "log_fn": staticmethod(log)},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    log(f"[READY] 127.0.0.1:{port} -> {upstream_base}")
    return server, client, port
