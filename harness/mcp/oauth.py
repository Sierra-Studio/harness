"""OAuth 2.1 client for remote MCP servers (the MCP authorization flow).

Implements: protected-resource metadata discovery (RFC 9728), authorization-
server metadata discovery (RFC 8414), Dynamic Client Registration (RFC 7591),
Authorization Code grant with PKCE (S256) via a local browser redirect, token
caching on disk, and refresh-token renewal.

Interactive: on first use it opens a browser for the user to authorize, then
catches the redirect on a localhost callback. Tokens are cached under
~/.harness/mcp-auth/<host>.json so later runs are non-interactive.
"""

from __future__ import annotations

import base64
import contextlib
import errno
import hashlib
import http.server
import json
import re
import secrets
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def make_pkce() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) using the S256 method."""
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


def origin(url: str) -> str:
    u = urllib.parse.urlparse(url)
    return f"{u.scheme}://{u.netloc}"


def parse_resource_metadata_url(www_authenticate: str) -> str | None:
    """Extract resource_metadata="..." from a WWW-Authenticate header."""
    if not www_authenticate:
        return None
    m = re.search(r'resource_metadata="([^"]+)"', www_authenticate)
    return m.group(1) if m else None


@dataclass
class OAuthConfig:
    scopes: str = ""
    redirect_host: str = "127.0.0.1"
    redirect_port: int = 8765
    open_browser: bool = True
    client_name: str = "Harness MCP Client"
    timeout: float = 300  # seconds to wait for the browser redirect
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".harness" / "mcp-auth")


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        self.server.oauth_result = {k: v[0] for k, v in params.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h3>Authorization complete.</h3>"
            b"You can close this tab and return to the terminal.</body></html>"
        )

    def log_message(self, *args):  # silence
        pass


class OAuthClient:
    """Drives the OAuth flow for one MCP resource, with on-disk token cache."""

    def __init__(self, http_client, cfg: OAuthConfig | None = None):
        self.http = http_client  # an httpx.Client
        self.cfg = cfg or OAuthConfig()

    # ---------------- public ----------------
    def get_token(self, resource_url: str, www_authenticate: str = "") -> str:
        host = urllib.parse.urlparse(resource_url).netloc
        cache = self._load_cache(host)
        tokens = cache.get("tokens")

        if tokens and not self._expired(tokens):
            return tokens["access_token"]
        if tokens and tokens.get("refresh_token"):
            try:
                tokens = self._refresh(cache, resource_url)
                cache["tokens"] = tokens
                self._save_cache(host, cache)
                return tokens["access_token"]
            except Exception:
                pass  # fall through to a fresh interactive authorization

        tokens = self._authorize(resource_url, www_authenticate, cache)
        cache["tokens"] = tokens
        self._save_cache(host, cache)
        return tokens["access_token"]

    # ---------------- discovery ----------------
    def _discover(self, resource_url: str, www_authenticate: str) -> dict:
        org = origin(resource_url)
        prm = None
        for url in filter(
            None,
            [
                parse_resource_metadata_url(www_authenticate),
                org + "/.well-known/oauth-protected-resource",
            ],
        ):
            prm = self._get_json(url)
            if prm:
                break
        resource = (prm or {}).get("resource", resource_url)
        as_url = ((prm or {}).get("authorization_servers") or [org])[0]

        meta = None
        for url in [
            as_url.rstrip("/") + "/.well-known/oauth-authorization-server",
            as_url.rstrip("/") + "/.well-known/openid-configuration",
        ]:
            meta = self._get_json(url)
            if meta:
                break
        if not meta:
            raise RuntimeError(f"could not discover OAuth metadata for {resource_url}")
        meta["_resource"] = resource
        return meta

    def _register(self, meta: dict, cache: dict) -> tuple[str, str | None]:
        if cache.get("client_id"):
            return cache["client_id"], cache.get("client_secret")
        reg_ep = meta.get("registration_endpoint")
        if not reg_ep:
            raise RuntimeError(
                "authorization server has no registration_endpoint; set a client_id out of band"
            )
        body = {
            "client_name": self.cfg.client_name,
            "redirect_uris": [self._redirect_uri()],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }
        if self.cfg.scopes:
            body["scope"] = self.cfg.scopes
        r = self.http.post(reg_ep, json=body)
        r.raise_for_status()
        data = r.json()
        cache["client_id"] = data["client_id"]
        cache["client_secret"] = data.get("client_secret")
        return cache["client_id"], cache.get("client_secret")

    # ---------------- authorization code + PKCE ----------------
    def _authorize(self, resource_url: str, www_authenticate: str, cache: dict) -> dict:
        meta = self._discover(resource_url, www_authenticate)
        client_id, client_secret = self._register(meta, cache)
        verifier, challenge = make_pkce()
        state = secrets.token_urlsafe(16)
        redirect_uri = self._redirect_uri()

        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "resource": meta["_resource"],
        }
        if self.cfg.scopes:
            params["scope"] = self.cfg.scopes
        auth_url = meta["authorization_endpoint"] + "?" + urllib.parse.urlencode(params)
        code = self._await_redirect(auth_url, state)

        token_req = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
            "resource": meta["_resource"],
        }
        if client_secret:
            token_req["client_secret"] = client_secret
        r = self.http.post(meta["token_endpoint"], data=token_req)
        r.raise_for_status()
        tok = r.json()
        tok["obtained_at"] = int(time.time())
        cache["token_endpoint"] = meta["token_endpoint"]
        cache["resource"] = meta["_resource"]
        return tok

    def _refresh(self, cache: dict, resource_url: str) -> dict:
        body = {
            "grant_type": "refresh_token",
            "refresh_token": cache["tokens"]["refresh_token"],
            "client_id": cache["client_id"],
            "resource": cache.get("resource", resource_url),
        }
        if cache.get("client_secret"):
            body["client_secret"] = cache["client_secret"]
        r = self.http.post(cache["token_endpoint"], data=body)
        r.raise_for_status()
        tok = r.json()
        tok["obtained_at"] = int(time.time())
        tok.setdefault("refresh_token", cache["tokens"].get("refresh_token"))
        return tok

    def _bind_callback_server(self) -> http.server.HTTPServer:
        """Bind the loopback callback server, tolerating a socket left in
        TIME_WAIT by a just-finished flow. The redirect URI is registered with a
        fixed port, so we cannot fall back to an ephemeral one — instead retry
        briefly, then fail with an actionable message."""
        addr = (self.cfg.redirect_host, self.cfg.redirect_port)
        last: OSError | None = None
        for attempt in range(5):
            try:
                # allow_reuse_address (SO_REUSEADDR) is set by HTTPServer; this
                # lets us rebind a port whose previous socket is in TIME_WAIT.
                return http.server.HTTPServer(addr, _CallbackHandler)
            except OSError as e:
                last = e
                if e.errno != errno.EADDRINUSE:
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            f"OAuth callback port {self.cfg.redirect_host}:{self.cfg.redirect_port} "
            f"is already in use. Another authorization may be in progress, or a "
            f"stale process is holding it — close it and try again."
        ) from last

    def _await_redirect(self, auth_url: str, state: str) -> str:
        server = self._bind_callback_server()
        server.oauth_result = None  # type: ignore[attr-defined]
        server.timeout = 1
        if self.cfg.open_browser:
            webbrowser.open(auth_url)
        print(f"\n[oauth] Authorize in your browser; if it didn't open, visit:\n{auth_url}\n")
        deadline = time.time() + self.cfg.timeout
        try:
            while server.oauth_result is None and time.time() < deadline:  # type: ignore[attr-defined]
                server.handle_request()
        finally:
            server.server_close()
        result = server.oauth_result  # type: ignore[attr-defined]
        if not result:
            raise RuntimeError("OAuth timed out waiting for the browser redirect")
        if result.get("error"):
            raise RuntimeError(
                f"OAuth error: {result['error']} {result.get('error_description', '')}"
            )
        if result.get("state") != state:
            raise RuntimeError("OAuth state mismatch (possible CSRF)")
        if "code" not in result:
            raise RuntimeError("OAuth redirect had no authorization code")
        return result["code"]

    # ---------------- helpers ----------------
    def _redirect_uri(self) -> str:
        return f"http://{self.cfg.redirect_host}:{self.cfg.redirect_port}/callback"

    def _get_json(self, url: str) -> dict | None:
        try:
            r = self.http.get(url)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        return None

    @staticmethod
    def _expired(tokens: dict) -> bool:
        exp = tokens.get("expires_in")
        if not exp:
            return False
        return time.time() >= tokens.get("obtained_at", 0) + exp - 30

    def _cache_path(self, host: str) -> Path:
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cfg.cache_dir / f"{host.replace(':', '_')}.json"

    def _load_cache(self, host: str) -> dict:
        p = self._cache_path(host)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return {}
        return {}

    def _save_cache(self, host: str, data: dict) -> None:
        path = self._cache_path(host)
        path.write_text(json.dumps(data))
        with contextlib.suppress(OSError):
            path.chmod(0o600)  # tokens are secrets
