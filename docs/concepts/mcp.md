# MCP

`harness/mcp/client.py` provides two transports behind the same duck-typed
interface (`name`, `list_tools()`, `call_tool()`): `McpClient` for local stdio
servers and `HttpMcpClient` for remote Streamable-HTTP servers.

Discovered tools are written into `tool_index` (**not** the system prompt), so
an arbitrary number of MCP tools never crowds out prompt budget. The model
finds them via `SearchTools` (keyword/full-text search over `tool_index`) and
dispatches with `CallTool`.

## Connecting a server

Two one-line helpers on `Harness`, both index the server's tools and enable
dispatch:

```python
h = Harness()

# local stdio server (subprocess)
h.add_mcp_stdio(["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                name="fs")

# remote Streamable-HTTP server — static bearer token...
h.add_mcp_http("https://example.com/mcp", name="example",
               headers={"Authorization": f"Bearer {token}"})
# ...or the interactive OAuth 2.1 flow (browser login + PKCE, token cached)
h.add_mcp_http("https://example.com/mcp", name="example", oauth=True)
```

See `examples/add_fellow_mcp.py` for a full example.

## Exposure policy: index vs. direct

Via the `ToolProvider` API (`harness/tools/capabilities.py`), `McpServer`/`McpStdioServer`
take an `expose` keyword:

- **`"index"`** (default) — the discovery model above: the server is indexed
  into `tool_index`, reached via `SearchTools`/`CallTool`. Right default for a
  large catalog you can't dump into context.
- **`"direct"`** — each of the server's tools is *also* registered as a
  first-class `McpProxyTool`: its schema is sent to the model directly and its
  guidance is composed into the prompt automatically. Better for a small,
  always-relevant server (e.g. a 6-tool sandbox) where the discovery hop is
  pure friction.

```python
h.add_mcp_http(sandbox_url, "sandbox", expose="direct")
```

See [ADR 0001](../adr/0001-mcp-exposure-and-runtime-lifecycle.md) for the full
design rationale, including the proposed `HarnessRuntime` split for
stateless-worker deployments and the MCP client concurrency contract.

## From the CLI (env-driven)

Remote HTTP servers are auto-connected from the environment — no code:

```bash
# .env
MCP_HTTP_SERVERS=fellow=https://fellow.app/mcp   # comma-separated name=url
MCP_FELLOW_OAUTH=1                                # browser OAuth (PKCE, token cached)
# MCP_FELLOW_TOKEN=...                            # or a static bearer instead
```

`uv run harness chat` connects each one at startup (failures are reported and
skipped) and disconnects them on exit. OAuth tokens are cached under
`~/.harness/mcp-auth/`, so the browser login happens only once.

## OAuth 2.1

`harness/mcp/oauth.py` implements the full MCP authorization flow: protected-resource
metadata discovery (RFC 9728), authorization-server metadata discovery
(RFC 8414), Dynamic Client Registration (RFC 7591), Authorization Code + PKCE
(S256) via a local browser redirect, on-disk token caching, and refresh-token
renewal. It's interactive on first use (opens a browser); this doesn't work in
a headless/ephemeral container — prefer a static bearer token there.
