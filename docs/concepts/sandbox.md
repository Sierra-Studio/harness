# Sandbox

`harness/tools/sandbox.py` is the execution backend behind the `Bash` tool.
`SandboxBackend` is the pluggable contract:

```python
class SandboxBackend(abc.ABC):
    def exec(self, session_id: str, command: str, timeout: int = 60) -> ExecResult: ...
    def destroy(self, session_id: str) -> None: ...
```

`LocalSubprocessSandbox` is the bundled implementation: one working directory
per session, per-command timeout, output capped with head/tail elision — and,
unlike a naive subprocess wrapper, the working directory **persists across
calls** within a session (a `cd` in one call carries to the next).

!!! danger "Not isolated"
    `LocalSubprocessSandbox` runs commands as a plain local subprocess — it is
    **not** kernel-isolated. Replace it with a gVisor/Firecracker/Kubernetes-backed
    implementation before exposing untrusted, multi-tenant Bash:

```python
Harness(cfg, sandbox=FirecrackerSandbox(...))
```

See [ADR 0001](../adr/0001-mcp-exposure-and-runtime-lifecycle.md) for the
related pattern of exposing a sandbox's control plane as an MCP server
(`expose="direct"`) so the model can call it without a discovery hop.
