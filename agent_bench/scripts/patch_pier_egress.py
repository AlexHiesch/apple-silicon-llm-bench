#!/usr/bin/env python3
"""Patch Pier's Squid egress policy for local ThinkingCap.

DeepSWE tasks set allow_internet=false, so Pier forces traffic through Squid.
Upstream Squid only allows destination ports 80/443, and Docker Desktop
registers host.docker.internal on an unreachable IPv6 as well as IPv4.

This patch (idempotent):

1. Adds Safe_ports 8080/8091 (Kevlar + openai_anthropic_shim)
2. Pins host.docker.internal to its IPv4 in /etc/hosts before Squid starts
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "# llm-bench ThinkingCap egress patch"

HOSTS_FIX = r"""
# llm-bench ThinkingCap egress patch: Docker Desktop publishes an unreachable
# IPv6 for host.docker.internal; force IPv4 so Squid can reach the host.
# (sed -i cannot rewrite Docker-mounted /etc/hosts; overwrite via cat instead.)
if command -v getent >/dev/null 2>&1; then
  IPV4=$(getent ahostsv4 host.docker.internal 2>/dev/null | awk '{print $1; exit}')
  if [ -n "${IPV4:-}" ]; then
    grep -v host.docker.internal /etc/hosts > /tmp/hosts.fixed || true
    printf '%s host.docker.internal\n' "$IPV4" >> /tmp/hosts.fixed
    cat /tmp/hosts.fixed > /etc/hosts
  fi
fi
"""

SAFE_PORTS_OLD = "acl Safe_ports port 80 443"
SAFE_PORTS_NEW = (
    f"{MARKER}\n"
    "acl Safe_ports port 80 443\n"
    "acl Safe_ports port 8080\n"
    "acl Safe_ports port 8091"
)


def pier_agent_setup() -> Path:
    # Prefer the pier used by the `pier` CLI (uv tool env). On the Z8 that
    # lives under ~/aa-index-runner-home (see bootstrap_aa_index_host.sh),
    # not ~/.local.
    import os

    home = Path.home()
    runner = Path(os.environ.get("RUNNER_HOME", home / "aa-index-runner-home"))
    tool_dir = Path(
        os.environ.get(
            "UV_TOOL_DIR",
            runner / ".local" / "share" / "uv" / "tools",
        )
    )
    candidates = [
        tool_dir
        / "datacurve-pier/lib/python3.12/site-packages"
        / "pier/environments/agent_setup.py",
        tool_dir
        / "datacurve-pier/lib/python3.13/site-packages"
        / "pier/environments/agent_setup.py",
        home
        / ".local/share/uv/tools/datacurve-pier/lib/python3.12/site-packages"
        / "pier/environments/agent_setup.py",
        home
        / ".local/share/uv/tools/datacurve-pier/lib/python3.13/site-packages"
        / "pier/environments/agent_setup.py",
    ]
    for c in candidates:
        if c.is_file():
            return c
    import pier.environments.agent_setup as mod

    return Path(mod.__file__)


def patch_text(src: str) -> str:
    out = src
    if SAFE_PORTS_OLD in out and "acl Safe_ports port 8080" not in out:
        out = out.replace(SAFE_PORTS_OLD, SAFE_PORTS_NEW, 1)
    # Insert hosts fix after `set -eu` inside the bootstrap script.
    if "tmp/hosts.fixed" not in out:
        needle = "set -eu\n"
        if needle not in out:
            raise SystemExit("Expected 'set -eu' in squid_bootstrap_command")
        # Only patch the bootstrap script's set -eu (first occurrence in return r"""...""")
        idx = out.index(needle)
        out = out[: idx + len(needle)] + HOSTS_FIX + out[idx + len(needle) :]
    return out


def main() -> int:
    path = pier_agent_setup()
    original = path.read_text()
    updated = patch_text(original)
    if updated == original:
        print(f"already patched: {path}")
        return 0
    path.write_text(updated)
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
