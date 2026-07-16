#!/usr/bin/env python3
"""Build a CA bundle for Harbor/Docker agent installs behind Netskope/corp MITM."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERT_DIR = ROOT / "certs"
CORP = CERT_DIR / "corp-ca-bundle.pem"
FULL = CERT_DIR / "docker-ca-bundle.pem"

KEEP_HINTS = (
    "netskope",
    "goskope",
    "mercedes",
    "daimler",
    "corp-root",
    "corp-issuing",
    "corp-proxy",
    "jss built",
)


def _dump_keychain(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        out = subprocess.run(
            ["security", "find-certificate", "-a", "-p", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout or ""
    except Exception:
        return ""


def extract_corp_certs() -> list[str]:
    blobs = []
    for kc in (
        Path("/Library/Keychains/System.keychain"),
        Path.home() / "Library/Keychains/login.keychain-db",
    ):
        blobs.append(_dump_keychain(kc))
    raw = "\n".join(blobs)
    certs = re.findall(
        r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
        raw,
        re.S,
    )
    keep: list[str] = []
    seen: set[str] = set()
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        for i, cert in enumerate(certs):
            pem = tdir / f"c{i}.pem"
            pem.write_text(cert + "\n")
            info = subprocess.run(
                ["openssl", "x509", "-in", str(pem), "-noout", "-subject", "-issuer"],
                capture_output=True,
                text=True,
            )
            text = (info.stdout + info.stderr).lower()
            if not any(h in text for h in KEEP_HINTS):
                continue
            # Dedup by subject+issuer
            key = text.strip()
            if key in seen:
                continue
            seen.add(key)
            keep.append(cert)
    return keep


def mozilla_bundle_via_docker() -> str:
    """Reuse Alpine ca-certificates as the public root set."""
    cmd = [
        "docker",
        "run",
        "--rm",
        "alpine:3.20",
        "sh",
        "-c",
        "apk add --no-cache ca-certificates >/dev/null && cat /etc/ssl/certs/ca-certificates.crt",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout


def main() -> int:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    corp = extract_corp_certs()
    CORP.write_text("\n".join(corp) + ("\n" if corp else ""))
    try:
        public = mozilla_bundle_via_docker()
    except Exception as e:
        print(f"WARN: could not fetch alpine CA bundle via docker: {e}")
        public = ""
    FULL.write_text(public + ("\n" if public and not public.endswith("\n") else "") + CORP.read_text())
    print(f"corp_certs={len(corp)} wrote {FULL} ({FULL.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
