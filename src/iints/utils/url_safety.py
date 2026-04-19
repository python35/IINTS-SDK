from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlparse


def _is_localish_hostname(hostname: str) -> bool:
    lowered = hostname.lower()
    if lowered in {"localhost", "host.docker.internal"}:
        return True
    if lowered.endswith(".local") or lowered.endswith(".lan") or lowered.endswith(".internal"):
        return True
    try:
        parsed = ip_address(lowered)
    except ValueError:
        return False
    return parsed.is_loopback or parsed.is_private or parsed.is_link_local


def validate_service_base_url(raw_url: str, *, label: str) -> str:
    parsed = urlparse(raw_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{label} must use http or https.")
    if not parsed.hostname:
        raise ValueError(f"{label} must include a hostname.")
    if parsed.username or parsed.password:
        raise ValueError(f"{label} must not include embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{label} must not include query or fragment components.")
    if parsed.scheme == "http" and not _is_localish_hostname(parsed.hostname):
        raise ValueError(f"{label} must use https for non-local hosts.")
    normalized = raw_url.strip().rstrip("/")
    return normalized
