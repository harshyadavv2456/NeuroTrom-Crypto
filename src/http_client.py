"""HTTP client with a safe macOS curl fallback for public market-data APIs.

Requests is attempted first. If Python's TLS/network path is reset while the system
curl succeeds, retry using the system curl binary without disabling TLS verification.
HTTP error responses are never retried through a different transport.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from datetime import timedelta
from urllib.parse import urlencode, urlsplit, urlunsplit

import requests
from requests.structures import CaseInsensitiveDict

log = logging.getLogger(__name__)
_MARKER = "\n__NEUROTROM_CRYPTO_HTTP_STATUS__:"


def _curl_get(url: str, params: dict | None, timeout: float, headers: dict | None,
              original_error: Exception) -> requests.Response:
    curl = shutil.which("curl")
    if not curl:
        raise original_error

    parts = urlsplit(url)
    query = urlencode(params or {}, doseq=True)
    merged_query = "&".join(part for part in (parts.query, query) if part)
    full_url = urlunsplit((parts.scheme, parts.netloc, parts.path, merged_query, parts.fragment))

    cmd = [
        curl, "--silent", "--show-error", "--location", "--max-redirs", "3",
        "--connect-timeout", str(max(1, min(float(timeout), 5.0))),
        "--max-time", str(float(timeout)),
        "--user-agent", "NeuroTrom-Crypto/0.1 (public market-data client)",
        "--header", "Accept: application/json",
        "--write-out", _MARKER + "%{http_code}",
    ]
    for key, value in (headers or {}).items():
        cmd.extend(["--header", f"{key}: {value}"])
    cmd.append(full_url)

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=float(timeout) + 2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise requests.ConnectionError(
            f"Python requests failed ({original_error}); system curl failed ({exc})"
        ) from exc

    body = proc.stdout
    status = 0
    if _MARKER in body:
        body, status_text = body.rsplit(_MARKER, 1)
        try:
            status = int(status_text.strip())
        except ValueError:
            status = 0

    if proc.returncode != 0 and status == 0:
        detail = proc.stderr.strip()[-500:] or f"curl exit code {proc.returncode}"
        raise requests.ConnectionError(
            f"Python requests failed ({original_error}); system curl failed ({detail})"
        )

    response = requests.Response()
    response.status_code = status
    response._content = body.encode("utf-8")
    response.url = full_url
    response.elapsed = timedelta(seconds=max(0.0, time.monotonic() - started))
    response.headers = CaseInsensitiveDict({
        "Content-Type": "application/json",
        "X-NeuroTrom-Transport": "curl-fallback",
    })
    if status < 200 or status >= 300:
        response.raise_for_status()
    return response


def http_get(url: str, params: dict | None = None, timeout: float = 15,
             headers: dict | None = None) -> requests.Response:
    """GET with requests first and system-curl fallback only for TLS/connection failures."""
    try:
        response = requests.get(url, params=params, timeout=timeout, headers=headers)
        # Raise HTTP responses here; rate limits and 4xx/5xx are not transport errors
        # and should not be retried with another client.
        response.raise_for_status()
        return response
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError,
            requests.exceptions.Timeout) as exc:
        log.warning("Python HTTP transport failed for %s (%s); trying system curl", url, exc)
        return _curl_get(url, params, timeout, headers, exc)
