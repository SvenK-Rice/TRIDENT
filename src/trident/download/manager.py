from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from typing import Optional

import requests

try:
    import certifi
except Exception:  # pragma: no cover
    certifi = None


@dataclass
class DownloadResult:
    url: str
    method: str
    n_bytes: int
    warning: Optional[str] = None


class DownloadManager:
    """Robust HTTP download helper used by all TRIDENT data sources.

    Strategy:
    1. Try normal verified HTTPS with requests.
    2. Try certifi CA bundle if available.
    3. Try system curl with verified TLS.
    4. Last resort: curl -k, with a controlled warning instead of urllib3 spam.

    The last fallback is necessary for some macOS/Python installations where
    Python cannot validate an otherwise reachable institutional archive.
    """

    def __init__(self, timeout: int = 120, retries: int = 2):
        self.timeout = timeout
        self.retries = retries
        self.last_result: Optional[DownloadResult] = None

    def _requests_get(self, url: str, verify):
        headers = {"User-Agent": "TRIDENT-NPP-Workbench/3.1"}
        r = requests.get(url, timeout=self.timeout, verify=verify, headers=headers)
        r.raise_for_status()
        return r.content

    def fetch_bytes(self, url: str) -> bytes:
        errors = []
        attempts = []
        attempts.append(("requests-verified", True))
        if certifi is not None:
            attempts.append(("requests-certifi", certifi.where()))

        for method, verify in attempts:
            for i in range(self.retries + 1):
                try:
                    data = self._requests_get(url, verify=verify)
                    self.last_result = DownloadResult(url=url, method=method, n_bytes=len(data))
                    return data
                except Exception as e:
                    errors.append(f"{method}: {e}")
                    time.sleep(0.25 * (i + 1))

        # Prefer curl with normal verification because macOS curl often has a
        # working system certificate store even when Python does not.
        try:
            data = subprocess.check_output(["curl", "-L", "--fail", "--silent", "--show-error", url], timeout=max(self.timeout, 240))
            self.last_result = DownloadResult(url=url, method="curl-verified", n_bytes=len(data))
            return data
        except Exception as e:
            errors.append(f"curl-verified: {e}")

        # Last resort: curl -k. This avoids urllib3's repeated warnings while
        # still recording that insecure TLS fallback was required.
        try:
            data = subprocess.check_output(["curl", "-k", "-L", "--fail", "--silent", "--show-error", url], timeout=max(self.timeout, 240))
            warning = "Secure TLS verification failed; used curl -k fallback for this OSU archive request."
            self.last_result = DownloadResult(url=url, method="curl-insecure", n_bytes=len(data), warning=warning)
            return data
        except Exception as e:
            errors.append(f"curl-insecure: {e}")
            raise RuntimeError("All download methods failed for " + url + "\n" + "\n".join(errors))

    def fetch_text(self, url: str, encoding: str = "latin1") -> str:
        return self.fetch_bytes(url).decode(encoding, errors="replace")

    def download_to_file(self, url: str, out: Path, *, min_bytes: int = 1) -> Path:
        out = Path(out)
        if out.exists() and out.stat().st_size >= min_bytes:
            self.last_result = DownloadResult(url=url, method="cache", n_bytes=out.stat().st_size)
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self.fetch_bytes(url)
        if len(data) < min_bytes:
            raise RuntimeError(f"Downloaded file from {url} is unexpectedly small: {len(data)} bytes")
        tmp = out.with_suffix(out.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(out)
        return out


def download_bytes(url: str) -> bytes:
    return DownloadManager().fetch_bytes(url)


def download_text(url: str) -> str:
    return DownloadManager().fetch_text(url)
