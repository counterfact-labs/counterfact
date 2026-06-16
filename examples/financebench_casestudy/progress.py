"""Tiny throttled progress + ETA reporter (stderr).

Thread-safe so parallel per-query diagnoses can share one counter. Prints at most
once every `every` seconds (plus a final line at completion), so it is safe to call
on every simulation without flooding the log.
"""
from __future__ import annotations

import sys
import threading
import time


class Progress:
    def __init__(self, total: int, label: str, every: float = 5.0):
        self.total = max(1, total)
        self.label = label
        self.every = every
        self.n = 0
        self._t0 = time.monotonic()
        self._last = 0.0
        self._lock = threading.Lock()

    def tick(self, inc: int = 1, status: str = "") -> None:
        with self._lock:
            self.n += inc
            now = time.monotonic()
            done = self.n >= self.total
            if now - self._last < self.every and not done:
                return
            self._last = now
            elapsed = now - self._t0
            rate = self.n / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self.n) / rate if rate > 0 else 0.0
            pct = 100.0 * self.n / self.total
            print(f"  [{self.label}] {self.n}/{self.total} ({pct:.0f}%) | "
                  f"elapsed {_fmt(elapsed)} | ETA {_fmt(eta)}{(' | ' + status) if status else ''}",
                  file=sys.stderr, flush=True)

    def done(self, status: str = "") -> None:
        with self._lock:
            elapsed = time.monotonic() - self._t0
            print(f"  [{self.label}] done {self.n}/{self.total} in {_fmt(elapsed)}"
                  f"{(' | ' + status) if status else ''}", file=sys.stderr, flush=True)


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"
